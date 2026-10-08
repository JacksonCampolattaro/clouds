from torch.nn import ModuleDict

from ...data import MultiGridData
from ...utils import expand_multigrid_kwargs
from ...utils.sequence import _normalize_list
from ..apply import apply_to_kwargs
from ..merge import CatMerge


class ProgressiveDownsample(ModuleDict):
    def __init__(
        self,
        downsample: type,
        in_channels: list[int | None],
        merge: type = CatMerge,
        conv: type | list[type] | None = None,
        downsample_conv_kwargs: dict | None = None,
        conv_kwargs: dict | None = None,
        return_merged_values: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        levels = list(range(len(in_channels)))
        self.in_channels, self.out_channels = in_channels, in_channels.copy()
        self.return_merged_values = return_merged_values

        conv = list(_normalize_list(conv, len(levels)))

        # Broadcast kwargs across levels
        conv_level_kwargs = expand_multigrid_kwargs(levels=levels, **(kwargs | (conv_kwargs or {})))
        downsample_level_kwargs = expand_multigrid_kwargs(levels=levels, **(kwargs | (downsample_conv_kwargs or {})))
        assert len(downsample_level_kwargs) == len(in_channels)

        carry_channels = in_channels[0]
        for fine_level in levels:
            coarse_level = fine_level + 1

            # Create convolution (optional)
            if (c := conv[fine_level]) and (c_kwargs := conv_level_kwargs.get(fine_level)):
                self[f'x{fine_level}__to__x{fine_level}'] = c = c(
                    in_channels=carry_channels,
                    **c_kwargs,
                )
                carry_channels = self.out_channels[fine_level] = c.out_channels

            # Create downsampler (only where coarse_level actually exists)
            if d_kwargs := downsample_level_kwargs.get(coarse_level):
                self[f'x{fine_level}__to__x{coarse_level}'] = d = downsample(
                    in_channels=carry_channels,
                    **d_kwargs,
                )
                carry_channels = self.out_channels[coarse_level] = d.out_channels

                # Create merger (if there are channels to merge with on the coarse level)
                if merge and self.in_channels[coarse_level]:
                    self[f"merge__x{coarse_level}"] = m = merge([self.in_channels[coarse_level], carry_channels])

                    carry_channels = m.out_channels
                    if self.return_merged_values:
                        self.out_channels[coarse_level] = m.out_channels

        if (not self.return_merged_values) and (not conv[0]):
            self.out_channels[0] = 0

    def forward(self, data: MultiGridData) -> MultiGridData:
        scales = data.node_types
        finest = scales[0]

        out = MultiGridData.from_dict(data.to_dict())
        carry = out.get_level(finest)
        for fine, coarse in zip(scales, [*scales[1:], None], strict=True):
            # Apply convolution (if present for this scale)
            if conv := getattr(self, f'{fine}__to__{fine}', None):
                carry.x = out[fine].x = apply_to_kwargs(
                    conv,
                    **(carry.to_dict() | data[fine, 'to', fine].to_dict()),
                    size=(carry.num_nodes, carry.num_nodes),
                )

            # Apply downsampling (if present for this scale)
            if downsample := getattr(self, f'{fine}__to__{coarse}', None):
                out[coarse].x = apply_to_kwargs(
                    downsample,
                    **({k: (carry[k], data[coarse].get(k, None)) for k in carry.keys()} | data[fine, 'to', coarse].to_dict()),  # noqa: SIM118
                    size=(carry.num_nodes, data[coarse].num_nodes),
                )
                carry = out.get_level(coarse)

                # Merge if necessary
                if merge := getattr(self, f'merge__{coarse}', None):
                    carry.x = merge(data[coarse].x, carry.x)

                    # (Optionally) write merged values to output
                    if self.return_merged_values:
                        out[coarse].x = carry.x

        if (not self.return_merged_values) and (f'{finest}__to__{finest}' not in self):
            # If we're not returning merged values and we don't have a conv,
            # we shouldn't have an output on the finest scale
            del out[finest].x

        return out
