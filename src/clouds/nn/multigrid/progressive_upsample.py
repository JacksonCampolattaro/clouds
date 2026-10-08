from torch.nn import ModuleDict

from ...data import MultiGridData
from ...utils import expand_multigrid_kwargs
from ...utils.sequence import _normalize_list
from ..apply import apply_to_kwargs
from ..merge import CatMerge


class ProgressiveUpsample(ModuleDict):
    def __init__(
        self,
        upsample: type,
        in_channels: list[int | None],
        merge: type = CatMerge,
        conv: type | list[type] | None = None,
        upsample_conv_kwargs: dict | None = None,
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
        upsample_level_kwargs = expand_multigrid_kwargs(levels=levels, **(kwargs | (upsample_conv_kwargs or {})))
        assert len(upsample_level_kwargs) == len(in_channels)

        carry_channels = in_channels[-1]
        for coarse_level in reversed(levels):
            fine_level = coarse_level - 1

            # Create convolution (optional)
            if (c := conv[coarse_level]) and (c_kwargs := conv_level_kwargs.get(coarse_level)):
                self[f'x{coarse_level}__to__x{coarse_level}'] = c = c(
                    in_channels=carry_channels,
                    **c_kwargs,
                )
                carry_channels = self.out_channels[coarse_level] = c.out_channels

            # Create upsampler (only where fine_level actually exists)
            if u_kwargs := upsample_level_kwargs.get(fine_level):
                self[f'x{coarse_level}__to__x{fine_level}'] = u = upsample(
                    in_channels=carry_channels,
                    **u_kwargs,
                )
                carry_channels = self.out_channels[fine_level] = u.out_channels

                # Create merger (if there are channels to merge with on the coarse level)
                if merge and self.in_channels[fine_level]:
                    self[f"merge__x{fine_level}"] = m = merge([self.in_channels[fine_level], carry_channels])

                    carry_channels = m.out_channels
                    if self.return_merged_values:
                        self.out_channels[fine_level] = m.out_channels

        if (not self.return_merged_values) and (not conv[-1]):
            self.out_channels[-1] = 0

    def forward(self, data: MultiGridData) -> MultiGridData:
        scales = data.node_types
        coarsest = scales[-1]

        out = MultiGridData.from_dict(data.to_dict())
        carry = out.get_level(coarsest)
        for coarse, fine in reversed(list(zip(scales, [None, *scales[:-1]], strict=True))):
            # Apply convolution (if present for this scale)
            if conv := getattr(self, f'{coarse}__to__{coarse}', None):
                carry.x = out[coarse].x = apply_to_kwargs(
                    conv,
                    **(carry.to_dict() | data[coarse, 'to', coarse].to_dict()),
                    size=(carry.num_nodes, carry.num_nodes),
                )

            # Apply upsampling (if present for this scale)
            if upsample := getattr(self, f'{coarse}__to__{fine}', None):
                out[fine].x = apply_to_kwargs(
                    upsample,
                    **({k: (carry[k], data[fine].get(k, None)) for k in carry.keys()} | data[coarse, 'to', fine].to_dict()),  # noqa: SIM118
                    size=(carry.num_nodes, data[fine].num_nodes),
                )
                carry = out.get_level(fine)

                # Merge if necessary
                if merge := getattr(self, f'merge__{fine}', None):
                    carry.x = merge(data[fine].x, carry.x)

                    # (Optionally) write merged values to output
                    if self.return_merged_values:
                        out[fine].x = carry.x

        if (not self.return_merged_values) and (f'{coarsest}__to__{coarsest}' not in self):
            # If we're not returning merged values and we don't have a conv,
            # we shouldn't have an output on the coarsest scale
            del out[coarsest].x

        return out
