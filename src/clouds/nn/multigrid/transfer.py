from ..merge import SumMerge
from ..residual import Identity
from ..sequential import DataSequential
from .multigrid import MultiGridModule
from .parallel import MultiGridParallel
from .progressive_downsample import ProgressiveDownsample
from .progressive_upsample import ProgressiveUpsample


class ParallelTransfer(MultiGridParallel):
    def __init__(
        self,
        in_channels: list[int | None],
        out_channels: list[int] | None = None,
        downsample_conv: type | None = None,
        downsample_kwargs: dict | None = None,
        upsample_conv: type | None = None,
        upsample_kwargs: dict | None = None,
        dropout: type | None = None,
        dropout_p: float = 0.0,
        merge: type = SumMerge,
        merge_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        transfer_modules = [
            ('id', MultiGridModule(Identity, in_channels=in_channels)),
        ]
        if upsample_conv:
            up = ProgressiveUpsample(
                upsample_conv,
                in_channels=in_channels,
                out_channels=out_channels,
                merge=SumMerge,
                return_merged_values=False,
                **(kwargs | (upsample_kwargs or {})),
            )
            if dropout and dropout_p:
                up = DataSequential(
                    up,
                    MultiGridModule(
                        dropout,
                        in_channels=up.out_channels,
                        levels=[i for i, c in enumerate(up.out_channels) if c],
                        p=dropout_p,
                    ),
                )

            transfer_modules.append(('up', up))

        if downsample_conv:
            down = ProgressiveDownsample(
                downsample_conv,
                in_channels=in_channels,
                out_channels=out_channels,
                merge=SumMerge,
                return_merged_values=False,
                **(kwargs | (downsample_kwargs or {})),
            )
            if dropout and dropout_p:
                down = DataSequential(
                    down,
                    MultiGridModule(
                        dropout,
                        in_channels=down.out_channels,
                        levels=[i for i, c in enumerate(down.out_channels) if c],
                        p=dropout_p,
                    ),
                )

            transfer_modules.append(('down', down))

        super().__init__(
            *transfer_modules,
            merge=merge,
            merge_kwargs=merge_kwargs,
        )


class SequentialTransfer(DataSequential):
    def __init__(
        self,
        in_channels: list[int | None],
        out_channels: list[int] | None = None,
        downsample_conv: type | None = None,
        downsample_kwargs: dict | None = None,
        upsample_conv: type | None = None,
        upsample_kwargs: dict | None = None,
        dropout: type | None = None,
        dropout_p: float = 0.0,
        merge: type = SumMerge,
        merge_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        transfer_modules = []

        if downsample_conv:
            transfer_modules.append(
                ParallelTransfer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    downsample_conv=downsample_conv,
                    downsample_kwargs=downsample_kwargs,
                    dropout=dropout,
                    dropout_p=dropout_p,
                    merge=merge,
                    merge_kwargs=merge_kwargs,
                    **kwargs,
                )
            )

        if upsample_conv:
            transfer_modules.append(
                ParallelTransfer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    upsample_conv=upsample_conv,
                    upsample_kwargs=upsample_kwargs,
                    dropout=dropout,
                    dropout_p=dropout_p,
                    merge=merge,
                    merge_kwargs=merge_kwargs,
                    **kwargs,
                )
            )

        super().__init__(*transfer_modules)
