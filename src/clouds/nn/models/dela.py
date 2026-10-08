from ..residual import make_residual_type
from .mlp import MLP


class DeLAMLP(MLP):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        hidden_factor: float = 2,
        weight_initializer: str | None = None,
        **kwargs,
    ) -> None:
        out_channels = out_channels or in_channels
        super().__init__(
            channel_list=[in_channels, int(out_channels * hidden_factor), out_channels],
            weight_initializer=weight_initializer,
            layout='lal',
            **(kwargs | dict(bias=[True, False])),
        )


class DeLANormProj(MLP):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        **kwargs,
    ) -> None:
        out_channels = in_channels if out_channels is None else out_channels
        super().__init__(channel_list=[in_channels, out_channels], **(kwargs | dict(bias=True, layout='nld')))


ResDeLAMLP = make_residual_type(DeLAMLP)
