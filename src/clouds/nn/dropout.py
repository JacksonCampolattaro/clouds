from torch import Tensor
from torch.nn import Module


class PathDropout(Module):
    """
    Drop entire paths per sample.
    Adapted from https://github.com/Matrix-ASC/DeLA/blob/main/utils/timm/models/layers/drop.py#L157
    """

    def __init__(self, p: float = 0.2, scale_by_keep: bool = True, in_channels: int | None = None) -> None:
        super().__init__()
        self.p = p
        self.scale_by_keep = scale_by_keep
        self.in_channels, self.out_channels = in_channels, in_channels

    def forward(self, x: Tensor, batch: Tensor, ptr: Tensor | None = None) -> Tensor:
        if not self.p or not self.training:
            return x

        # ptr must be passed for use with torch dynamo to avoid data-dependent sizes!
        b = ptr.size(0) if ptr is not None else int(batch.amax() + 1)

        keep_p = 1.0 - self.p
        random = x.new_empty([b]).bernoulli_(keep_p)

        if self.scale_by_keep:
            random /= keep_p

        mask = random[batch].unsqueeze(-1)
        return x * mask

    def extra_repr(self) -> str:
        return f'p={self.p:0.3f}'


class PointDropout(Module):
    """
    Drop points at random.
    Adapted from https://github.com/Matrix-ASC/DeLA/blob/main/utils/timm/models/layers/drop.py#L157
    """

    def __init__(self, p: float = 0.2, scale_by_keep: bool = True, in_channels: int | None = None) -> None:
        super().__init__()
        self.p = p
        self.scale_by_keep = scale_by_keep
        self.in_channels, self.out_channels = in_channels, in_channels

    def forward(self, x: Tensor) -> Tensor:
        if not self.p or not self.training:
            return x

        n, _c = x.shape

        keep_p = 1.0 - self.p
        random = x.new_empty([n]).bernoulli_(keep_p)

        if self.scale_by_keep:
            random /= keep_p

        mask = random.unsqueeze(-1)
        return x * mask

    def extra_repr(self) -> str:
        return f'p={self.p:0.3f}'
