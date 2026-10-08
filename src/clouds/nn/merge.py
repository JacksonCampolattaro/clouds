import copy
from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from ..utils.sequence import _normalize_list
from .resolver import normalization_resolver


class Merge(Module):
    def __init__(self, in_channels: list[int]) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels

    def extra_repr(self) -> str:
        return f"{self.in_channels}, {self.out_channels}"


class SumMerge(Merge):
    def __init__(self, in_channels: list[int]) -> None:
        super().__init__(in_channels)
        assert all(c == in_channels[0] for c in in_channels)
        self.out_channels = in_channels[0]

    def forward(self, *x: Tensor) -> Tensor:
        return sum(x)


class NormSumMerge(Merge):
    def __init__(self, in_channels: list[int], norm: str | Callable | None = 'batch', norm_kwargs: dict | None = None) -> None:
        super().__init__(in_channels)
        assert len(in_channels) == 2
        assert all(c == in_channels[0] for c in in_channels)
        self.out_channels = in_channels[0]
        self.norms = ModuleList(
            [
                normalization_resolver(norm, in_channels=c, **(norm_kwargs or {}))  #
                for c in in_channels[1:]
            ]
        )

    def forward(self, *x: Tensor) -> Tensor:
        return x[0] + sum(n(xi) for n, xi in zip(self.norms, x[1:], strict=True))


class IdentityMerge(Merge):
    def __init__(self, in_channels: list[int], **_) -> None:
        assert len(in_channels) == 1
        super().__init__(in_channels)
        self.out_channels = in_channels[0]

    def forward(self, x: Tensor) -> Tensor:
        return x


class CatMerge(Merge):
    def __init__(self, in_channels: list[int], **_) -> None:
        super().__init__(in_channels)
        self.out_channels = sum(in_channels)

    def forward(self, *x: Tensor) -> Tensor:
        return torch.cat(x, dim=-1)


class WeightedSumMerge(Merge):
    def __init__(
        self,
        in_channels: list[int],
        weights: list[float] | None = None,
        normalize: bool = False,
    ) -> None:
        super().__init__(in_channels)
        assert all(c == in_channels[0] for c in in_channels)
        self.out_channels = in_channels[0]
        self.normalize = normalize

        if len(in_channels) == 1:
            weights = [1.0]

        weights = copy.copy(weights) or [1 / len(in_channels)] * len(in_channels)
        if ... in weights and len(weights) == 2:
            explicit_weights = [w for w in weights if w != ...]
            implied_weight = (1 - sum(explicit_weights)) / (len(in_channels) - len(explicit_weights))

            for i in range(len(in_channels)):
                if not i < len(weights):
                    weights.append(implied_weight)
                elif weights[i] == ...:
                    weights[i] = implied_weight
        else:
            weights = _normalize_list(weights, len(in_channels))

        self.weights = torch.nn.Parameter(torch.tensor(weights))

    def forward(self, *x: Tensor) -> Tensor:
        weights = self.weights
        if self.normalize:
            weights = weights / self.weights.sum()
        return (torch.stack(x) * weights.reshape(-1, 1, 1)).sum(dim=0)

    def extra_repr(self) -> str:
        return f"{self.in_channels}, {self.out_channels}, weights={self.weights.tolist()}"
