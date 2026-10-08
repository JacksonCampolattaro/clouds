from collections.abc import Iterable

import torch
from torch.nn import Module, ModuleList


@torch.compiler.disable  # FIXME: shouldn't be necessary
class CombinedLoss(ModuleList):
    def __init__(self, losses: Iterable[Module], weights: Iterable[float] | None = None) -> None:
        losses = list(losses)
        super().__init__(losses)
        self.weights = weights or [1.0 for _ in losses]

    def forward(self, output, target):
        return sum(w * loss(output, target) for w, loss in zip(self.weights, self, strict=True))

    # TODO: improve repr
