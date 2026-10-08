from collections.abc import Iterable

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, Center

from .hetero import require_homogeneous


class ClampPos(BaseTransform):
    """Clamp the ``pos`` attribute of every node (or node type) into a box."""

    def __init__(self, min: Iterable, max: Iterable) -> None:
        super().__init__()
        self.min, self.max = torch.tensor(min, dtype=torch.float), torch.tensor(max, dtype=torch.float)

    def forward(self, data: Data) -> Data:
        for store in data.node_stores:
            if isinstance(getattr(store, 'pos', None), Tensor):
                store.pos = torch.clamp(
                    store.pos,
                    min=self.min.unsqueeze(0).to(store.pos.device),
                    max=self.max.unsqueeze(0).to(store.pos.device),
                )

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min={self.min.tolist()}, max={self.max.tolist()})"


class NormalizeScale(BaseTransform):
    """Center the cloud and rescale so the 99th percentile of ``|pos|`` is ~1."""

    def __init__(self) -> None:
        super().__init__()
        self.center = Center()

    def forward(self, data: Data) -> Data:
        require_homogeneous(data, self.__class__.__name__)
        data = self.center(data)

        assert data.pos is not None
        scale = (1.0 / torch.quantile(data.pos.abs().flatten(), 0.99)) * 0.999999
        data.pos = data.pos * scale

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
