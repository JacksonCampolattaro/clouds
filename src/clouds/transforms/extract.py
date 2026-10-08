import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class ExtractHeights(BaseTransform):
    def __init__(self, gravity_axis: int = 2, ground: bool = False, scale: float | None = None) -> None:
        super().__init__()
        self.gravity_axis = gravity_axis
        self.ground = ground
        self.scale = scale

    def forward(self, data: Data) -> Data:
        for store in data.node_stores:
            pos = store.get('pos')
            if not isinstance(pos, Tensor):
                continue

            height = pos[:, self.gravity_axis].unsqueeze(-1)
            if self.ground:
                assert not isinstance(getattr(store, 'batch', None), Tensor)
                height = height - torch.amin(height)
            if self.scale:
                height = height * self.scale
            store.height = height

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(gravity_axis={self.gravity_axis}, ground={self.ground})'


class ExtractCoords(BaseTransform):
    def __init__(self, dims: list[int] | None = None, ground: bool = False, scale: float | None = None) -> None:
        super().__init__()
        self.dims = dims
        self.ground = ground
        self.scale = scale

    def forward(self, data: Data) -> Data:
        for store in data.node_stores:
            pos = store.get('pos')
            if not isinstance(pos, Tensor):
                continue

            coord = pos[:, self.dims] if self.dims else pos
            if self.ground:
                assert not isinstance(getattr(store, 'batch', None), Tensor)
                coord = coord - torch.amin(coord)
            if self.scale:
                coord = coord * self.scale
            store.coord = coord

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(dims={self.dims}, ground={self.ground}, scale={self.scale})'
