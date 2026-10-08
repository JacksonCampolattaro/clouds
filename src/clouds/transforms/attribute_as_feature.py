import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class AttributeAsFeature(BaseTransform):
    def __init__(
        self,
        attributes: list[str],
        overwrite: bool = True,
        drop: bool = True,
    ) -> None:
        super().__init__()
        self.attributes = attributes
        self.overwrite = overwrite
        self.drop = drop

    def forward(self, data: Data) -> Data:
        for store in data.node_stores:
            pos = store.get('pos')
            if not isinstance(pos, Tensor):
                continue
            assert isinstance(store.num_nodes, int)

            if not isinstance(store.get('x'), Tensor) or self.overwrite:
                store.x = torch.zeros([store.num_nodes, 0], dtype=pos.dtype, device=pos.device)

            for key in self.attributes:
                attribute = store.get(key)
                if not isinstance(attribute, Tensor):
                    # Levels of a multigrid may not all carry the same attributes
                    assert isinstance(data, HeteroData)
                    continue
                assert isinstance(store.x, Tensor)
                store.x = torch.cat([store.x, attribute.to(dtype=store.x.dtype)], dim=-1)
                if self.drop:
                    del store[key]

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(attributes={self.attributes}, overwrite={self.overwrite}, drop={self.drop})"
