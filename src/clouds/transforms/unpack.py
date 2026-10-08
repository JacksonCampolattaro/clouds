import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class UnpackSourceGraph(BaseTransform):
    """Convert source-indexed ``(N, K)`` edges into PyG's pairwise ``(2, E)`` form.

    The inverse of the dense kNN edge convention used by
    :class:`SourceIndexedData` and the ``*Select``/``*Sample`` transforms.
    """

    def forward(self, data: Data) -> Data:
        if not isinstance(data, HeteroData):
            assert isinstance(data.edge_index, Tensor)

        for store in data.edge_stores:
            edge_index = store.get('edge_index')
            if not isinstance(edge_index, Tensor):
                continue

            store.edge_index = torch.stack(
                [
                    edge_index.flatten(),
                    torch.arange(
                        edge_index.size(0),
                        dtype=edge_index.dtype,
                        device=edge_index.device,
                    ).repeat_interleave(edge_index.size(1)),
                ]
            )

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
