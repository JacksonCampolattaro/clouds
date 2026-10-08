import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class UnpackSourceGraph(BaseTransform):
    """Convert source-indexed ``(N, K)`` edges into PyG's pairwise ``(2, E)`` form.

    The inverse of the dense kNN edge convention used by
    :class:`SourceIndexedData` and the ``*Select``/``*Sample`` transforms.
    """

    def forward(self, data: Data) -> Data:
        assert isinstance(data.edge_index, Tensor)
        data.edge_index = torch.stack(
            [
                data.edge_index.flatten(),
                torch.arange(
                    data.edge_index.size(0),
                    dtype=data.edge_index.dtype,
                    device=data.edge_index.device,
                ).repeat_interleave(data.edge_index.size(1)),
            ]
        )
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
