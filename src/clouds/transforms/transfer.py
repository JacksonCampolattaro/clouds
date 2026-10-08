import torch
from torch import Tensor
from torch_geometric.transforms import BaseTransform

from ..data import MultiGridData
from .knn import knn


class KNNUpsampleNeighbors(BaseTransform):
    """Connect each coarse-level node to its ``k`` nearest fine-level neighbors.

    Writes the inter-level ``(coarse, 'to', fine)`` ``edge_index`` for every
    adjacent pair of levels in a :class:`MultiGridData`.
    """

    def __init__(self, k: int | list[int] = 1) -> None:
        super().__init__()
        self.k = k

    def forward(self, data: MultiGridData) -> MultiGridData:
        ks = self.k if isinstance(self.k, list) else [self.k] * (len(data.node_types) - 1)

        for fine, coarse, k in zip(data.node_types[:-1], data.node_types[1:], ks, strict=True):
            if hasattr(data[fine], 'pos') and hasattr(data[coarse], 'pos'):
                data[coarse, 'to', fine].edge_index = knn(
                    pos=data[coarse].pos,
                    batch=getattr(data[coarse], 'batch', None),
                    query_pos=data[fine].pos,
                    query_batch=getattr(data[fine], 'batch', None),
                    k=k,
                )

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k})"


class EdgesToDownsampleNeighbors(BaseTransform):
    """Derive inter-level edges from intra-level edges plus a ``selection_index``.

    For each adjacent pair of levels ``(fine, coarse)``, edges of the fine level
    which point at a selected node become ``(fine, 'to', coarse)`` downsampling
    edges. Both dense ``(N, K)`` and pairwise ``(2, E)`` fine edge formats are
    supported.
    """

    def forward(self, data: MultiGridData) -> MultiGridData:
        for fine, coarse in zip(data.node_types[:-1], data.node_types[1:], strict=True):
            if hasattr(data[fine], 'selection_index') and hasattr(data[fine, 'to', fine], 'edge_index'):
                fine_edge_index = data[fine, 'to', fine].edge_index

                if isinstance(fine_edge_index, Tensor) and fine_edge_index.size(0) == 2:
                    # PyG-style edge-pairs require some special handling
                    # TODO: untested!

                    # Only take nodes present in the selection index
                    node_mask = torch.zeros([data[fine].num_nodes], dtype=torch.bool, device=data[fine].selection_index.device)
                    node_mask[data[fine].selection_index.flatten()] = True

                    # Only take edges which connect to a selected node
                    edge_mask = node_mask[fine_edge_index[1]]

                    for key, item in data[fine, 'to', fine].items():
                        if key == 'edge_index':
                            # Filter edges
                            restriction_edge_index = item[:, edge_mask]

                            # Re-index selected edges
                            new_index_map = torch.zeros(data[fine].num_nodes, dtype=torch.long, device=fine_edge_index.device)
                            new_index_map[data[fine].selection_index] = torch.arange(
                                len(data[fine].selection_index), device=fine_edge_index.device
                            )
                            restriction_edge_index[1, :] = new_index_map[restriction_edge_index[1, :]]

                            data[fine, 'to', coarse].edge_index = restriction_edge_index

                        else:
                            # Filter edge properties
                            data[fine, 'to', coarse][key] = item[edge_mask]

                else:
                    # Conveniently, sparse tensors can be handled exactly the same as source indices!
                    data[fine, 'to', coarse].edge_index = fine_edge_index[data[fine].selection_index]

        return data


# Edges from the coarse scale can't be converted to downsampling edges,
# so "EdgesToUpsampleNeighbors" isn't possible
