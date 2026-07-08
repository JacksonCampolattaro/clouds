import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils._scatter import scatter_argmax


def _select_random_node_per_cluster(cluster: Tensor) -> Tensor:
    """Randomly select one point from each cluster."""
    rand_values = torch.rand(cluster.size(0), device=cluster.device)
    selection_index = scatter_argmax(rand_values, cluster)
    return selection_index


def _select_nth_node_per_cluster(cluster: Tensor, n: int) -> Tensor:
    """Deterministically select a point from each cluster."""
     
    # Calculate cluster sizes and starting indices
    cluster_sizes = torch.bincount(cluster)
    cluster_starts = torch.cumsum(cluster_sizes, dim=0) - cluster_sizes
    
    # Select offset point from each cluster
    cluster_offsets = n % cluster_sizes
    return (cluster_starts + cluster_offsets).to(torch.long)


class ClusterSelect(BaseTransform):
    """Select points from clusters using deterministic or random selection.
                        
    Assumes clusters are contiguous, sequential, and not shared between batches.
    """

    def __init__(
        self, 
        deterministic: bool = False, 
        pick: int | None = None, 
    ) -> None:
        super().__init__()
        self.deterministic = deterministic
        self.pick = pick
        self.current_pick = 0

    def forward(self, data: Data) -> Data:
        if self.deterministic:
            pick = self.pick if self.pick is not None else self.current_pick
            self.current_pick += 1
            data.selection_index = _select_nth_node_per_cluster(data.cluster, pick)
        else:
            data.selection_index = _select_random_node_per_cluster(data.cluster)

        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"deterministic={self.deterministic}, "
            f"pick={self.pick}, "
        )
