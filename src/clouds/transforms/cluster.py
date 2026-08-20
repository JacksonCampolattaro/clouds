import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils._scatter import scatter_argmax

from clouds.transforms.apply_selection import apply_selection
from clouds.transforms.knn import knn


def _select_random_node_per_cluster(cluster: Tensor) -> Tensor:
    """Randomly select one point from each cluster."""
    rand_values = torch.rand(cluster.size(0), device=cluster.device)
    selection_index = scatter_argmax(rand_values, cluster)
    return selection_index


def _select_nth_node_per_cluster(cluster: Tensor, n: int, cluster_is_sorted: bool | None = False) -> Tensor:
    """Deterministically select a point from each cluster."""
    # FIXME: assumes clusters are in order and contiguous!

    if (cluster_is_sorted is None and torch.any(cluster[:-1] > cluster[1:])) or cluster_is_sorted is False:
        sorted_clusters, sorted_indices = torch.sort(cluster)
        sorted_selection = _select_nth_node_per_cluster(sorted_clusters, n, cluster_is_sorted=True)
        return sorted_indices[sorted_selection]
    
    # Clusters must be in order:
    sorted_clusters, sorted_indices = torch.sort(cluster)

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
        return f"{self.__class__.__name__}(deterministic={self.deterministic}, pick={self.pick}, "


class ClusterSample(ClusterSelect):
    def forward(self, data: Data) -> Data:
        return apply_selection(super().forward(data))


class NearestSelectionCluster(BaseTransform):
    def forward(self, data: Data) -> Data:
        data.cluster = knn(
            pos=data.pos[data.selection_index],
            batch=data.batch[data.selection_index] if isinstance(data.batch, Tensor) else None,
            query_pos=data.pos,
            query_batch=data.batch,
            k=1,
        )
        return data
