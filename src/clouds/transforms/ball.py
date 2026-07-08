import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from .knn import knn


class BallGraph(BaseTransform):
    def __init__(
        self,
        r: float,
        loop: bool = False,
        max_num_neighbors: int = 32,
        num_threads: int = 4,
    ) -> None:
        self.r = r
        self.k = max_num_neighbors
        self.num_threads = num_threads

    def forward(self, data: Data) -> Data:
        assert isinstance(data.pos, Tensor)

        edge_index = knn(
            data.pos,
            k=self.k,
            batch=data.get('batch', None),
            num_threads=self.num_threads,
        )
        dest = torch.arange(
            edge_index.size(0),
            device=edge_index.device,
            dtype=torch.long,
        ).repeat_interleave(edge_index.size(1))
        source = edge_index.flatten()

        dist = torch.linalg.vector_norm(data.pos[dest] - data.pos[source], dim=-1)
        mask = (dist < self.r).flatten().nonzero().flatten()

        data.edge_index = torch.stack([source[mask], dest[mask]], dim=0)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r={self.r}, loop={self.loop}, max_num_neighbors={self.k}, num_threads={self.num_threads})"
