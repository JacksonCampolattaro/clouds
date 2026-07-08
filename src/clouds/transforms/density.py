import itertools
import math

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.transforms import BaseTransform

from clouds.transforms.knn import knn
from clouds.transforms.random_select import RandomSelect


def _unit_ball_volume(d: float) -> float:
    return math.pi ** (d / 2) / (math.gamma(1 + d / 2))


class EstimateDensity(BaseTransform):
    def __init__(
        self,
        pointwise: bool = True,
        estimation_factor: float = 0.05,
        d: float = 2,
    ):
        super().__init__()
        self.pointwise, self.estimation_factor = pointwise, estimation_factor
        self.k = 15
        self.d = d
        self.V_d = _unit_ball_volume(d)

    def forward(self, data: Data) -> Data:
        assert isinstance(data.pos, Tensor)

        # Sample some points
        # TODO: use random select!
        coarse_data = RandomSelect(selection_factor=self.estimation_factor, min_num_points=64)(data)

        # Find distances to the kth nearest neighbors
        query_data = data if self.pointwise else coarse_data
        distances, _knn = knn(
            pos=coarse_data.pos,
            batch=coarse_data.batch,
            query_pos=query_data.pos,
            query_batch=query_data.batch,
            k=self.k + 1,
            return_distances=True,
        )
        r_k = distances[:, -1]

        # Estimate lambda
        # TODO: sampling correction can come after mean pooling
        approx_lambda = (1 / self.estimation_factor) * self.k / (self.V_d * r_k**2)

        if self.pointwise:
            # Pointwise density doesn't require any pooling step
            data.density = approx_lambda.unsqueeze(-1)
        else:
            data.density = MeanAggregation()(
                approx_lambda,
                index=query_data.batch,
                ptr=query_data.ptr,
                dim=0,
            )

        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"pointwise={self.pointwise}, "
            f"estimation_factor={self.estimation_factor}, "
            f"d={self.d}, "
            f"k={self.k})"
        )


class InverseDensitySelect(BaseTransform):
    def forward(self, data: Data) -> Data:
        assert isinstance(data.density, Tensor) and data.density.size(0) == data.num_nodes

        def _inverse_density_select(density: Tensor):
            # Can I avoid the argsort? Does it belong here?
            return torch.argsort(torch.multinomial(1 / density, num_samples=density.size(0)))

        if isinstance(data.batch, Tensor):
            assert isinstance(data.ptr, Tensor)
            item_selections = [
                start + _inverse_density_select(data.density[start:end, 0])  #
                for start, end in itertools.pairwise(data.ptr)
            ]
            data.selection_index = torch.cat(item_selections, dim=0)
        else:
            data.selection_index = _inverse_density_select(data.density.flatten())

        return data
