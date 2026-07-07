import pytest
import torch
from torch_geometric.data import Data


@pytest.fixture
def make_point_cloud():
    """Factory fixture building a minimal point-cloud `Data` object."""

    def _make_point_cloud(num_nodes=4, dim=3, with_norm=False):
        pos = torch.arange(num_nodes * dim, dtype=torch.float).reshape(num_nodes, dim)
        data = Data(pos=pos)
        if with_norm:
            norm = torch.zeros(num_nodes, dim)
            norm[:, 0] = 1.0  # unit vectors pointing along the first axis
            data.norm = norm
        return data

    return _make_point_cloud
