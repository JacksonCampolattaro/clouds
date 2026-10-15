import pytest
import torch
from torch_geometric.data import Data

from clouds.transforms import CenterPoints, GroundPoints


@pytest.mark.parametrize('transform', [CenterPoints(), GroundPoints()])
def test_center_does_not_write_through_pos_views(transform):
    storage = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0],
        ]
    )
    before = storage.clone()
    # Datasets hand out views into one flat cached tensor, so this is what a
    # DataLoader worker sees for a single sample.
    data = Data(pos=storage[:2])

    out = transform(data)

    assert torch.equal(storage, before)
    assert out.pos.data_ptr() != storage.data_ptr()


def test_center_points_centers_only_the_selected_dims():
    data = Data(pos=torch.tensor([[1.0, 10.0, 3.0], [3.0, 20.0, 5.0]]))

    out = CenterPoints(dims=[0, 2])(data)

    assert torch.allclose(out.pos, torch.tensor([[-1.0, 10.0, -1.0], [1.0, 20.0, 1.0]]))


def test_ground_points_lands_on_the_per_axis_minimum():
    data = Data(pos=torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]))

    out = GroundPoints()(data)

    assert torch.allclose(out.pos, torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]))
