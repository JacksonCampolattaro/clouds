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


def test_center_points_centers_utm_scale_coordinates():
    generator = torch.Generator().manual_seed(0)
    # DALES stores points in UTM metres, where a float32 mean is off by ~100 m.
    pos = torch.tensor([510_476.125, 5_449_424.246, 100.0]) + 50 * torch.rand(4096, 3, generator=generator)
    pos = pos.float()

    out = CenterPoints()(Data(pos=pos))

    expected = (pos.double() - pos.double().mean(0)).float()
    assert torch.allclose(out.pos, expected, atol=1e-4)


def test_center_points_centers_utm_scale_coordinates_per_graph():
    generator = torch.Generator().manual_seed(0)
    pos = torch.tensor([510_476.125, 5_449_424.246, 100.0]) + 50 * torch.rand(4096, 3, generator=generator)
    pos = pos.float()
    batch = torch.tensor([0] * 2048 + [1] * 2048)

    out = CenterPoints()(Data(pos=pos, batch=batch))

    for graph in (0, 1):
        mask = batch == graph
        expected = (pos[mask].double() - pos[mask].double().mean(0)).float()
        assert torch.allclose(out.pos[mask], expected, atol=1e-4)


def test_center_points_centers_only_the_selected_dims():
    data = Data(pos=torch.tensor([[1.0, 10.0, 3.0], [3.0, 20.0, 5.0]]))

    out = CenterPoints(dims=[0, 2])(data)

    assert torch.allclose(out.pos, torch.tensor([[-1.0, 10.0, -1.0], [1.0, 20.0, 1.0]]))


def test_ground_points_lands_on_the_per_axis_minimum():
    data = Data(pos=torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]))

    out = GroundPoints()(data)

    assert torch.allclose(out.pos, torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]))
