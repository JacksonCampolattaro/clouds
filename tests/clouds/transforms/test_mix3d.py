import itertools

import pytest
import torch
from torch_geometric.data import Data

from clouds.transforms import Mix3D


@pytest.fixture
def make_batch():
    def _make_batch(sizes):
        batch = torch.cat([torch.full((n,), i) for i, n in enumerate(sizes)])
        ptr = torch.tensor([0, *torch.cumsum(torch.tensor(sizes), dim=0)])
        return Data(batch=batch, ptr=ptr)

    return _make_batch


def test_ptr_batch_consistency(make_batch):
    data = Mix3D(p=0.8)(make_batch([4, 3, 5, 2, 6, 1, 7]))

    expected_batch = torch.repeat_interleave(torch.arange(data.ptr.size(0) - 1), data.ptr[1:] - data.ptr[:-1])

    assert torch.equal(data.batch, expected_batch)


def test_never_merges_more_than_two_items(make_batch):
    torch.manual_seed(0)
    sizes = [3, 4, 5, 2, 6, 3, 4, 5, 2, 6]
    original_ptr = make_batch(sizes).ptr

    transform = Mix3D(p=0.9)  # high p to stress-test merging

    for _ in range(50):
        data = transform(make_batch(sizes))

        original_indices = [original_ptr.tolist().index(v) for v in data.ptr.tolist()]
        gaps = [b - a for a, b in itertools.pairwise(original_indices)]

        assert all(gap <= 2 for gap in gaps)


def test_endpoints_always_preserved(make_batch):
    data = Mix3D(p=1.0)(make_batch([2, 3, 4, 5]))

    assert data.ptr[0].item() == 0
    assert data.ptr[-1].item() == 14
