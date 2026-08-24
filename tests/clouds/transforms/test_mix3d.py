import pytest
import torch
from torch_geometric.data import Data

from clouds.transforms import Mix3D


class TestMix3D:

    @pytest.fixture
    def make_batch(self):
        """Factory fixture: build a simple batched Data object from per-item sizes."""
        def _make_batch(sizes: list[int]) -> Data:
            batch = torch.cat([torch.full((n,), i) for i, n in enumerate(sizes)])
            ptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(sizes), dim=0)))
            return Data(batch=batch, ptr=ptr)
        return _make_batch

    def test_ptr_batch_consistency(self, make_batch):
        """After the transform, batch must always match what ptr implies."""
        data = make_batch([4, 3, 5, 2, 6, 1, 7])
        transform = Mix3D(p=0.8)

        data = transform(data)

        expected_batch = torch.repeat_interleave(
            torch.arange(data.ptr.size(0) - 1),
            data.ptr[1:] - data.ptr[:-1],
        )
        assert torch.equal(data.batch, expected_batch)

    def test_never_merges_more_than_two_items(self, make_batch):
        """No two adjacent interior boundaries should ever both be deleted."""
        torch.manual_seed(0)
        sizes = [3, 4, 5, 2, 6, 3, 4, 5, 2, 6]  # 10 items -> 9 interior boundaries
        original_ptr = make_batch(sizes).ptr

        transform = Mix3D(p=0.9)  # high p to stress-test merging

        for _ in range(50):  # run many times to exercise randomness
            data = make_batch(sizes)
            data = transform(data)

            original_indices = [
                original_ptr.tolist().index(v) for v in data.ptr.tolist()
            ]

            gaps = [b - a for a, b in zip(original_indices, original_indices[1:])]
            assert all(gap <= 2 for gap in gaps), (
                f"Found a merge of more than 2 items: gaps={gaps}"
            )

    def test_endpoints_always_preserved(self, make_batch):
        """First and last ptr values must never be deleted."""
        data = make_batch([2, 3, 4, 5])
        transform = Mix3D(p=1.0)  # maximize deletion attempts

        data = transform(data)

        assert data.ptr[0].item() == 0
        assert data.ptr[-1].item() == 14  # sum of sizes
