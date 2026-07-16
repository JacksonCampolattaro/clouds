import time

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from clouds.data.threading_dataloader import ThreadingDataLoader

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


class RangeDataset(Dataset):
    """Simple dataset returning integers 0..n-1, optionally with an
    artificial per-item delay to encourage out-of-order completion across
    worker threads."""

    def __init__(self, n, delay=0.0):
        self.n = n
        self.delay = delay

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.delay:
            # Vary delay so items don't necessarily finish in submission order,
            # exercising the reordering logic in _next_data / _task_info.
            time.sleep(self.delay * ((idx % 3) + 1))
        return idx


class FaultyDataset(Dataset):
    """Dataset that raises on a specific index, to test exception
    propagation through the worker -> ExceptionWrapper -> reraise path."""

    def __init__(self, n, fail_idx):
        self.n = n
        self.fail_idx = fail_idx

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if idx == self.fail_idx:
            raise ValueError(f"intentional failure at index {idx}")
        return idx


def collect(loader):
    """Exhaust a DataLoader and return the list of yielded batches (as
    plain Python values, flattened for scalar datasets)."""
    out = []
    for batch in loader:
        out.extend(batch.tolist() if torch.is_tensor(batch) else list(batch))
    return out


# ---------------------------------------------------------------------------
# Correctness vs. the reference single-process loader
# ---------------------------------------------------------------------------


class TestMatchesReferenceLoader:
    @pytest.mark.parametrize("n,batch_size", [(20, 1), (20, 4), (17, 5)])
    def test_same_items_no_shuffle(self, n, batch_size):
        ds = RangeDataset(n)

        ref = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        threaded = ThreadingDataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=3, prefetch_factor=2)

        assert collect(ref) == collect(threaded)

    def test_order_preserved_despite_uneven_worker_delays(self):
        # Items complete out of submission order across threads (thanks to the
        # variable delay), but _next_data's rcvd_idx bookkeeping must still
        # yield batches in strict dataset order, matching the reference loader.
        ds = RangeDataset(30, delay=0.005)

        ref = DataLoader(RangeDataset(30), batch_size=3, shuffle=False, num_workers=0)
        threaded = ThreadingDataLoader(ds, batch_size=3, shuffle=False, num_workers=4, prefetch_factor=2)

        assert collect(ref) == collect(threaded)

    def test_drop_last_matches_reference(self):
        ds = RangeDataset(23)  # not evenly divisible by batch_size

        ref = DataLoader(ds, batch_size=5, shuffle=False, num_workers=0, drop_last=True)
        threaded = ThreadingDataLoader(ds, batch_size=5, shuffle=False, num_workers=3, prefetch_factor=2, drop_last=True)

        ref_batches = list(ref)
        threaded_batches = list(threaded)

        assert len(ref_batches) == len(threaded_batches)
        assert collect(ref) == collect(threaded)

    def test_shuffle_yields_same_multiset_of_items(self):
        # With shuffling, exact order isn't expected to match a differently
        # seeded reference loader, but the *set* of items produced over one
        # full epoch must still be exactly the dataset once each.
        ds = RangeDataset(40)
        threaded = ThreadingDataLoader(ds, batch_size=4, shuffle=True, num_workers=4, prefetch_factor=2)

        result = sorted(collect(threaded))
        assert result == list(range(40))


# ---------------------------------------------------------------------------
# Multi-epoch / persistent workers behavior
# ---------------------------------------------------------------------------


class TestEpochsAndPersistence:
    def test_multiple_epochs_each_see_full_dataset(self):
        ds = RangeDataset(15)
        loader = ThreadingDataLoader(ds, batch_size=3, shuffle=False, num_workers=2, prefetch_factor=2)

        first_epoch = collect(loader)
        second_epoch = collect(loader)

        assert first_epoch == list(range(15))
        assert second_epoch == list(range(15))

    def test_non_persistent_workers_are_shutdown_after_epoch(self):
        ds = RangeDataset(10)
        loader = ThreadingDataLoader(ds, batch_size=2, num_workers=2, prefetch_factor=2, persistent_workers=False)
        it = iter(loader)
        for _ in it:
            pass

        # After exhausting a non-persistent iterator, worker threads must
        # have been joined (i.e., no longer alive).
        assert it._shutdown is True
        for worker in it._workers:
            assert not worker.is_alive()

    def test_persistent_workers_survive_across_epochs(self):
        ds = RangeDataset(10)
        loader = ThreadingDataLoader(ds, batch_size=2, num_workers=2, prefetch_factor=2, persistent_workers=True)

        epoch1 = collect(loader)
        # The underlying persistent iterator/threads should still be alive
        # between epochs.
        it = loader._iterator
        assert it is not None
        for worker in it._workers:
            assert worker.is_alive()

        epoch2 = collect(loader)

        assert epoch1 == list(range(10))
        assert epoch2 == list(range(10))


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------


class TestExceptionPropagation:
    def test_worker_exception_propagates_to_main_thread(self):
        ds = FaultyDataset(20, fail_idx=7)
        loader = ThreadingDataLoader(ds, batch_size=1, shuffle=False, num_workers=2, prefetch_factor=2)

        with pytest.raises(Exception):
            for _ in loader:
                pass

    def test_reference_loader_raises_same_exception_type(self):
        # Confirms parity: both loaders fail, and fail for the same
        # underlying reason (ValueError raised in __getitem__), not just
        # "some" exception.
        fail_idx = 5

        ref = DataLoader(FaultyDataset(20, fail_idx), batch_size=1, num_workers=0)
        threaded = ThreadingDataLoader(FaultyDataset(20, fail_idx), batch_size=1, num_workers=2, prefetch_factor=2)

        with pytest.raises(Exception) as ref_exc:
            list(ref)
        with pytest.raises(Exception) as threaded_exc:
            list(threaded)

        assert "intentional failure" in str(ref_exc.value)
        assert "intentional failure" in str(threaded_exc.value)


# ---------------------------------------------------------------------------
# Resource cleanup
# ---------------------------------------------------------------------------


class TestResourceCleanup:
    def test_shutdown_workers_is_idempotent(self):
        ds = RangeDataset(10)
        loader = ThreadingDataLoader(ds, batch_size=2, num_workers=2, prefetch_factor=2)
        it = iter(loader)
        list(it)  # exhaust -> triggers shutdown internally for non-persistent

        # Calling shutdown again must not raise or hang.
        it._shutdown_workers()
        it._shutdown_workers()
        assert it._shutdown is True
