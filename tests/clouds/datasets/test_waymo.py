"""Tests for the SemanticWaymo per-frame ``.npz`` storage and read path. Synthetic only (no raw/ or network)."""

import json
import os
import pickle

import pytest
import torch
from torch_geometric.data import Data

from clouds.datasets.waymo import SemanticWaymo, _read_frame, _write_frame, verify_split


def _frame(n: int, labeled: bool = True) -> Data:
    data = Data(
        pos=torch.randn(n, 3, dtype=torch.float32),
        intensity=torch.rand(n, 1, dtype=torch.float32).tanh(),
    )
    if labeled:
        data.y = torch.randint(-1, 22, (n,), dtype=torch.long)
    return data


@pytest.mark.parametrize('labeled', [True, False])
@pytest.mark.parametrize('n', [0, 1, 7, 300])
def test_write_read_round_trip(tmp_path, n, labeled):
    path = str(tmp_path / 'frame.npz')
    data = _frame(n, labeled=labeled)
    _write_frame(path, data)

    out = _read_frame(path)
    assert torch.equal(out.pos, data.pos)
    assert torch.equal(out.intensity, data.intensity)
    if labeled:
        assert torch.equal(out.y, data.y)
        assert out.y.dtype == torch.long
    else:
        assert 'y' not in out


def test_write_rejects_labels_outside_int8(tmp_path):
    data = _frame(3)
    data.y = torch.tensor([0, 1, 300])
    with pytest.raises(AssertionError):
        _write_frame(str(tmp_path / 'frame.npz'), data)


def test_read_returns_writable_tensors(tmp_path):
    path = str(tmp_path / 'frame.npz')
    _write_frame(path, _frame(5))
    out = _read_frame(path)
    out.pos[0, 0] = 42.0
    out.intensity[0, 0] = 0.5
    assert out.pos[0, 0].item() == 42.0


def test_write_is_atomic_no_tmp_left_behind(tmp_path):
    path = str(tmp_path / 'frame.npz')
    _write_frame(path, _frame(4))
    assert os.path.exists(path)
    assert not os.path.exists(path + '.tmp')


def _write_scan(frame_dir: str, frames: list[Data]) -> int:
    os.makedirs(frame_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        _write_frame(os.path.join(frame_dir, f'{i:04d}.npz'), frame)
    return len(frames)


@pytest.fixture
def synthetic_root(tmp_path):
    """A processed-only root (no raw/) with a labeled training scan pair and one unlabeled testing scan."""
    root = tmp_path / 'SemanticWaymo'
    processed = root / 'processed'
    processed.mkdir(parents=True)

    train_scans = []
    for i, frames in enumerate([[_frame(10), _frame(0), _frame(3)], [_frame(5)]]):
        rel = f'training/seg{i}'
        train_scans.append([rel, _write_scan(str(processed / rel), frames)])

    test_frames = [_frame(4, labeled=False), _frame(6, labeled=False)]
    test_scans = [['testing/seg0', _write_scan(str(processed / 'testing/seg0'), test_frames)]]

    for name, split, scans in [
        ('index_training.json', 'training', train_scans),
        ('index_testing.json', 'testing', test_scans),
    ]:
        with open(processed / name, 'w') as f:
            json.dump({'version': 2, 'split': split, 'compression': 'npz-deflate', 'scans': scans}, f)
    return str(root)


def test_train_split_length_and_labels(synthetic_root):
    ds = SemanticWaymo(root=synthetic_root, split='train')
    assert len(ds) == 4
    sample = ds[0]
    assert sample.pos.shape == (10, 3)
    assert sample.y.dtype == torch.long
    assert sample.intensity.shape == (10, 1)


def test_get_returns_writable_tensors(synthetic_root):
    ds = SemanticWaymo(root=synthetic_root, split='train')
    data = ds.get(0)
    before = data.pos.clone()
    data.pos.add_(1.0)
    assert torch.allclose(data.pos, before + 1.0)


def test_getitem_with_zero_point_frame(synthetic_root):
    ds = SemanticWaymo(root=synthetic_root, split='train')
    empty = ds[1]
    assert empty.pos.shape == (0, 3)
    assert empty.y.numel() == 0


def test_pred_split_without_labels(synthetic_root):
    ds = SemanticWaymo(root=synthetic_root, split='pred')
    assert len(ds) == 2
    data = ds[0]
    assert 'y' not in data
    assert data.pos.shape == (4, 3)


def test_mix3d_concatenates_frames(synthetic_root):
    ds = SemanticWaymo(root=synthetic_root, split='train', mix3d_p=1.0)
    data = ds[0]
    assert data.pos.shape[0] > 10
    assert data.pos.shape[0] == data.intensity.shape[0] == data.y.shape[0]


def test_pickle_round_trip(synthetic_root):
    ds = SemanticWaymo(root=synthetic_root, split='train')
    clone = pickle.loads(pickle.dumps(ds))
    assert torch.equal(clone[0].pos, ds[0].pos)


def test_val_and_test_map_to_validation_index(tmp_path):
    root = tmp_path / 'SemanticWaymo'
    processed = root / 'processed'
    _write_scan(str(processed / 'validation/seg'), [_frame(2)])
    with open(processed / 'index_validation.json', 'w') as f:
        json.dump(
            {'version': 2, 'split': 'validation', 'compression': 'npz-deflate', 'scans': [['validation/seg', 1]]},
            f,
        )
    assert len(SemanticWaymo(root=str(root), split='val')) == 1
    assert len(SemanticWaymo(root=str(root), split='test')) == 1


def test_verify_split_passes_and_detects_missing_frame(synthetic_root):
    processed = os.path.join(synthetic_root, 'processed')
    assert verify_split(processed, 'training') == []

    os.remove(os.path.join(processed, 'training/seg0/0001.npz'))
    problems = verify_split(processed, 'training')
    assert any('frames, index says' in p for p in problems)


def test_verify_split_detects_corrupt_frame(synthetic_root):
    processed = os.path.join(synthetic_root, 'processed')
    frame_path = os.path.join(processed, 'training/seg0/0000.npz')
    with open(frame_path, 'wb') as f:
        f.write(b'not a zip file')
    problems = verify_split(processed, 'training')
    assert any('training/seg0/0000.npz' in p for p in problems)


def test_processed_only_root_needs_no_raw(synthetic_root):
    assert not os.path.exists(os.path.join(synthetic_root, 'raw'))
    SemanticWaymo(root=synthetic_root, split='train')
