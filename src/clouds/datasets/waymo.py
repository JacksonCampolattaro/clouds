import argparse
import glob
import io
import json
import logging
import multiprocessing
import os
import random
import shutil
import sys
import time
import urllib
import warnings
import zipfile
import zlib
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import torch
from rich import progress
from rich.progress import Progress
from tfrecord import tfrecord_iterator
from torch import Tensor
from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import BaseData  # private path; no public re-export in PyG 2.8
from torch_geometric.data.dataset import IndexType

INDEX_VERSION = 2
SPLIT_INDEX_FILES = {'training': 'index_training.json', 'validation': 'index_validation.json', 'testing': 'index_testing.json'}
TEST_IDS_FILE = '3d_semseg_test_set_frames.txt'

logger = logging.getLogger(__name__)

_worker_test_ids: set[str] = set()


def _get_rotation_matrix(roll: Tensor, pitch: Tensor, yaw: Tensor) -> Tensor:
    cos_roll, sin_roll = torch.cos(roll), torch.sin(roll)
    cos_pitch, sin_pitch = torch.cos(pitch), torch.sin(pitch)
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)

    ones = torch.ones_like(yaw)
    zeros = torch.zeros_like(yaw)

    r_roll = torch.stack(
        [
            ones,
            zeros,
            zeros,
            zeros,
            cos_roll,
            -sin_roll,
            zeros,
            sin_roll,
            cos_roll,
        ],
        axis=-1,
    ).reshape(*yaw.shape, 3, 3)
    r_pitch = torch.stack(
        [
            cos_pitch,
            zeros,
            sin_pitch,
            zeros,
            ones,
            zeros,
            -sin_pitch,
            zeros,
            cos_pitch,
        ],
        axis=-1,
    ).reshape(*yaw.shape, 3, 3)
    r_yaw = torch.stack(
        [
            cos_yaw,
            -sin_yaw,
            zeros,
            sin_yaw,
            cos_yaw,
            zeros,
            zeros,
            zeros,
            ones,
        ],
        axis=-1,
    ).reshape(*yaw.shape, 3, 3)

    return r_yaw @ r_pitch @ r_roll


def _get_transform(rotation: Tensor, translation: Tensor) -> Tensor:
    n = rotation.shape[-1]
    batch_shape = rotation.shape[:-2]
    transform = torch.zeros((*batch_shape, n + 1, n + 1), dtype=rotation.dtype)
    transform[..., :n, :n] = rotation
    transform[..., :n, n] = translation
    transform[..., n, n] = 1.0
    return transform


def _compute_inclination(inclination_min: float, inclination_max: float, height: int) -> Tensor:
    diff = inclination_max - inclination_min
    indices = torch.arange(height, dtype=torch.float)
    return (0.5 + indices) / height * diff + inclination_min


def _range_image_to_point_image(
    range_image: Tensor,
    extrinsic: Tensor,
    inclination: Tensor,
    pixel_pose: Tensor | None = None,
    frame_pose: Tensor | None = None,
) -> Tensor:
    depth_image = range_image[..., 0]
    height, width = depth_image.shape  # FIXME: range image has batch dim first?

    az_correction = torch.arctan2(extrinsic[1, 0], extrinsic[0, 0])
    ratios = (torch.arange(width, 0, -1).to(dtype=torch.float) - 0.5) / width
    azimuth = (ratios * 2.0 - 1.0) * torch.pi - az_correction.unsqueeze(-1)

    azimuth_tile = torch.tile(azimuth[torch.newaxis, :], (height, 1))
    inclination_tile = torch.tile(inclination[:, torch.newaxis], (1, width))

    cos_azimuth, sin_azimuth = torch.cos(azimuth_tile), torch.sin(azimuth_tile)
    cos_incl, sin_incl = torch.cos(inclination_tile), torch.sin(inclination_tile)

    x = cos_azimuth * cos_incl * depth_image
    y = sin_azimuth * cos_incl * depth_image
    z = sin_incl * depth_image

    depth_image_points = torch.stack([x, y, z], axis=-1)

    rotation = extrinsic[0:3, 0:3]
    translation = extrinsic[torch.newaxis, 0:3, 3]

    # Sensor frame -> vehicle frame.
    depth_image_points = torch.einsum('kr,hwr->hwk', rotation, depth_image_points) + translation

    if pixel_pose is not None:
        assert frame_pose is not None

        pixel_rotation = pixel_pose[..., 0:3, 0:3]
        pixel_translation = pixel_pose[..., 0:3, 3]
        depth_image_points = torch.einsum('hwij,hwj->hwi', pixel_rotation, depth_image_points) + pixel_translation

        world_to_vehicle = torch.linalg.inv(frame_pose)
        world_to_vehicle_rotation = world_to_vehicle[0:3, 0:3]
        world_to_vehicle_translation = world_to_vehicle[0:3, 3]
        depth_image_points = (
            torch.einsum('ij,hwj->hwi', world_to_vehicle_rotation, depth_image_points) + world_to_vehicle_translation
        )

    return depth_image_points


def _frame_path(processed_dir: str, rel_seg: str, frame_idx: int) -> str:
    return os.path.join(processed_dir, rel_seg, f'{frame_idx:04d}.npz')


def _write_frame(path: str, data: Data) -> None:
    """Write one frame as a compressed ``.npz`` (arrays: ``pos``, ``intensity``, and ``y`` when present).

    Written to ``<path>.tmp`` and atomically renamed, so an interrupted build never leaves a partial frame.
    """
    arrays = {
        'pos': data.pos.detach().cpu().numpy().astype(np.float32, copy=False),
        'intensity': data.intensity.detach().cpu().numpy().astype(np.float32, copy=False),
    }
    y = getattr(data, 'y', None)
    if y is not None:
        labels = y.detach().cpu().numpy()
        assert labels.size == 0 or (int(labels.min()) >= -128 and int(labels.max()) <= 127), 'y does not fit in int8'
        arrays['y'] = labels.astype(np.int8, copy=False)
    tmp = f'{path}.tmp'
    with open(tmp, 'wb') as f:  # a file object avoids numpy appending '.npz' to the temp name
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def _read_frame(path: str) -> Data:
    """Load one frame written by :func:`_write_frame`. Returns tensors that own writable memory."""
    with np.load(path) as archive:
        data = Data(
            pos=torch.from_numpy(archive['pos']),
            intensity=torch.from_numpy(archive['intensity']),
        )
        if 'y' in archive:
            data.y = torch.from_numpy(archive['y']).long()
    return data


def _init_worker(test_ids: set[str]) -> None:
    global _worker_test_ids
    torch.set_num_threads(1)
    _worker_test_ids = test_ids


def _convert_scan(args: tuple[str, str, str]) -> tuple[str, int]:
    """Convert one raw TFRecord scan into ``processed/<split>/<segment>/*.npz``. Returns ``(rel_seg, frame_count)``."""
    raw_path, rel_seg, out_dir = args

    if os.path.exists(out_dir):
        return rel_seg, len(glob.glob(os.path.join(out_dir, '*.npz')))

    from waymo_open_dataset.dataset_pb2 import Frame, LaserName, MatrixFloat, MatrixInt32

    def _decompress_tensor(v: bytes, dtype: torch.dtype = torch.float32) -> Tensor:
        matrix = MatrixFloat() if dtype == torch.float32 else MatrixInt32()
        matrix.ParseFromString(zlib.decompress(v))
        return torch.tensor(matrix.data, dtype=dtype).view(*matrix.shape.dims)

    def _frame_to_data(frame: Any) -> Data:

        # Extract relevant data from Waymo's spec
        frame_pose_transform = torch.tensor(frame.pose.transform, dtype=torch.float32).reshape(4, 4)
        range_images, camera_projections, label_images = dict(), dict(), dict()
        top_pose: Tensor or None = None
        for laser in frame.lasers:
            assert len(laser.ri_return1.range_image_compressed) and len(laser.ri_return2.range_image_compressed)
            range_images[laser.name] = [
                _decompress_tensor(laser.ri_return1.range_image_compressed),
                _decompress_tensor(laser.ri_return2.range_image_compressed),
            ]
            camera_projections[laser.name] = [
                _decompress_tensor(laser.ri_return1.camera_projection_compressed, dtype=torch.int32),
                _decompress_tensor(laser.ri_return2.camera_projection_compressed, dtype=torch.int32),
            ]
            if len(laser.ri_return1.segmentation_label_compressed) or len(laser.ri_return2.segmentation_label_compressed):
                label_images[laser.name] = [
                    _decompress_tensor(laser.ri_return1.segmentation_label_compressed, dtype=torch.int32),
                    _decompress_tensor(laser.ri_return2.segmentation_label_compressed, dtype=torch.int32),
                ]
            if laser.name == LaserName.TOP:
                top_pose = _decompress_tensor(laser.ri_return1.range_image_pose_compressed)

        assert top_pose is not None

        # Convert each pixel to a cartesian coordinate
        point_images = dict()
        top_pose_transform = _get_transform(
            rotation=_get_rotation_matrix(roll=top_pose[..., 0], pitch=top_pose[..., 1], yaw=top_pose[..., 2]),
            translation=top_pose[..., 3:],
        )
        for c in frame.context.laser_calibrations:
            extrinsic = torch.tensor(c.extrinsic.transform, dtype=torch.float).view(4, 4)
            point_images[c.name] = [
                _range_image_to_point_image(
                    ri,
                    extrinsic=extrinsic,
                    inclination=(
                        torch.tensor(c.beam_inclinations)
                        if len(c.beam_inclinations)
                        else _compute_inclination(c.beam_inclination_min, c.beam_inclination_max, ri.shape[0])
                    ).flip(-1),
                    pixel_pose=top_pose_transform if c.name == LaserName.TOP else None,
                    frame_pose=frame_pose_transform if c.name == LaserName.TOP else None,
                )
                for ri in range_images[c.name]
            ]
            # TODO: where do normals come from?

        # Flatten images
        points, intensities, labels = [], [], []
        for r in [0, 1]:
            for c in frame.context.laser_calibrations:
                # Identify valid pixels (e.g. sky will be ignored)
                mask = range_images[c.name][r][..., 0] > 0

                # Flatten points
                points.append(point_images[c.name][r].reshape(-1, 3)[mask.flatten(), :])

                # Extract intensity
                intensities.append(range_images[c.name][r][..., 1].reshape(-1, 1)[mask.flatten(), :])

                # Flatten labels (if present)
                if not label_images:
                    pass
                elif c.name in label_images:
                    labels.append(label_images[c.name][r].reshape(-1, 2)[mask.flatten(), :] - 1)
                else:
                    labels.append(torch.full((mask.sum(), 2), -1, dtype=torch.int32))

                # TODO: produce normals, intensity, etc.

        # Flatten everything into a single point cloud
        return Data(
            pos=torch.cat(points, dim=0),
            intensity=torch.cat(intensities, dim=0).tanh(),
            y=torch.cat(labels, dim=0)[:, 1].long() if labels else None,
        )

    tmp_dir = out_dir + '.tmp'
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    try:
        count = 0
        for raw_record in tfrecord_iterator(raw_path):
            frame = Frame()
            frame.ParseFromString(raw_record)

            # Skip unlabeled frames, unless they're in the test set
            if (
                not len(frame.lasers[0].ri_return1.segmentation_label_compressed)
                and f'{frame.context.name},{frame.timestamp_micros}' not in _worker_test_ids
            ):
                continue

            _write_frame(os.path.join(tmp_dir, f'{count:04d}.npz'), _frame_to_data(frame))
            count += 1
        os.replace(tmp_dir, out_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return rel_seg, count


def _load_test_ids(path: str) -> set[str]:
    with open(path, 'r') as f:
        return set(x.rstrip() for x in f.readlines())


def _write_json_atomic(path: str, obj: dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_index(path: str) -> dict:
    with open(path, 'r') as f:
        index = json.load(f)
    assert index['version'] == INDEX_VERSION, f'{path}: unsupported index version {index["version"]}'
    return index


def build_split(
    raw_dir: str,
    processed_dir: str,
    split_name: str,
    *,
    test_ids: set[str],
    workers: int = 8,
    limit: int | None = None,
    log_path: str | None = None,
) -> str:
    """Convert every scan in ``raw/<split_name>`` into ``processed/<split_name>/<segment>/*.npz`` and write the split index.

    Scans whose output directory already exists are skipped (their frame count is read back), so an interrupted
    build resumes. Returns the path of the written ``index_<split>.json``.
    """
    assert split_name in SPLIT_INDEX_FILES, f'unknown split {split_name!r}'
    out_split_dir = os.path.join(processed_dir, split_name)
    os.makedirs(out_split_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(out_split_dir, '*.tmp')):
        shutil.rmtree(stale, ignore_errors=True)

    raw_paths = sorted(glob.glob(os.path.join(raw_dir, split_name, '*.tfrecord')))
    if limit is not None:
        raw_paths = raw_paths[:limit]
    jobs = []
    for raw_path in raw_paths:
        segment = os.path.basename(raw_path).replace('.tfrecord', '')
        jobs.append((raw_path, f'{split_name}/{segment}', os.path.join(out_split_dir, segment)))

    file_handler = None
    if log_path is not None:
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    sys.path.append(raw_dir)
    results: dict[str, int] = {}
    try:
        logger.info('build %s: %d scans, workers=%d', split_name, len(jobs), workers)
        with (
            Progress(*Progress.get_default_columns(), progress.TimeElapsedColumn()) as p,
            multiprocessing.Pool(
                processes=workers,
                initializer=_init_worker,
                initargs=(test_ids,),
            ) as pool,
        ):
            task = p.add_task(f'Building Waymo {split_name}', total=len(jobs))
            started = time.monotonic()
            for rel_seg, frame_count in pool.imap_unordered(_convert_scan, jobs):
                results[rel_seg] = frame_count
                logger.info('%s frames=%d', rel_seg, frame_count)
                p.update(task, advance=1)
        logger.info('build %s finished in %.0fs', split_name, time.monotonic() - started)
    finally:
        if file_handler is not None:
            logger.removeHandler(file_handler)
            file_handler.close()

    index_path = os.path.join(processed_dir, SPLIT_INDEX_FILES[split_name])
    _write_json_atomic(
        index_path,
        {
            'version': INDEX_VERSION,
            'split': split_name,
            'compression': 'npz-deflate',
            'scans': [[rel_seg, results[rel_seg]] for rel_seg in sorted(results)],
        },
    )
    return index_path


def verify_split(processed_dir: str, split_name: str) -> list[str]:
    """Check that every frame named by the split index exists and loads. Returns a list of problems (empty if OK)."""
    index = _read_index(os.path.join(processed_dir, SPLIT_INDEX_FILES[split_name]))
    problems = []
    for rel_seg, frame_count in index['scans']:
        frame_dir = os.path.join(processed_dir, rel_seg)
        frame_paths = sorted(glob.glob(os.path.join(frame_dir, '*.npz')))
        if len(frame_paths) != frame_count:
            problems.append(f'{rel_seg}: {len(frame_paths)} frames, index says {frame_count}')
            continue
        for frame_path in frame_paths:
            try:
                with np.load(frame_path) as archive:
                    _ = archive['pos'].size, archive['intensity'].size
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as e:
                problems.append(f'{frame_path}: {e}')
    return problems


class SemanticWaymo(Dataset):
    proto_files: ClassVar[list[str]] = [
        'waymo_open_dataset/dataset_pb2.py',
        'waymo_open_dataset/label_pb2.py',
        'waymo_open_dataset/protos/keypoint_pb2.py',
        'waymo_open_dataset/protos/vector_pb2.py',
        'waymo_open_dataset/protos/map_pb2.py',
    ]

    def __init__(
        self,
        root: str,
        split='train',
        transform: Callable | None = None,
        log: bool = True,
        mix3d_p: float = 0,
        **kwargs,
    ) -> None:
        self.split = split
        self._split_names = [
            name
            for name, needed in (
                ('training', 'train' in split),
                ('validation', 'val' in split or 'test' in split),
                ('testing', 'pred' in split),
            )
            if needed
        ]
        super().__init__(root, transform=transform, log=log, **kwargs)
        self.mix3d_p = mix3d_p if 'train' in split else 0

        self._data: list[tuple[str, int]] = []
        for name in self._split_names:
            index = _read_index(os.path.join(self.processed_dir, SPLIT_INDEX_FILES[name]))
            for rel_seg, frame_count in index['scans']:
                self._data.extend((rel_seg, i) for i in range(frame_count))

    def len(self) -> int:
        return len(self._data)

    def get(self, idx: int) -> BaseData:
        rel_seg, frame_idx = self._data[idx]
        return _read_frame(_frame_path(self.processed_dir, rel_seg, frame_idx))

    def __getitem__(self, idx: int | np.integer | IndexType) -> "Dataset | BaseData":
        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):
            data = self.get(self.indices()[idx])

            if random.random() < self.mix3d_p:
                aug_data = self.get(random.choice(self.indices()))
                data.pos = torch.cat([data.pos, aug_data.pos], dim=0)
                data.intensity = torch.cat([data.intensity, aug_data.intensity], dim=0)
                data.y = torch.cat([data.y, aug_data.y], dim=0)

            data = data if self.transform is None else self.transform(data)
            return data

        else:
            return self.index_select(idx)

    def _download(self) -> None:
        if all(os.path.exists(path) for path in self.processed_paths):
            return
        super()._download()

    def download(self) -> None:
        PROTO_PACKAGE = "waymo-open-dataset-tf-2-12-0"

        with urllib.request.urlopen(f"https://pypi.org/pypi/{PROTO_PACKAGE}/json") as resp:
            data = json.load(resp)
        version = data["info"]["version"]
        wheel_url = next(f["url"] for f in data["releases"][version] if f["filename"].endswith(".whl"))

        with urllib.request.urlopen(wheel_url) as resp:
            wheel_bytes = resp.read()

        os.makedirs(os.path.join(self.raw_dir, 'waymo_open_dataset'), exist_ok=True)
        open(os.path.join(self.raw_dir, 'waymo_open_dataset', '__init__.py'), 'a').close()
        os.makedirs(os.path.join(self.raw_dir, 'waymo_open_dataset/protos'), exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as zf:
            for filename in self.proto_files:
                with open(os.path.join(self.raw_dir, filename), "wb") as f:
                    f.write(zf.read(filename))

        # Confirm all raw data files are present
        filenames = list(reversed(glob.glob(os.path.join(self.raw_dir, '*/*.tfrecord'), recursive=True)))
        if len(filenames) != 1150:
            warnings.warn("Not all files present in raw/", stacklevel=3)

    def process(self) -> None:
        warnings.warn(
            'Building SemanticWaymo from raw/ takes hours (~1.1 TB read). Prefer '
            '`python -m clouds.datasets.waymo process --root <root> --split <split>` for monitoring and --limit.',
            stacklevel=2,
        )
        test_ids = _load_test_ids(self.raw_paths[-1])
        for name in self._split_names:
            if os.path.exists(os.path.join(self.processed_dir, SPLIT_INDEX_FILES[name])):
                continue
            build_split(self.raw_dir, self.processed_dir, name, test_ids=test_ids)

    @property
    def raw_file_names(self) -> list[str]:
        return [
            *self.proto_files,
            'training/segment-15832924468527961_1564_160_1584_160_with_camera_labels.tfrecord',
            'validation/segment-17065833287841703_2980_000_3000_000_with_camera_labels.tfrecord',
            'testing/segment-39847154216997509_6440_000_6460_000_with_camera_labels.tfrecord',
            TEST_IDS_FILE,
        ]

    @property
    def processed_file_names(self) -> list[str]:
        return [SPLIT_INDEX_FILES[name] for name in self._split_names]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m clouds.datasets.waymo')
    commands = parser.add_subparsers(dest='command', required=True)

    splits = list(SPLIT_INDEX_FILES)
    build = commands.add_parser('process', help='Build processed/<split>/<segment>/*.npz from raw/<split>.')
    build.add_argument('--root', required=True, help='Dataset root containing raw/ and processed/')
    build.add_argument('--split', required=True, choices=splits)
    build.add_argument('--limit', type=int, default=None, help='Only the first N scans (smoke tests; index is partial)')
    build.add_argument('--workers', type=int, default=8)

    check = commands.add_parser('verify', help='Check that every indexed frame exists and loads; never reads raw/.')
    check.add_argument('--root', required=True)
    check.add_argument('--split', required=True, choices=splits)

    args = parser.parse_args(argv)
    processed_dir = os.path.join(args.root, 'processed')

    if args.command == 'process':
        raw_dir = os.path.join(args.root, 'raw')
        test_ids = _load_test_ids(os.path.join(raw_dir, TEST_IDS_FILE))
        log_path = os.path.join(processed_dir, f'build_{args.split}.log')
        os.makedirs(processed_dir, exist_ok=True)
        if args.limit is not None:
            print(f'warning: --limit {args.limit} writes a partial index; use a throwaway root', file=sys.stderr)
        build_split(
            raw_dir,
            processed_dir,
            args.split,
            test_ids=test_ids,
            workers=args.workers,
            limit=args.limit,
            log_path=log_path,
        )
        return 0

    problems = verify_split(processed_dir, args.split)
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f'{args.split}: {"FAILED" if problems else "OK"} ({len(problems)} problems)')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
