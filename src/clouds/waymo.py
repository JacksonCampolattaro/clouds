import glob
import io
import json
import os
import random
import sys
import urllib
import zipfile
import zlib
from typing import Callable, ClassVar, Union

import numpy as np
import torch
from rich import print, progress
from tfrecord import tfrecord_iterator
from torch import Tensor
from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import BaseData, Data
from torch_geometric.data.dataset import IndexType


def _get_rotation_matrix(roll, pitch, yaw):
  cos_roll, sin_roll = torch.cos(roll), torch.sin(roll)
  cos_pitch, sin_pitch = torch.cos(pitch), torch.sin(pitch)
  cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)

  ones = torch.ones_like(yaw)
  zeros = torch.zeros_like(yaw)

  r_roll = torch.stack([
      ones, zeros, zeros,
      zeros, cos_roll, -sin_roll,
      zeros, sin_roll, cos_roll,
  ], axis=-1).reshape(*yaw.shape, 3, 3)
  r_pitch = torch.stack([
      cos_pitch, zeros, sin_pitch,
      zeros, ones, zeros,
      -sin_pitch, zeros, cos_pitch,
  ], axis=-1).reshape(*yaw.shape, 3, 3)
  r_yaw = torch.stack([
      cos_yaw, -sin_yaw, zeros,
      sin_yaw, cos_yaw, zeros,
      zeros, zeros, ones,
  ], axis=-1).reshape(*yaw.shape, 3, 3)

  return r_yaw @ r_pitch @ r_roll


def _get_transform(rotation, translation):
    n = rotation.shape[-1]
    batch_shape = rotation.shape[:-2]
    transform = torch.zeros((*batch_shape, n + 1, n + 1), dtype=rotation.dtype)
    transform[..., :n, :n] = rotation
    transform[..., :n, n] = translation
    transform[..., n, n] = 1.0
    return transform


def _compute_inclination(inclination_min, inclination_max, height):
    diff = inclination_max - inclination_min
    indices = torch.arange(height, dtype=torch.float)
    return (0.5 + indices) / height * diff + inclination_min


def _range_image_to_point_image(range_image, extrinsic, inclination, pixel_pose=None, frame_pose=None):
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
        split='trainval',
        transform: Callable | None = None,
        log: bool = True,
        mix3d_p: float = 0,
        **kwargs,
    ):
        super().__init__(root, transform=transform, log=log, **kwargs)
        self.mix3d_p = mix3d_p if 'train' in split else 0

        # Identify relevant data
        splits = torch.load(self.processed_paths[0], weights_only=False)
        self._data = []
        if 'train' in split:
            self._data.extend(splits['training'])
        if 'val' in split:
            self._data.extend(splits['validation'])
        if 'test' in split or 'pred' in split:
            self._data.extend(splits['testing'])

    def len(self):
        return len(self._data)

    def get(self, idx: int) -> BaseData:
        frame_id = self._data[idx]
        data = torch.load(os.path.join(self.processed_dir, f'{frame_id}.pt'), weights_only=False)
        return data

    def __getitem__(self, idx: Union[int, np.integer, IndexType]) -> Union['Dataset', BaseData]:
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


    def download(self):
        PROTO_PACKAGE = "waymo-open-dataset-tf-2-12-0"

        with urllib.request.urlopen(f"https://pypi.org/pypi/{PROTO_PACKAGE}/json") as resp:
            data = json.load(resp)
        version = data["info"]["version"]
        wheel_url = next(f["url"] for f in data["releases"][version] if f["filename"].endswith(".whl"))

        with urllib.request.urlopen(wheel_url) as resp:
            wheel_bytes = resp.read()

        os.makedirs(os.path.join(self.raw_dir, 'waymo_open_dataset'), exist_ok=True)
        open(os.path.join(self.processed_dir, 'waymo_open_dataset', '__init__.py'), 'a').close()
        os.makedirs(os.path.join(self.raw_dir, 'waymo_open_dataset/protos'), exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as zf:
            for filename in self.proto_files:
                with open(os.path.join(self.raw_dir, filename), "wb") as f:
                    f.write(zf.read(filename))

    def process(self) -> None:

        sys.path.append(self.raw_dir)
        from waymo_open_dataset.dataset_pb2 import Frame, LaserName, MatrixFloat, MatrixInt32

        def _decompress_tensor(v, dtype=torch.float32) -> Tensor:
            matrix = MatrixFloat() if dtype == torch.float32 else MatrixInt32()
            matrix.ParseFromString(zlib.decompress(v))
            return torch.tensor(matrix.data, dtype=dtype).view(*matrix.shape.dims)

        def _frame_to_data(frame) -> Data:

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
                y=torch.cat(labels, dim=0)[:, 1] if labels else None,
            )

        # Identify the test set
        test_frame_ids = set(x.rstrip() for x in (open(self.raw_paths[-1], "r").readlines()))

        # Identify raw data files
        filenames = list(reversed(glob.glob(os.path.join(self.raw_dir, '*/*.tfrecord'), recursive=True)))
        assert len(filenames) == 1150

        # Collect frames
        split_ids = dict(training=[], validation=[], testing=[])
        for f in progress.track(filenames, description="Loading Waymo data", show_speed=True):
            split = os.path.split(os.path.split(f)[0])[-1]
            for raw_record in tfrecord_iterator(f):
                frame = Frame()
                frame.ParseFromString(raw_record)

                # Skip unlabeled frames, unless they're in the test set
                frame_id = f'{frame.context.name},{frame.timestamp_micros}'
                if not len(frame.lasers[0].ri_return1.segmentation_label_compressed) and frame_id not in test_frame_ids:
                    continue

                data = _frame_to_data(frame)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                split_ids[split].append(frame_id)
                torch.save(data, os.path.join(self.processed_dir, f'{frame_id}.pt'))

        torch.save(split_ids, os.path.join(self.processed_dir, 'splits.pt'))


    @property
    def raw_file_names(self) -> list[str]:
        return [
            *self.proto_files,
            'training/segment-15832924468527961_1564_160_1584_160_with_camera_labels.tfrecord',
            'validation/segment-17065833287841703_2980_000_3000_000_with_camera_labels.tfrecord',
            'testing/segment-39847154216997509_6440_000_6460_000_with_camera_labels.tfrecord',
            '3d_semseg_test_set_frames.txt',
        ]

    @property
    def processed_file_names(self) -> list[str]:
        return ['splits.pt']

if __name__ == '__main__':
    root = os.path.join(os.path.realpath(sys.argv[1]), 'SemanticWaymo')
    print(root)
    dataset = SemanticWaymo(root=root, split='val')
    for i, data in enumerate(dataset):

        import polyscope
        polyscope.init()
        polyscope.set_up_dir('z_up')
        c = polyscope.register_point_cloud('waymo', data.pos)
        c.add_scalar_quantity('y', data.y)
        c.add_scalar_quantity('intensity', data.intensity.flatten())
        polyscope.show()

        print(i, data)



