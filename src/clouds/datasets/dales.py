import os
import os.path as osp
import shutil
import tarfile
from collections.abc import Callable

import laspy
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from clouds.home import get_dataset_root

# Class names keyed by the raw 1-indexed LAS `classification` ids. Id 0 marks
# "unknown" points, kept for scene continuity but excluded from evaluation.
DALES_CLASS_NAMES = {
    0: 'unknown',
    1: 'ground',
    2: 'vegetation',
    3: 'cars',
    4: 'trucks',
    5: 'power lines',
    6: 'fences',
    7: 'poles',
    8: 'buildings',
}

# Remap the raw ids to the contiguous 0-based ids models are trained against,
# matching the other semantic-segmentation loaders in `clouds`.
ID_TO_Y = {raw: raw - 1 for raw in range(1, len(DALES_CLASS_NAMES))}

DALES_NUM_CLASSES = len(ID_TO_Y)  # excludes the ignored "unknown" points

# LAS `classification` is a single byte (0-255); anything outside `ID_TO_Y`
# maps to the ignored (-1) class rather than raising.
ID_TO_Y_LUT = torch.full((256,), -1, dtype=torch.long)
for _id, _y in ID_TO_Y.items():
    ID_TO_Y_LUT[_id] = _y

IDS_TO_LABELS = {y: DALES_CLASS_NAMES[i] for i, y in ID_TO_Y.items()}

# gmnn's data module addresses splits as 'train'/'trainval'/'val'/'test' (and
# probes for 'pred'). DALES only ships `train/` and `test/` tiles, so resolve
# those views to the two raw directories. There is no separate validation
# split; 'val' and 'test' share the held-out tiles and a single processed cache
# (the filename is keyed on the raw split).
SPLIT_TO_RAW = {
    'train': 'train',
    'trainval': 'train',
    'val': 'test',
    'test': 'test',
    'pred': 'test',
}


class DALES(InMemoryDataset):
    RAW_ARCHIVE_NAME = 'dales_semantic_segmentation_las.tar.gz'

    def __init__(
        self,
        root: str,
        split: str = 'train',
        chunk_size: float = 50.0,
        chunk_stride: float | None = None,
        min_points: int = 100,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        force_reload: bool = False,
    ) -> None:
        if split not in SPLIT_TO_RAW:
            raise ValueError(f"split must be one of {sorted(SPLIT_TO_RAW)}, got {split!r}")

        self.raw_split = SPLIT_TO_RAW[split]
        self.chunk_size = float(chunk_size)
        self.chunk_stride = float(chunk_stride) if chunk_stride is not None else float(chunk_size)
        self.min_points = int(min_points)

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if self.chunk_stride <= 0:
            raise ValueError(f"chunk_stride must be positive, got {chunk_stride}")

        super().__init__(
            root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        return [self.RAW_ARCHIVE_NAME]

    @property
    def processed_file_names(self) -> list[str]:
        # Encode every chunking parameter that changes the resulting data in the
        # filename, so that changing them triggers reprocessing instead of
        # silently loading stale chunks.
        tag = f"chunks_size{self.chunk_size:g}_stride{self.chunk_stride:g}_minpts{self.min_points}_{self.raw_split}.pt"
        return [tag]

    def download(self) -> None:
        raise FileNotFoundError(
            f"Missing raw archive {osp.join(self.raw_dir, self.RAW_ARCHIVE_NAME)}. "
            "Download the DALES semantic-segmentation LAS archive and place it there."
        )

    def _extract_archive(self, archive_path: str) -> None:
        train_dir = osp.join(self.raw_dir, 'train')
        test_dir = osp.join(self.raw_dir, 'test')
        if osp.isdir(train_dir) and osp.isdir(test_dir):
            return  # already extracted

        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=self.raw_dir, filter='data')

        # The official archive nests the tiles under `dales_las/`; flatten it so
        # `raw/train` and `raw/test` end up directly under the raw directory.
        nested_dir = osp.join(self.raw_dir, 'dales_las')
        if osp.isdir(nested_dir):
            for entry in os.listdir(nested_dir):
                shutil.move(osp.join(nested_dir, entry), osp.join(self.raw_dir, entry))
            os.rmdir(nested_dir)

        if not (osp.isdir(train_dir) and osp.isdir(test_dir)):
            raise RuntimeError(
                f"Extracted '{archive_path}' but could not locate 'train/' and "
                f"'test/' subdirectories under {self.raw_dir}. Please check the "
                "archive contents."
            )

    def process(self) -> None:
        self._extract_archive(osp.join(self.raw_dir, self.RAW_ARCHIVE_NAME))

        split_dir = osp.join(self.raw_dir, self.raw_split)
        if not osp.isdir(split_dir):
            raise FileNotFoundError(
                f"Expected raw split directory {split_dir} to exist after "
                "extraction; found nothing. Re-check the raw archive contents."
            )

        las_files = sorted(f for f in os.listdir(split_dir) if f.lower().endswith('.las'))
        if len(las_files) == 0:
            raise FileNotFoundError(f"No .las files found in {split_dir}.")

        data_list: list[Data] = []
        for las_name in las_files:
            las_path = osp.join(split_dir, las_name)
            data_list.extend(self._chunk_tile(las_path, tile_name=osp.splitext(las_name)[0]))

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])

    def _chunk_tile(self, las_path: str, tile_name: str) -> list[Data]:
        """Read one .las tile and split it into square (x, y) chunks.

        This follows the standard KPConv large-scene chunking approach: lay a
        regular grid over the tile's (x, y) bounding box with cell size
        `chunk_size` and step `chunk_stride` between cell origins, then assign
        every point to every grid cell whose square footprint contains it. With
        `chunk_stride < chunk_size` this naturally produces overlapping chunks
        (each point can fall into more than one chunk); with
        `chunk_stride == chunk_size` (the default) it produces a non-overlapping
        partition of the tile.
        """
        las = laspy.read(las_path)

        # Keep coordinates at float32 up front; the chunks are consumed as
        # float32 anyway.
        xyz = np.stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], axis=1).astype(np.float32)

        # DALES stores per-point semantic class in the standard LAS
        # `classification` field (0 = "unknown", 1-8 = the labeled classes).
        # Remap to contiguous 0-based ids, with -1 for the ignored "unknown"
        # points.
        raw_labels = torch.from_numpy(np.asarray(las.classification).astype(np.int64))
        labels = ID_TO_Y_LUT[raw_labels].numpy()

        # Intensity is present in the official DALES .las files and is a commonly
        # used feature; keep it as raw signal, any normalization belongs in a
        # transform.
        intensity = np.asarray(las.intensity).astype(np.float32)

        x, y = xyz[:, 0], xyz[:, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        # Grid cell origins: start at the tile's min corner and step by
        # `chunk_stride` until the cell would start past the tile's max corner.
        # `np.arange` with a small epsilon guards against float rounding adding
        # an origin exactly at the max corner (which would double-count boundary
        # points).
        eps = 1e-6
        x_starts = np.arange(x_min, x_max - eps, self.chunk_stride)
        y_starts = np.arange(y_min, y_max - eps, self.chunk_stride)

        chunks: list[Data] = []
        for x0 in x_starts:
            x1 = x0 + self.chunk_size
            in_x = (x >= x0) & (x < x1)
            if not np.any(in_x):
                continue
            for y0 in y_starts:
                y1 = y0 + self.chunk_size
                mask = in_x & (y >= y0) & (y < y1)

                if int(mask.sum()) < self.min_points:
                    continue

                data = Data(
                    pos=torch.from_numpy(xyz[mask]),
                    y=torch.from_numpy(labels[mask]),
                    intensity=torch.from_numpy(intensity[mask]),
                )
                # Bookkeeping useful for stitching predictions back onto the
                # original tile, or for debugging/visualizing chunks.
                data.tile_name = tile_name
                data.chunk_origin = torch.tensor([x0, y0], dtype=torch.float)

                chunks.append(data)

        return chunks


if __name__ == '__main__':
    from clouds.transforms import CenterPoints
    from clouds.visualization import show_data

    root = get_dataset_root('DALES')
    dataset = DALES(root=root, transform=CenterPoints([0, 1]))
    print(len(dataset))
    show_data(dataset[0])
