import os
import os.path as osp
import sys
import tarfile
from collections.abc import Callable

import laspy
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

DALES_NUM_CLASSES = 8

DALES_CLASS_NAMES = {
    0: "unknown",
    1: "ground",
    2: "vegetation",
    3: "cars",
    4: "trucks",
    5: "power lines",
    6: "fences",
    7: "poles",
    8: "buildings",
}


class DALES(InMemoryDataset):

    RAW_ARCHIVE_NAME = "dales_semantic_segmentation_las.tar.gz"

    def __init__(
        self,
        root: str,
        split: str = "train",
        chunk_size: float = 50.0,
        chunk_stride: float | None = None,
        min_points: int = 100,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        force_reload: bool = False,
    ):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.split = split
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
        return [osp.join("train"), osp.join("test")]

    @property
    def processed_file_names(self) -> list[str]:
        # Encode every chunking parameter that changes the resulting data
        # in the filename, so that changing them triggers reprocessing
        # instead of silently loading stale chunks.
        tag = (
            f"chunks_size{_fmt_num(self.chunk_size)}"
            f"_stride{_fmt_num(self.chunk_stride)}"
            f"_minpts{self.min_points}"
            f"_{self.split}.pt"
        )
        return [tag]

    def download(self):
        archive_path = osp.join(self.root, self.RAW_ARCHIVE_NAME)
        if not osp.exists(archive_path):
            raise FileNotFoundError(f"Missing raw archive: {archive_path}.")

        self._extract_archive(archive_path)


    def _extract_archive(self, archive_path: str):
        train_dir = osp.join(self.raw_dir, "train")
        test_dir = osp.join(self.raw_dir, "test")
        if osp.isdir(train_dir) and osp.isdir(test_dir):
            return  # already extracted

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=self.raw_dir)

        for entry in os.listdir(self.raw_dir):
            entry_path = osp.join(self.raw_dir, entry)
            if not osp.isdir(entry_path) or entry == "train" or entry == "test":
                continue

        if not (osp.isdir(train_dir) and osp.isdir(test_dir)):
            raise RuntimeError(
                f"Extracted '{archive_path}' but could not locate 'train/' and "
                f"'test/' subdirectories under {self.raw_dir}. Please check the "
                "archive contents."
            )

    def process(self):
        split_dir = osp.join(self.raw_dir, self.split)
        if not osp.isdir(split_dir):
            raise FileNotFoundError(
                f"Expected raw split directory {split_dir} to exist after "
                "extraction; found nothing. Re-check the raw archive contents."
            )

        las_files = sorted(
            f for f in os.listdir(split_dir) if f.lower().endswith(".las")
        )
        if len(las_files) == 0:
            raise FileNotFoundError(f"No .las files found in {split_dir}.")

        data_list: list[Data] = []
        for las_name in las_files:
            las_path = osp.join(split_dir, las_name)
            tile_data = self._read_las_tile(las_path)
            chunks = self._chunk_tile(tile_data, tile_name=osp.splitext(las_name)[0])
            data_list.extend(chunks)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])

    def _read_las_tile(self, las_path: str) -> dict:
        """Read one .las tile into plain numpy arrays."""
        las = laspy.read(las_path)

        xyz = np.stack(
            [np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], axis=1
        ).astype(np.float64)

        # DALES stores per-point semantic class in the standard LAS
        # `classification` field.
        labels = np.asarray(las.classification).astype(np.int64)

        # Intensity is present in the official DALES .las files and is a
        # commonly used feature; keep it as raw signal, any normalization
        # belongs in a transform.
        intensity = None
        if hasattr(las, "intensity"):
            intensity = np.asarray(las.intensity).astype(np.float32)

        return {"xyz": xyz, "labels": labels, "intensity": intensity}

    def _chunk_tile(self, tile_data: dict, tile_name: str) -> list[Data]:
        """Split one tile into square (x, y) chunks on a regular grid.

        This follows the standard KPConv large-scene chunking approach:
        lay a regular grid over the tile's (x, y) bounding box with cell
        size `chunk_size` and step `chunk_stride` between cell origins,
        then assign every point to every grid cell whose square footprint
        contains it. With `chunk_stride < chunk_size` this naturally
        produces overlapping chunks (each point can fall into more than
        one chunk); with `chunk_stride == chunk_size` (the default) it
        produces a non-overlapping partition of the tile.
        """
        xyz = tile_data["xyz"]
        labels = tile_data["labels"]
        intensity = tile_data["intensity"]

        x, y = xyz[:, 0], xyz[:, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        # Grid cell origins: start at the tile's min corner and step by
        # `chunk_stride` until the cell would start past the tile's max
        # corner. `np.arange` with a small epsilon guards against float
        # rounding dropping the last valid origin.
        eps = 1e-6
        x_starts = np.arange(x_min, x_max - eps, self.chunk_stride)
        y_starts = np.arange(y_min, y_max - eps, self.chunk_stride)
        if len(x_starts) == 0:
            x_starts = np.array([x_min])
        if len(y_starts) == 0:
            y_starts = np.array([y_min])

        chunks: list[Data] = []
        for i, x0 in enumerate(x_starts):
            x1 = x0 + self.chunk_size
            in_x = (x >= x0) & (x < x1)
            if not np.any(in_x):
                continue
            for j, y0 in enumerate(y_starts):
                y1 = y0 + self.chunk_size
                mask = in_x & (y >= y0) & (y < y1)

                n_points = int(mask.sum())
                if n_points < self.min_points:
                    continue

                pos = torch.from_numpy(xyz[mask]).float()
                y_t = torch.from_numpy(labels[mask]).long()

                data = Data(pos=pos, y=y_t)
                if intensity is not None:
                    data.intensity = torch.from_numpy(intensity[mask]).float()

                # Bookkeeping useful for stitching predictions back onto
                # the original tile, or for debugging/visualizing chunks.
                data.tile_name = tile_name
                data.chunk_origin = torch.tensor([x0, y0], dtype=torch.float)
                data.chunk_size = torch.tensor(self.chunk_size, dtype=torch.float)

                chunks.append(data)

        return chunks


def _fmt_num(value: float) -> str:
    """Format a float for use in a filename: '50' instead of '50.0' when
    the value is integral, otherwise a compact decimal representation
    with dots replaced (dots are filesystem-safe, but this keeps things
    tidy and avoids accidental double extensions)."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


if __name__ == '__main__':
    from clouds.show import show_data
    from clouds.transforms import CenterPoints
    root = os.path.join(os.path.realpath(sys.argv[1]), 'DALES')
    dataset = DALES(root=root, transform=CenterPoints([0, 1]))
    print(len(dataset))
    show_data(dataset[0])
