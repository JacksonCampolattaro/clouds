import glob
import os
import pickle
from collections.abc import Callable

import numpy as np
import pandas
import torch
from rich import print, progress
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.io import fs

IDS_TO_LABELS = {
    0: 'ceiling',
    1: 'floor',
    2: 'wall',
    3: 'beam',
    4: 'column',
    5: 'window',
    6: 'door',
    7: 'chair',
    8: 'table',
    9: 'bookcase',
    10: 'sofa',
    11: 'board',
    12: 'clutter',
}

LABELS_TO_IDS = {
    'ceiling': 0,
    'floor': 1,
    'wall': 2,
    'beam': 3,
    'column': 4,
    'window': 5,
    'door': 6,
    'chair': 7,
    'table': 8,
    'bookcase': 9,
    'sofa': 10,
    'board': 11,
    'clutter': 12,
    'stairs': 12,  # stairs are also treated as clutter, since they appear so rarely
}

LABELS_TO_COLORS = {
    'clutter': [44, 44, 44],
    'window': [166, 206, 227],
    'door': [31, 120, 180],
    'wall': [178, 223, 138],
    'board': [51, 160, 44],
    'chair': [251, 154, 153],
    'sofa': [227, 26, 28],
    'column': [253, 191, 111],
    'beam': [255, 127, 0],
    'bookcase': [202, 178, 214],
    'table': [106, 61, 154],
    'ceiling': [255, 255, 153],
    'floor': [177, 89, 40],
}

LABELS_TO_COLORS_TABLE = torch.zeros([13, 3])
for label, color in LABELS_TO_COLORS.items():
    LABELS_TO_COLORS_TABLE[LABELS_TO_IDS[label], :] = torch.tensor(color) / 255

VALIDATION_ROOMS = [
    'hallway_1',
    'hallway_6',
    'hallway_11',
    'office_1',
    'office_6',
    'office_11',
    'office_16',
    'office_21',
    'office_26',
    'office_31',
    'office_36',
    'WC_2',
    'storage_1',
    'storage_5',
    'conferenceRoom_2',
    'auditorium_1',
]


def load_s3dis_room(room_path: str) -> Data:
    scan_files = glob.glob(os.path.join(room_path, 'Annotations/*.txt'))

    x_scans, y_scans = [], []
    for scan in scan_files:
        try:
            # Determine ID based on scan name
            label = scan.split('/')[-1].split('_')[0]
            label_id = LABELS_TO_IDS[label]

            # Data is extracted as one large table
            x = torch.from_numpy(
                pandas.read_csv(scan, delimiter=' ', dtype=np.float32).to_numpy()
            )
            assert x.shape[-1] == 6

            # Drop invalid rows
            invalid_rows = torch.isnan(x).any(dim=-1)
            if invalid_rows.any():
                x = x[~invalid_rows]

            x_scans.append(x)
            y_scans.append(torch.full_like(x[:, 0], label_id, dtype=torch.long))

        except:  # noqa: E722
            print(f"\nEncountered invalid data in file '{room_path}', skipping")

    # Merge all loaded data
    x = torch.cat(x_scans)
    y = torch.cat(y_scans)

    # Split the x tensor into several properties
    pos = x[:, :3]
    pos[:, [0, 1]] = pos[:, [1, 0]]
    color = x[:, 3:] / 255.0
    return Data(pos=pos, color=color, y=y)


class S3DIS(InMemoryDataset):
    url = 'https://cvg-data.inf.ethz.ch/s3dis/Stanford3dDataset_v1.2_Aligned_Version.zip'

    def __init__(
        self,
        root,
        fold: int = 5,
        split='trainval',
        num_loops: int = 1,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        **kwargs,
    ):
        self.split = split
        self.test_area = 6 if fold == 0 else fold
        super().__init__(root=root, transform=transform, pre_filter=pre_filter, pre_transform=pre_transform, **kwargs)
        self.data, self.slices = torch.load(os.path.join(self.processed_paths[-1]), weights_only=False)

        self._indices = []
        with open(os.path.join(self.processed_dir, f'splits_area{self.test_area}.pkl'), 'rb') as splits_file:
            splits = pickle.load(splits_file)
            for split_name, indices in splits.items():
                if split_name in split:
                    self._indices += indices
         
        # Having the indices in a canonical order can be useful for testing
        self._indices = sorted(self._indices)

    @property
    def raw_file_names(self) -> list[str]:
        # If the first file is present, we assume the entire dataset was unpacked
        return ['Area_1/office_1/office_1.txt']

    @property
    def processed_file_names(self) -> list[str]:
        return [f'splits_area{i}.pkl' for i in range(1, 7)] + ['data.pt']

    def download(self) -> None:
        # Download the zip file
        path = download_url(self.url, self.root)

        # Extract the zip into the current directory
        extract_zip(path, self.root)
        unzipped_path = os.path.splitext(self.url.split('/')[-1])[0]

        # The unzipped data is our raw directory
        fs.rm(self.raw_dir)
        os.rename(os.path.join(self.root, unzipped_path), self.raw_dir)
        # os.unlink(path) # todo: why delete the zip?

    def process(self) -> None:
        filenames = glob.glob(os.path.join(self.raw_dir, '*/*'), recursive=True)
        splits: dict[int, dict[str, list[int]]] = {
            a: {'train': [], 'val': [], 'test': []} for a in range(1, 7)
        }
        data_list = []

        def load_room(idx: int, path: str):
            area_name, room = path.split('/')[-2:]
            area_number = int(area_name.split('_')[1])

            # Read & fuse the room
            data = load_s3dis_room(path)

            # Filter & transform each room
            if self.pre_filter is not None and not self.pre_filter(data):
                return
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

            # Assign a split for the room
            for area, area_splits in splits.items():
                if area_number == area:
                    area_splits['test'].append(idx)
                elif room in VALIDATION_ROOMS:
                    area_splits['val'].append(idx)
                else:
                    area_splits['train'].append(idx)

        # Load all rooms
        for i, path in progress.track(enumerate(filenames), total=len(filenames), description="Loading S3DIS data"):
            load_room(i, path)

        # Save each split
        torch.save(self.collate(data_list), self.processed_paths[-1])
        for area, area_splits in splits.items():
            splits_path = os.path.join(self.processed_dir, f'splits_area{area}.pkl')
            with open(splits_path, 'wb') as f:
                pickle.dump(area_splits, f)


if __name__ == '__main__':
    root = os.path.realpath(os.path.join(os.path.dirname(__file__), 'data', 'S3DIS'))
    dataset = S3DIS(root=root, split='val')
    print(len(dataset))
    print(dataset.get(0))
