import os
import pickle
import sys
from typing import Callable

import numpy as np
import torch
from torch_geometric.data import Data, Dataset, download_url, extract_tar
from torch_geometric.data.data import BaseData

# Adapted from:
# https://github.com/VisualComputingInstitute/ditr/blob/main/pointcept/datasets/nuscenes.py
ID_TO_Y = {
    0: -1,
    1: -1,
    2: 6,
    3: 6,
    4: 6,
    5: -1,
    6: 6,
    7: -1,
    8: -1,
    9: 0,
    10: -1,
    11: -1,
    12: 7,
    13: -1,
    14: 1,
    15: 2,
    16: 2,
    17: 3,
    18: 4,
    19: -1,
    20: -1,
    21: 5,
    22: 8,
    23: 9,
    24: 10,
    25: 11,
    26: 12,
    27: 13,
    28: 14,
    29: -1,
    30: 15,
    31: -1,
}

ID_TO_Y_LUT = torch.arange(max(ID_TO_Y.keys()) + 1, dtype=torch.long)
for id, y in ID_TO_Y.items():
    ID_TO_Y_LUT[id] = y


class SemanticNuScenes(Dataset):
    url = 'https://huggingface.co/datasets/Pointcept/nuscenes-compressed/resolve/main/nuscenes.tar.gz?download=true'

    def __init__(
        self,
        root: str,
        split='trainval',
        transform: Callable | None = None,
        log: bool = True,
        mix3d_p: float = 0,
        **kwargs,
    ):
        super().__init__(root, transform=transform, log=log)
        self.mix3d_p = mix3d_p if 'train' in split else 0

        self.data_list = []
        if 'train' in split:
            with open(self.processed_paths[0], 'rb') as f:
                self.data_list.extend(pickle.load(f))
        if 'val' in split or 'test' in split:
            with open(self.processed_paths[1], 'rb') as f:
                self.data_list.extend(pickle.load(f))
        if 'pred' in split:
            with open(self.processed_paths[2], 'rb') as f:
                self.data_list.extend(pickle.load(f))

    def len(self) -> int:
        return len(self.data_list)

    @property
    def raw_file_names(self) -> list[str]:
        return ['nuscenes.tar.gz', 'LICENSE']

    @property
    def processed_file_names(self) -> list[str]:
        return [
            'info/nuscenes_infos_10sweeps_train.pkl',
            'info/nuscenes_infos_10sweeps_val.pkl',
            'info/nuscenes_infos_10sweeps_test.pkl',
        ]

    def download(self) -> None:
        download_url(self.url, self.raw_dir)

    def process(self) -> None:
        extract_tar(self.raw_paths[0], self.processed_dir, log=True)

    def get(self, idx: int) -> BaseData:
        info = self.data_list[idx]
        lidar_path = os.path.join(self.raw_dir, info['lidar_path'])

        lidar_data = torch.from_numpy(np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5))
        data = Data(pos=lidar_data[:, :3], intensity=lidar_data[:, 3:4] / 255)

        if label_file := info.get('gt_segment_path', None):
            label_path = os.path.join(self.raw_dir, label_file)
            ids = torch.from_numpy(np.fromfile(label_path, dtype=np.uint8)).long()
            data.y = ID_TO_Y_LUT[ids]
        else:
            data.y = torch.full((data.pos.size(0),), -1, dtype=torch.long)

        return data

if __name__ == '__main__':
    root = os.path.join(os.path.realpath(sys.argv[1]), 'SemanticNuScenes')
    print(root)
    dataset = SemanticNuScenes(root=root, split='train')
    print(len(dataset))
    print(dataset.get(0))
