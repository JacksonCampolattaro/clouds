import os
import pickle
import random
import sys
from typing import Callable, Union

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset, download_url, extract_tar
from torch_geometric.data.data import BaseData
from torch_geometric.data.dataset import IndexType

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

ID_TO_Y_LUT = torch.full((max(ID_TO_Y.keys()) + 1,), -1, dtype=torch.long)
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
        pos, intensity = lidar_data[:, :3], lidar_data[:, 3:4] / 255

        if label_file := info.get('gt_segment_path', None):
            label_path = os.path.join(self.raw_dir, label_file)
            ids = torch.from_numpy(np.fromfile(label_path, dtype=np.uint8)).long()
            assert ids.max() < ID_TO_Y_LUT.size(0)
            y = ID_TO_Y_LUT[ids]
        else:
            y = torch.full((pos.size(0),), -1, dtype=torch.long)

        return Data(pos=pos, intensity=intensity, y=y)

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


if __name__ == '__main__':
    root = os.path.join(os.path.realpath(sys.argv[1]), 'SemanticNuScenes')
    print(root)
    dataset = SemanticNuScenes(root=root, split='val')
    print(len(dataset))
    for i in range(len(dataset)):
        print(dataset.get(i))
