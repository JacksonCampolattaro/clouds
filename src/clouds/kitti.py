import os
from typing import Callable, ClassVar

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

# Adapted from:
# https://github.com/Pointcept/Pointcept/blob/04a0232b70f5c7091ffdc6bfe7a476e3eb7daff2/pointcept/datasets/semantic_kitti.py#L22
#
ID_TO_Y = {
    0: -1,  # "unlabeled"
    1: -1,  # "outlier" mapped to "unlabeled"
    10: 0,  # "car"
    11: 1,  # "bicycle"
    13: 4,  # "bus" mapped to "other-vehicle"
    15: 2,  # "motorcycle"
    16: 4,  # "on-rails" mapped to "other-vehicle"
    18: 3,  # "truck"
    20: 4,  # "other-vehicle"
    30: 5,  # "person"
    31: 6,  # "bicyclist"
    32: 7,  # "motorcyclist"
    40: 8,  # "road"
    44: 9,  # "parking"
    48: 10,  # "sidewalk"
    49: 11,  # "other-ground"
    50: 12,  # "building"
    51: 13,  # "fence"
    52: -1,  # "other-structure" mapped to "unlabeled"
    60: 8,   # "lane-marking" to "road"
    70: 14,  # "vegetation"
    71: 15,  # "trunk"
    72: 16,  # "terrain"
    80: 17,  # "pole"
    81: 18,  # "traffic-sign"
    99: -1,  # "other-object" to "unlabeled"
    252: 0,  # "moving-car" to "car"
    253: 6,  # "moving-bicyclist" to "bicyclist"
    254: 5,  # "moving-person" to "person"
    255: 7,  # "moving-motorcyclist" to "motorcyclist"
    256: 4,  # "moving-on-rails" mapped to "other-vehicle"
    257: 4,  # "moving-bus" mapped to "other-vehicle"
    258: 3,  # "moving-truck" to "truck"
    259: 4,  # "moving-other"-vehicle to "other-vehicle"
}

ID_TO_Y_LUT = torch.arange(max(ID_TO_Y.keys()) + 1, dtype=torch.long)
for id, y in ID_TO_Y.items():
    ID_TO_Y_LUT[id] = y


YS_TO_IDS = {
    -1: -1,  # "unlabeled"
    0: 10,  # "car"
    1: 11,  # "bicycle"
    2: 15,  # "motorcycle"
    3: 18,  # "truck"
    4: 20,  # "other-vehicle"
    5: 30,  # "person"
    6: 31,  # "bicyclist"
    7: 32,  # "motorcyclist"
    8: 40,  # "road"
    9: 44,  # "parking"
    10: 48,  # "sidewalk"
    11: 49,  # "other-ground"
    12: 50,  # "building"
    13: 51,  # "fence"
    14: 70,  # "vegetation"
    15: 71,  # "trunk"
    16: 72,  # "terrain"
    17: 80,  # "pole"
    18: 81,  # "traffic-sign"
}

IDS_TO_LABELS = {
    -1: "unlabeled",
    0: "car",
    1: "bicycle",
    2: "motorcycle",
    3: "truck",
    4: "other-vehicle",
    5: "person",
    6: "bicyclist",
    7: "motorcyclist",
    8: "road",
    9: "parking",
    10: "sidewalk",
    11: "other-ground",
    12: "building",
    13: "fence",
    14: "vegetation",
    15: "trunk",
    16: "terrain",
    17: "pole",
    18: "traffic-sign",
}

class SemanticKITTI(Dataset):
    splits: ClassVar[dict[str, list[int]]] = dict(
        train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
        val=[8],
        test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    )

    def __init__(
        self,
        root: str,
        split='trainval',
        transform: Callable | None = None,
        log: bool = True,
        **kwargs,
    ):
        super().__init__(root, transform=transform, log=log)

        sequences = []
        for split_name, split_seq in SemanticKITTI.splits.items():
            if split_name in split:
                sequences += split_seq

        self._samples = []
        for seq in sequences:
            seq_dir = os.path.join(self.root, 'dataset', 'sequences', f'{seq:02d}')
            scans_dir, labels_dir = os.path.join(seq_dir, 'velodyne'), os.path.join(seq_dir, 'labels')
            assert os.path.exists(scans_dir) and os.path.exists(labels_dir)
            for scan_file in sorted(os.scandir(scans_dir), key=lambda e: e.name):
                self._samples.append(
                    (
                        # Scan
                        scan_file.path,
                        # Labels
                        os.path.join(labels_dir, os.path.splitext(scan_file.name)[0] + '.label'),
                    )
                )

    def len(self) -> int:
        return len(self._samples)

    def get(self, idx: int) -> Data:
        scan_path, label_path = self._samples[idx]

        # Load scan data
        scan = torch.from_numpy(np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4))
        pos, intensity = scan[:, :3], scan[:, 3:4]

        # Load labels
        ids = torch.from_numpy(np.fromfile(label_path, dtype=np.uint32) & 0xFFFF).long()
        y = ID_TO_Y_LUT[ids]

        return Data(pos=pos, intensity=intensity, y=y)


if __name__ == '__main__':
    # data.daic needs to be mounted with sshfs, this is a massive dataset!
    root = os.path.realpath(os.path.join(os.path.dirname(__file__), 'data.daic', 'SemanticKITTI'))
    dataset = SemanticKITTI(root=root, split='train')
    print(len(dataset))
    print(dataset.get(0))
