import os
from typing import Callable, ClassVar

import h5py
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip

# Adapted from https://github.com/rubenwiersma/deltaconv/blob/master/experiments/datasets/scanobjectnn.py


class ScanObjectNN(InMemoryDataset):
    url =  "https://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip"

    class_names: ClassVar[list[str]] = [
        'bag',
        'bed',
        'bin',
        'box',
        'cabinets',
        'chair',
        'desk',
        'display',
        'door',
        'pillow',
        'shelves',
        'sink',
        'sofa',
        'table',
        'toilet',
    ]
    augmentation_variants: ClassVar[list[str | None]] = [None, 'PB_T25', 'PB_T25_R', 'PB_T50_R', 'PB_T50_RS']
    raw_file_dict: ClassVar[dict] = {
        None: ['training_objectdataset.h5', 'test_objectdataset.h5'],
        'PB_T25': ['training_objectdataset_augmented25_norot.h5', 'test_objectdataset_augmented25_norot.h5'],
        'PB_T25_R': ['training_objectdataset_augmented25rot.h5', 'test_objectdataset_augmented25rot.h5'],
        'PB_T50_R': ['training_objectdataset_augmentedrot.h5', 'test_objectdataset_augmentedrot.h5'],
        'PB_T50_RS': ['training_objectdataset_augmentedrot_scale75.h5', 'test_objectdataset_augmentedrot_scale75.h5'],
    }

    def __init__(
        self,
        root: str,
        split: str = 'trainval',
        background: bool = False,
        augmentation: str | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        **kwargs,
    ):
        assert augmentation in self.augmentation_variants
        self.augmentation = augmentation
        self.background = background
        self.bg_path = 'main_split' if background else 'main_split_nobg'

        super().__init__(root, transform, pre_transform, pre_filter, **kwargs)

        path = self.processed_paths[0] if 'train' in split else self.processed_paths[1]
        self.data, self.slices = torch.load(path, weights_only=False)
    
    @property
    def raw_file_names(self):
        return [os.path.join(self.bg_path, filename) for filename in self.raw_file_dict[self.augmentation]]

    @property
    def processed_file_names(self):
        bg_string = 'bg' if self.background else 'nobg'
        augmentation_string = self.augmentation or 'vanilla'
        folder = bg_string + '_' + augmentation_string
        return [os.path.join(folder, 'training.pt'), os.path.join(folder, 'test.pt')]

    def download(self):
        if (not os.path.exists(os.path.join(self.raw_dir, self.raw_file_names[0]))):
            path = download_url(self.url, self.raw_dir)
            extract_zip(path, self.raw_dir)

    def process(self):
        for raw_path, path in zip(self.raw_paths, self.processed_paths, strict=True):
            training = h5py.File(os.path.join(self.raw_dir, raw_path), 'r')
            data_list = []
            for i, pos in enumerate(training['data']):
                y = training['label'][i]
                data_list.append(Data(pos=torch.from_numpy(pos), y=torch.Tensor([y]).long()))

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            torch.save(self.collate(data_list), path)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

if __name__ == '__main__':
    root = os.path.realpath(os.path.join(os.path.dirname(__file__), 'data', 'ScanObjectNN'))
    dataset = ScanObjectNN(root=root, split='train')
    print(len(dataset))
    print(dataset.get(0))
