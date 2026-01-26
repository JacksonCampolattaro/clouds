import os
from typing import Callable, ClassVar

import gdown
import h5py
import torch
from torch_geometric.data import Data, InMemoryDataset, extract_zip


class ShapeNetC(InMemoryDataset):
    # see: https://pointcloud-c.github.io/home.html
    url = 'https://drive.google.com/uc?id=1OqdJ9q1GjVu46QdPC4coa8M5e-xOrJ_g'

    splits: ClassVar[list[str]] = [
        'clean',
        *[f'scale_{level}' for level in range(5)],
        *[f'jitter_{level}' for level in range(5)],
        *[f'rotate_{level}' for level in range(5)],
        *[f'dropout_global_{level}' for level in range(5)],
        *[f'dropout_local_{level}' for level in range(5)],
        *[f'add_global_{level}' for level in range(5)],
        *[f'add_local_{level}' for level in range(5)],
    ]

    def __init__(
        self,
        root: str,
        split: str = 'clean',
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        **kwargs,
    ):
        super().__init__(root, transform, pre_transform, pre_filter, **kwargs)
        assert 'train' not in split
        split = 'clean' if split == 'val' else split
        self.data, self.slices = torch.load(os.path.join(self.processed_dir, f'{split}.pt'), weights_only=False)


    @property
    def raw_file_names(self):
        return [f'{split}.h5' for split in self.splits]

    @property
    def processed_file_names(self):
        return [f'{split}.pt' for split in self.splits]

    def download(self):
        if (not os.path.exists(os.path.join(self.raw_dir, self.raw_file_names[0]))):
            path = gdown.download(self.url, self.root + '/')
            extract_zip(path, self.root)
            os.rename(os.path.join(self.root, 'shapenet_c'), self.raw_dir)

    def process(self):
        for raw_path, path in zip(self.raw_paths, self.processed_paths, strict=False):
            f = h5py.File(os.path.join(self.raw_dir, raw_path), 'r')

            pos, cat, label = f['data'][:].astype('float32'), f['label'][:].astype('int64'), f['pid'][:].astype('int64')

            data_list = []
            for data_pos, data_label, data_cat in zip(pos, label, cat, strict=True):
                # data_pos[:, [1, 2]] = data_pos[:, [2, 1]]
                data_list.append(
                    Data(
                        pos=torch.from_numpy(data_pos),
                        y=torch.Tensor(data_label).long(),
                        category=torch.Tensor(data_cat),
                    )
                )

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            torch.save(self.collate(data_list), path)



if __name__ == '__main__':
    root = os.path.realpath(os.path.join(os.path.dirname(__file__), 'data', 'ShapeNetC'))
    dataset = ShapeNetC(root=root, split='clean')
    print(len(dataset))
    print(dataset.get(0))
