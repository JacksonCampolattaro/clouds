import os
from typing import Callable, ClassVar

import h5py
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.io import fs


class ModelNet40(InMemoryDataset):

    # Thanks to Msun for mirroring this data on HuggingFace!
    url = 'https://huggingface.co/datasets/Msun/modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip'

    raw_file_dict: ClassVar[dict] = {
        'train': [f'ply_data_train{i}.h5' for i in range(5)],
        'test': [f'ply_data_test{i}.h5' for i in range(2)],
    }


    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        **kwargs,
    ):
        super().__init__(root, transform, pre_transform, pre_filter, **kwargs)
        self.data, self.slices = torch.load(os.path.join(self.processed_dir, f'{split}.pt'), weights_only=False)

    @property
    def raw_file_names(self):
        return [*self.raw_file_dict['train'], *self.raw_file_dict['test']]

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']

    def download(self):
        path = download_url(self.url, self.root + '/')
        extract_zip(path, self.root)
        unzipped_path = os.path.splitext(self.url.split('/')[-1])[0]
        fs.rm(self.raw_dir)
        os.rename(os.path.join(self.root, unzipped_path), self.raw_dir)

    def process(self):
        for split, file_names in self.raw_file_dict.items():
            data_list = []
            for file_name in file_names:
                raw_path = os.path.join(self.raw_dir, file_name)
                f = h5py.File(os.path.join(self.raw_dir, raw_path), 'r')
                # This is a roundabout way of doing this, but isn't a performance problem
                f_pos = torch.from_numpy(f['data'][:]).float()
                f_y = torch.from_numpy(f['label'][:]).long()
                for pos, y in zip(f_pos, f_y, strict=True):
                    data_list.append(Data(pos=pos, y=y))

            torch.save(self.collate(data_list), os.path.join(self.processed_dir, f'{split}.pt'))


if __name__ == '__main__':
    root = os.path.realpath(os.path.join(os.path.dirname(__file__), 'data', 'ModelNet40'))
    dataset = ModelNet40(root=root)
    print(len(dataset))
    print(dataset.get(0))
