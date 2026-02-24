import os
import os.path as osp
import shutil

import gdown
import rich
import torch
from torch import Tensor
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_off

# Adapted from:
# https://github.com/rubenwiersma/hsn/blob/master/datasets/faust.py


class FAUSTRemeshed(InMemoryDataset):
    r"""The remeshed FAUST humans dataset from the paper `Multi-directional
    Geodesic Neural Networks via Equivariant Convolution`
    containing 100 watertight meshes representing 10 different poses for 10
    different subjects.

    .. note::

        Data objects hold mesh faces instead of edge indices.
        To convert the mesh to a graph, use the
        :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
        To convert the mesh to a point cloud, use the
        :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
        sample a fixed number of points on the mesh faces according to their
        face area.

    Args:
        root (string): Root directory where the dataset should be saved.
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    # url = 'https://surfdrive.surf.nl/files/index.php/s/KLSxAN0QEsfJuBV/download'
    url = 'https://drive.google.com/uc?id=1C-9GFsTl5xwa0RUmC_m1nnj87QUguh6j'

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None, num_loops: int = 10):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[-1]
        self.data, self.slices = torch.load(path, weights_only=False)

        self.num_loops = num_loops if train else 1

    def len(self) -> int:
        return super().len() * self.num_loops

    def get(self, idx):
        return super().get(idx % super().len())

    @property
    def reference_mesh(self):
        return self.get(0)

    @property
    def raw_file_names(self):
        return [f'tr_reg_{i:03d}.off' for i in range(100)]

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        if not os.path.exists(self.raw_paths[0]):
            path = gdown.download(self.url, self.root + '/')
            extract_zip(path, self.root)
            os.rename(os.path.join(self.root, 'OFF'), self.raw_dir)

    def process(self):

        data_list = []
        for path in self.raw_paths:
            data = read_off(path)
            # Convert to Z-up
            assert isinstance(data.pos, Tensor)
            data.pos = data.pos[:, [2, 0, 1]]
            data_list.append(data)
            # Create labels
            data.y = torch.arange(data.pos.size(0), dtype=torch.long)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        torch.save(self.collate(data_list[:79]), self.processed_paths[0])
        torch.save(self.collate(data_list[79:]), self.processed_paths[1])

if __name__ == '__main__':
    root = os.path.realpath(os.path.join(os.path.dirname(__file__), 'data', 'FAUSTRemeshed'))
    dataset = FAUSTRemeshed(root=root)
    print(len(dataset))
    print(dataset.get(0))
