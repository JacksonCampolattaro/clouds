import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RandomJitter(BaseTransform):
    def __init__(self, sigma=0.01, clip=0.05):
        super().__init__()
        self.sigma, self.clip = sigma, clip

    def forward(self, data: Data) -> Data:
        if hasattr(data, 'pos'):
            if self.clip is None:
                data.pos += torch.empty_like(data.pos).normal_(std=self.sigma)
            else:
                data.pos += torch.empty_like(data.pos).normal_(std=self.sigma).clamp_(-self.clip, self.clip)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sigma={self.sigma}, clip={self.clip})'
