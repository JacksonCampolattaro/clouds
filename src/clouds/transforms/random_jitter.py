import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RandomJitter(BaseTransform):
    def __init__(self, sigma: float = 0.01, clip: float | None = 0.05) -> None:
        super().__init__()
        self.sigma, self.clip = sigma, clip

    def forward(self, data: Data) -> Data:
        for store in data.node_stores:
            pos = store.get('pos')
            if not isinstance(pos, Tensor):
                continue

            noise = torch.empty_like(pos).normal_(std=self.sigma)
            if self.clip is not None:
                noise = noise.clamp_(-self.clip, self.clip)
            store.pos = pos + noise

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sigma={self.sigma}, clip={self.clip})'
