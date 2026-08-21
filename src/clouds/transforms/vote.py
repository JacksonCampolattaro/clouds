import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.transforms import BaseTransform


class VoteAugmentations(BaseTransform):
    def __init__(self, augmentations: list[BaseTransform]):
        self.augmentations = augmentations

    def forward(self, data: Data) -> Data:
        # This will introduce a batch dimension, doesn't play nice with batched input!
        if isinstance(data.batch, Tensor):
            assert torch.all(data.batch == data.batch[0])
            data.batch = None

        # TODO: apply all augmentations
         
        return collate(
            cls=type(data)
            # TODO
        )

# TODO: combine votes transform
