from itertools import chain

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.transforms import BaseTransform


class VoteAugmentations(BaseTransform):
    def __init__(self, augmentations: list[BaseTransform]):
        super().__init__()
        self.augmentations = augmentations
        assert len(augmentations)

    def forward(self, data: Data) -> Data:

        augmented_data = [aug(data) for aug in self.augmentations]

        if isinstance(data.batch, Tensor):
            # If the data is already batched, we'll need to re-build the indices from scratch
            augmented_data = list(chain(*[d.to_data_list() for d in augmented_data]))

        out, _, _ = collate(
            cls=type(data),
            data_list=augmented_data,
        )
        out.num_votes = len(self.augmentations)
        return out


class CombineVotes(BaseTransform):
    def forward(self, data: Data) -> Data:
        assert hasattr(data, 'num_votes')

        for store in data.node_stores:

            if hasattr(store, 'pred'):
                store.vote_pred = store.pred
                store.pred = store.pred.reshape(data.num_votes, -1, store.pred.size(-1)).mean(dim=0)
            if hasattr(store, 'y'):
                store.vote_y = store.y
                store.y = store.y.reshape(data.num_votes, -1)[0]

        return data
