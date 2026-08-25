
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import one_hot


class VoteAugmentations(BaseTransform):
    def __init__(self, augmentations: list[BaseTransform]):
        super().__init__()
        self.augmentations = augmentations
        assert len(augmentations)

    def forward(self, data: Data) -> Data:

        augmented_data = [aug(data) for aug in self.augmentations]

        if isinstance(data.batch, Tensor):
            augmented_data = [aug(d.clone()) for aug in self.augmentations for d in data.to_data_list()]
        else:
            augmented_data = [aug(data.clone()) for aug in self.augmentations]

        out, _, _ = collate(
            cls=type(augmented_data[0]),
            data_list=augmented_data,
        )
        out.num_votes = len(self.augmentations)
        return out

    def __repr__(self) -> str:
        # TODO: make this prettier?
        return f"{self.__class__.__name__}({self.augmentations})"


class CombineVotes(BaseTransform):
    def __init__(self, combine: str = 'mean_logits'):
        super().__init__()
        self.combine = combine

    def forward(self, data: Data) -> Data:
        assert hasattr(data, 'num_votes')

        for store in data.node_stores:

            if hasattr(store, 'pred'):
                store.vote_pred = store.pred
                if self.combine == 'mean_logits':
                    store.pred = store.pred.reshape(data.num_votes, -1, store.pred.size(-1)).mean(dim=0)
                elif self.combine == 'mean_probs':
                    probs = torch.softmax(store.pred)
                    mean_probs = probs.reshape(data.num_votes, -1, store.pred.size(-1)).mean(dim=0)
                    store.pred = mean_probs.clamp_min(1e-8).log()
                elif self.combine == 'popularity':
                    choices = one_hot(store.pred.argmax(dim=-1), num_classes=store.pred.size(-1), dtype=torch.float)
                    popularities = choices.reshape(data.num_votes, -1, store.pred.size(-1)).mean(dim=0)
                    # TODO: argmax chooses the first in case of ties; maybe we can improve on this?
                    store.pred = popularities.clamp_min(1e-8).log()
                else:
                    # TODO: mode prediction?
                    raise ValueError(f"Unsupported combination type '{self.combine}'")
            if hasattr(store, 'y'):
                store.vote_y = store.y
                store.y = store.y.reshape(data.num_votes, -1)[0]

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.combine})"
