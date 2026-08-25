
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import one_hot
from clouds.transforms.apply_selection import apply_selection

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
        num_nodes = data.num_nodes
        new_num_nodes = num_nodes // data.num_votes
        assert new_num_nodes

        # Output will only have one vote's worth of nodes
        data.selection_index = torch.arange(new_num_nodes, dtype=torch.long, device=data.pred.device)
        out = apply_selection(data)

        # Predictions on the output will be the mean of the votes
        if self.combine == 'mean_logits':
            out.pred = data.pred.reshape(data.num_votes, -1, data.pred.size(-1)).mean(dim=0)
        elif self.combine == 'mean_probs':
            probs = torch.softmax(data.pred, dim=-1)
            mean_probs = probs.reshape(data.num_votes, -1, data.pred.size(-1)).mean(dim=0)
            out.pred = mean_probs.clamp_min(1e-8).log()
        elif self.combine == 'popularity':
            choices = one_hot(data.pred.argmax(dim=-1), num_classes=data.pred.size(-1), dtype=torch.float)
            popularities = choices.reshape(data.num_votes, -1, data.pred.size(-1)).mean(dim=0)
            # TODO: argmax chooses the first in case of ties; maybe we can improve on this?
            out.pred = popularities.clamp_min(1e-8).log()
        else:
            # TODO: mode prediction?
            raise ValueError(f"Unsupported combination type '{self.combine}'")

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.combine})"
