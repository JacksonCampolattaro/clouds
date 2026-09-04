
import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.collate import collate
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import one_hot


class VoteAugmentations(BaseTransform):
    def __init__(self, augmentations: list[BaseTransform]):
        super().__init__()
        self.augmentations = augmentations
        assert len(augmentations)

    def forward(self, data: Data) -> Data:


        if isinstance(data.batch, Tensor):
            split_data = data.to_data_list()
            augmented_data = []
            for d in split_data:
                for store in d.node_stores:
                    store.batch, store.ptr = None, None
                batch_augmented_data = [aug(d.clone()) for aug in self.augmentations]
                augmented_data.extend(batch_augmented_data)
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
        batch_size = (
            data.batch_size
            if hasattr(data, 'batch_size') #
            else (data.ptr.size(0) - 1 if hasattr(data, 'ptr') else None)
        )
        new_batch_size = batch_size // data.num_votes if batch_size else None

        def merge_preds(pred: Tensor):
            # Predictions on the output will be the mean of the votes
            if self.combine == 'mean_logits':
                return item.reshape(data.num_votes, -1, item.size(-1)).mean(dim=0)
            elif self.combine == 'mean_probs':
                probs = torch.softmax(item, dim=-1)
                mean_probs = probs.reshape(data.num_votes, -1, item.size(-1)).mean(dim=0)
                return mean_probs.clamp_min(1e-8).log()
            elif self.combine == 'popularity':
                choices = one_hot(item.argmax(dim=-1), num_classes=item.size(-1), dtype=torch.float)
                popularities = choices.reshape(data.num_votes, -1, item.size(-1)).mean(dim=0)
                # TODO: argmax chooses the first in case of ties; maybe we can improve on this?
                return popularities.clamp_min(1e-8).log()
            else:
                raise ValueError(f"Unsupported combination type '{self.combine}'")

        # Output will only have one vote's worth of nodes
        out = type(data)()
        if isinstance(data, HeteroData):
            # Global features
            for key, item in data._global_store.items():
                if isinstance(item, Tensor) and item.dim() and item.size(0) == batch_size:
                    out[key] = item[:new_batch_size]
                else:
                    out[key] = item

            # Node features
            for store in data.node_stores:
                new_num_nodes = store.num_nodes // data.num_votes

                for key, item in store.items():
                    if key == 'pred':
                        out[store._key][key] = merge_preds(item)
                    elif '_index' in key or key == 'ptr':
                        # Drop stale bookkeeping fields
                        pass
                    elif isinstance(item, Tensor) and item.dim() and item.size(0) == store.num_nodes:
                        out[store._key][key] = item[:new_num_nodes]
                        if 'pos' not in data:
                            out[store._key].num_nodes = out[store._key][key].size(0)
                    elif isinstance(item, Tensor) and item.dim() and item.size(0) == batch_size:
                        out[store._key][key] = item[:new_batch_size]
                    else:
                        out[store._key][key] = item

            # Edge features
            for store in data.edge_stores:
                (_src, _to, dest) = store._key
                for key, item in store.items():
                    if key == 'edge_index':
                        assert item.size(0) == data[dest].num_nodes  # Only works on kNN-formatted edges
                        out[store._key][key] = item[: out[dest].num_nodes]

        else:
            num_nodes = data.num_nodes
            new_num_nodes = num_nodes // data.num_votes

            # Non-hetero case is simpler
            for key, item in data.items():
                if key == 'edge_index':
                    assert item.size(0) == num_nodes  # Only works on kNN-formatted edges
                    out[key] = item[:new_num_nodes]
                elif key == 'pred':
                    out[key] = merge_preds(item)
                elif '_index' in key:
                    # Drop stale index/cluster bookkeeping fields
                    pass
                elif isinstance(item, Tensor) and not item.dim():
                    out[key] = item
                elif data.is_edge_attr(key) or key in ('ptr',):
                    # Drop invalidated edge attributes
                    # TODO: handle this correctly
                    pass
                elif data.is_node_attr(key) and item.size(0) == num_nodes:
                    out[key] = item[:new_num_nodes]
                    if 'pos' not in data:
                        out.num_nodes = out[key].size(0)
                elif isinstance(item, Tensor) and item.dim() and item.size(0) == batch_size:
                    out[key] = item[:new_batch_size]
                else:
                    out[key] = item


        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.combine})"
