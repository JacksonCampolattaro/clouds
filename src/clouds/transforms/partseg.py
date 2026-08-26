import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import one_hot

from clouds.transforms.knn import knn


def class_map_to_table(class_map: dict[str, list]) -> Tensor:
    num_categories = len(class_map)
    num_classes = max(max(classes) for classes in class_map.values()) + 1
    categories_to_classes = torch.zeros([num_categories, num_classes], dtype=torch.bool)
    for category, classes in enumerate(class_map.values()):
        categories_to_classes[category, classes] = True
    return categories_to_classes


class CategoryClassMask(BaseTransform):
    def __init__(
        self,
        categories_to_classes: Tensor | dict[str, list],
    ):
        super().__init__()
        if not isinstance(categories_to_classes, Tensor):
            categories_to_classes = class_map_to_table(categories_to_classes)
        self.categories_to_classes = categories_to_classes

    def forward(self, data: Data) -> Data:
        self.categories_to_classes = self.categories_to_classes.to(device=data.category.device)
        batch_classes = self.categories_to_classes[data.category, :]
        batch_masks = ~batch_classes
        for store in data.node_stores:
            if hasattr(store, 'pred'):
                point_masks = batch_masks[store.batch, :]
                # store.pred[point_masks] = -torch.inf  # -1e7
                # store.pred[point_masks] -= 10
                store.pred = torch.where(point_masks, -torch.inf, store.pred)

        return data


class RefinePartSegmentation(BaseTransform):
    """
    A simplified version of part_seg_refinement, as done in DeLA and PointNeXt
    https://github.com/Matrix-ASC/DeLA/blob/main/ShapeNetPart/putil.py#L62
    """

    def __init__(
        self,
        categories_to_classes: Tensor | dict[str, list],
        k: int = 10,
        replace_with_neighbors: bool = True,
    ):
        super().__init__()
        self.k = k
        self.replace_with_neighbors = replace_with_neighbors
        if not isinstance(categories_to_classes, Tensor):
            categories_to_classes = class_map_to_table(categories_to_classes)
        self.categories_to_classes = categories_to_classes
        # self.register_buffer('categories_to_classes', categories_to_classes) # Valid for later PyG

    def forward(self, data: Data) -> Data:
        # FIXME: disable AMP!
        self.categories_to_classes = self.categories_to_classes.to(device=data.category.device)
        for store in data.node_stores:
            if hasattr(store, 'pred') and hasattr(store, 'y'):
                preds = store.pred.argmax(dim=-1)

                # Count predictions for each batch
                pred_counts = global_add_pool(one_hot(preds, num_classes=store.pred.size(-1)), batch=store.batch)

                # Determine which predictions should be replaced
                rare_classes = pred_counts < self.k
                irrelevant_classes = ~self.categories_to_classes[data.category, :]
                bad_classes = (irrelevant_classes | rare_classes)[store.batch, :]

                if self.replace_with_neighbors:
                    bad_preds = bad_classes.gather(1, preds.unsqueeze(1)).flatten()
                    if not bad_preds.any():
                        continue

                    # Find neighbors of any nodes with bad predictions
                    neighbors = knn(
                        pos=store.pos,
                        batch=store.batch,
                        query_pos=store.pos[bad_preds],
                        query_batch=store.batch[bad_preds],
                        k=self.k + 1,
                    )

                    # Determine the most popular of the neighbor labels
                    store.pred[bad_preds] = one_hot(preds, num_classes=store.pred.size(-1))[neighbors[:, 1:]].mean(dim=1)
                    # store.pred[bad_preds] = store.pred[neighbors[:, 1:]].mean(dim=1)

                # Overwrite values
                store.pred = torch.where(bad_classes, -torch.inf, store.pred)

        return data
