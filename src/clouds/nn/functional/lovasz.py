import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


def _lovasz_grad(sorted_foregrounds: Tensor) -> Tensor:
    """
    Compute gradient of the Lovasz extension w.r.t sorted errors

    See Alg. 1 in paper
    """
    target_total = sorted_foregrounds.sum()
    intersection = target_total - sorted_foregrounds.float().cumsum(0)
    union = target_total + (1 - sorted_foregrounds).float().cumsum(0)
    jaccard = 1.0 - (intersection / union)

    if jaccard.size(0) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]

    return jaccard


def _lovasz_softmax(predictions: Tensor, labels: Tensor, ignore_index: int | None = None) -> Tensor:
    """Multi-class Lovasz-Softmax loss
    Args:
        @param predictions: [N, C] Predicted class probabilities (between 0 and 1).
        @param labels: [N] Tensor, ground truth labels (between 0 and C - 1)
        @param ignore_index: ignored class label
    """

    # Remove ignored samples
    if ignore_index is not None:
        valid_samples = (labels != ignore_index).nonzero().flatten()
        assert valid_samples.size(0)
        labels = labels[valid_samples]
        predictions = predictions[valid_samples, :]

    losses = []
    for class_index in labels.unique():
        foreground = (labels == class_index).type_as(predictions)
        predicted_foreground = predictions[:, class_index]

        errors = (foreground - predicted_foreground).abs()

        # Sort foreground predictions in order of decreasing error
        sorted_errors, permutation = torch.sort(errors, 0, descending=True)
        sorted_foregrounds = foreground[permutation.data]

        # Loss is the sum of errors weighted by gradients
        losses.append(torch.dot(sorted_errors, _lovasz_grad(sorted_foregrounds)))

    return sum(losses) / len(losses)


class MulticlassLovaszLoss(_Loss):
    """
    Lovasz loss for multiclass segmentation tasks.

    Adapted from: https://github.com/Pointcept/Pointcept/blob/7b37078ae301288309d62c4c88401716b6bdbe0e/pointcept/models/losses/lovasz.py#L211

    Shape
        - **y_pred** - torch.Tensor of shape (N, C, H, W)
        - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

    Reference
    https://github.com/BloodAxe/pytorch-toolbelt
    """

    def __init__(self, ignore_index: int | None = None) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, y_pred, y_true):
        y_pred = y_pred.softmax(dim=1)
        return _lovasz_softmax(
            y_pred,
            y_true,
            ignore_index=self.ignore_index,
        )
