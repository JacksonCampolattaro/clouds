from unittest.mock import Mock

import pytest
import torch
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.transforms import BaseTransform

from clouds.transforms import CombineVotes, Identity, VoteAugmentations


def test_vote_augmentations_init():
    augmentations = [Mock(spec=BaseTransform), Mock(spec=BaseTransform)]

    assert VoteAugmentations(augmentations).augmentations == augmentations


def test_vote_augmentations_forward_basic():
    mock_aug1 = Mock(spec=BaseTransform)
    mock_aug2 = Mock(spec=BaseTransform)
    mock_aug1.return_value = Data(
        x=torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=torch.tensor([0, 1]),
    )
    mock_aug2.return_value = Data(
        x=torch.tensor([[1.5, 2.5], [3.5, 4.5]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=torch.tensor([0, 1]),
    )

    result = VoteAugmentations([mock_aug1, mock_aug2])(
        Data(x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]), edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.tensor([0, 1]))
    )

    assert result.num_votes == 2
    assert hasattr(result, 'batch')
    assert hasattr(result, 'ptr')


def test_vote_augmentations_forward_with_batch_input_combine_votes_compatibility():
    data = Batch.from_data_list(
        [
            Data(x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]), y=torch.tensor([0, 1]), category=torch.tensor(0)),
            Data(x=torch.tensor([[5.0, 6.0], [7.0, 8.0]]), y=torch.tensor([1, 0]), category=torch.tensor(1)),
        ]
    )

    augmented = VoteAugmentations([Identity(), Identity()])(data)
    augmented.pred = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    combined = CombineVotes(combine='mean_logits')(augmented)

    assert combined.num_votes == 2
    assert combined.x.shape[0] == 4
    assert combined.pred.shape[0] == 4
    assert combined.y.shape[0] == 4
    assert combined.category.shape[0] == 2
    assert torch.allclose(combined.pred, torch.tensor(0.5))
    assert torch.allclose(combined.category, data.category)


def test_combine_votes_forward_basic():
    data = Data(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0], [3.0, 4.0]]),
        pred=torch.tensor(
            [
                [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]],
                [[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]],
                [[0.1, 0.4, 0.5], [0.2, 0.3, 0.5]],
            ]
        ),
        y=torch.tensor([0, 1, 1]),
        category=torch.tensor([1, 1, 1]),
        num_votes=3,
    )

    result = CombineVotes()(data)

    assert result.num_votes == 3
    assert torch.allclose(result.pred, data.pred.mean(dim=0))
    assert torch.equal(result.y, data.y.reshape(3, -1)[0])
    assert torch.equal(result.category, data.category.reshape(3, -1)[0])


def test_combine_votes_forward_with_batched_input_from_vote_augmentations():
    pred_vote1_graph1 = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    pred_vote1_graph2 = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3], [0.4, 0.3, 0.3]])
    pred_vote2_graph1 = torch.tensor([[0.15, 0.25, 0.6], [0.35, 0.35, 0.3]])
    pred_vote2_graph2 = torch.tensor([[0.25, 0.35, 0.4], [0.15, 0.55, 0.3], [0.45, 0.25, 0.3]])

    data = Data(
        x=torch.randn(10, 5),
        pred=torch.cat([pred_vote1_graph1, pred_vote1_graph2, pred_vote2_graph1, pred_vote2_graph2], dim=0),
        y=torch.tensor([0, 1, 1, 0, 1, 0, 1, 1, 0, 1]),
        batch=torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 3]),
        num_votes=2,
    )

    result = CombineVotes()(data)

    assert torch.allclose(
        result.pred,
        torch.cat([(pred_vote1_graph1 + pred_vote2_graph1) / 2, (pred_vote1_graph2 + pred_vote2_graph2) / 2], dim=0),
    )
    assert result.y.shape[0] == 5


def test_combine_votes_forward_asserts_num_votes():
    data = Data(x=torch.tensor([[1.0, 2.0]]), pred=torch.tensor([[[0.1, 0.2, 0.7]]]))

    with pytest.raises(AssertionError):
        CombineVotes()(data)


def test_combine_votes_forward_predictions_mean_correctly():
    data = Data(
        x=torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]),
        pred=torch.tensor([[[0.1, 0.2, 0.7]], [[0.2, 0.3, 0.5]], [[0.1, 0.4, 0.5]]]),
        num_votes=3,
    )

    result = CombineVotes(combine='mean_logits')(data)

    assert torch.allclose(result.pred, torch.tensor([[0.1333, 0.3000, 0.5667]]), rtol=1e-3)


def test_combine_votes_forward_hetero_basic():
    data = HeteroData()
    data.num_votes = 3

    data['paper'].x = torch.randn(2 * 3, 4)
    data['paper'].pred = torch.tensor(
        [
            [0.1, 0.2, 0.7],
            [0.3, 0.4, 0.3],
            [0.2, 0.3, 0.5],
            [0.1, 0.6, 0.3],
            [0.1, 0.4, 0.5],
            [0.2, 0.3, 0.5],
        ]
    )
    data['paper'].y = torch.tensor([0, 1, 0, 1, 0, 1])

    data['author'].x = torch.randn(3 * 3, 4)
    data['author'].pred = torch.tensor(
        [
            [0.4, 0.3, 0.3],
            [0.2, 0.5, 0.3],
            [0.1, 0.1, 0.8],
            [0.3, 0.4, 0.3],
            [0.3, 0.4, 0.3],
            [0.2, 0.2, 0.6],
            [0.5, 0.2, 0.3],
            [0.1, 0.6, 0.3],
            [0.3, 0.3, 0.4],
        ]
    )
    data['author'].y = torch.tensor([1, 0, 1, 1, 0, 1, 1, 0, 1])

    data['author', 'of', 'paper'].edge_index = torch.zeros([2 * 3, 4])

    result = CombineVotes()(data)

    assert torch.allclose(
        result['paper'].pred,
        data['paper'].pred.reshape(3, -1, data['paper'].pred.size(-1)).mean(dim=0),
    )
    assert torch.equal(result['paper'].y, data['paper'].y.reshape(3, -1)[0])
    assert torch.allclose(
        result['author'].pred,
        data['author'].pred.reshape(3, -1, data['author'].pred.size(-1)).mean(dim=0),
    )
    assert torch.equal(result['author'].y, data['author'].y.reshape(3, -1)[0])


def test_combine_votes_forward_hetero_batched_node_stores():
    data = HeteroData()
    data.num_votes = 2

    pred_vote1_g1 = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    pred_vote1_g2 = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3], [0.4, 0.3, 0.3]])
    pred_vote2_g1 = torch.tensor([[0.15, 0.25, 0.6], [0.35, 0.35, 0.3]])
    pred_vote2_g2 = torch.tensor([[0.25, 0.35, 0.4], [0.15, 0.55, 0.3], [0.45, 0.25, 0.3]])

    data['paper'].pred = torch.cat([pred_vote1_g1, pred_vote1_g2, pred_vote2_g1, pred_vote2_g2], dim=0)
    data['paper'].batch = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 3])
    data['paper'].x = torch.randn(10, 5)

    result = CombineVotes()(data)

    assert torch.allclose(
        result['paper'].pred,
        torch.cat([(pred_vote1_g1 + pred_vote2_g1) / 2, (pred_vote1_g2 + pred_vote2_g2) / 2], dim=0),
    )
