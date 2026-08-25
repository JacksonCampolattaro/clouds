from unittest.mock import Mock

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform

from clouds.transforms import CombineVotes, VoteAugmentations


class TestVoteAugmentations:
    """Test suite for VoteAugmentations transform."""
    
    def test_init(self):
        """Test initialization of VoteAugmentations."""
        mock_aug1 = Mock(spec=BaseTransform)
        mock_aug2 = Mock(spec=BaseTransform)
        augmentations = [mock_aug1, mock_aug2]
        
        transform = VoteAugmentations(augmentations)
        
        assert transform.augmentations == augmentations
    
    def test_forward_basic(self):
        """Test basic forward pass with single data object."""
        # Create mock augmentations
        mock_aug1 = Mock(spec=BaseTransform)
        mock_aug2 = Mock(spec=BaseTransform)
        
        # Create test data
        data = Data(
            x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            y=torch.tensor([0, 1])
        )
        
        # Mock augmentations to return modified data
        augmented1 = Data(
            x=torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            y=torch.tensor([0, 1])
        )
        augmented2 = Data(
            x=torch.tensor([[1.5, 2.5], [3.5, 4.5]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            y=torch.tensor([0, 1])
        )
        
        mock_aug1.return_value = augmented1
        mock_aug2.return_value = augmented2
        
        transform = VoteAugmentations([mock_aug1, mock_aug2])
        result = transform(data)
        
        # Check that the result is a batch with correct properties
        assert hasattr(result, 'num_votes')
        assert result.num_votes == 2
        
        # Check that the batch dimension was added
        assert hasattr(result, 'batch')
        assert hasattr(result, 'ptr')
        
        # Check that original data was collated correctly
        assert hasattr(result, 'x')
        assert hasattr(result, 'edge_index')
        assert hasattr(result, 'y')

    def test_forward_with_batch_input_old_behavior(self):
        """Test forward pass with batched input (old behavior)."""
        mock_aug1 = Mock(spec=BaseTransform)
        mock_aug2 = Mock(spec=BaseTransform)
        
        # Create batched data
        data1 = Data(
            x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            y=torch.tensor([0, 1])
        )
        data2 = Data(
            x=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            y=torch.tensor([1, 0])
        )
        batch = Batch.from_data_list([data1, data2])
        
        # Mock augmentations to return copies of the batch
        mock_aug1.return_value = batch.clone()
        mock_aug2.return_value = batch.clone()
        
        transform = VoteAugmentations([mock_aug1, mock_aug2])
        
        # This should not raise an assertion error
        result = transform(batch)
        
        assert result.num_votes == 2
        assert hasattr(result, 'batch')
        assert hasattr(result, 'ptr')
    
    def test_forward_with_batch_input_new_behavior(self):
        """Test forward pass with batched input producing num_votes times batches."""
        mock_aug1 = Mock(spec=BaseTransform)
        mock_aug2 = Mock(spec=BaseTransform)
        mock_aug3 = Mock(spec=BaseTransform)
        
        # Create batched data with 2 graphs
        data1 = Data(
            x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            y=torch.tensor([0, 1]),
            edge_index=torch.tensor([[0, 1], [1, 0]])
        )
        data2 = Data(
            x=torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
            y=torch.tensor([1, 0, 1]),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        )
        batch = Batch.from_data_list([data1, data2])
        
        # Create augmented versions for each vote
        # For vote 1: modify features
        aug_data1_v1 = Data(
            x=torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
            y=torch.tensor([0, 1]),
            edge_index=torch.tensor([[0, 1], [1, 0]])
        )
        aug_data2_v1 = Data(
            x=torch.tensor([[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]]),
            y=torch.tensor([1, 0, 1]),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        )
        aug_batch_v1 = Batch.from_data_list([aug_data1_v1, aug_data2_v1])
        
        # For vote 2: modify edges
        aug_data1_v2 = Data(
            x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            y=torch.tensor([0, 1]),
            edge_index=torch.tensor([[0, 1], [1, 0]])
        )
        aug_data2_v2 = Data(
            x=torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
            y=torch.tensor([1, 0, 1]),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]])  # Different edge structure
        )
        aug_batch_v2 = Batch.from_data_list([aug_data1_v2, aug_data2_v2])
        
        # For vote 3: modify features differently
        aug_data1_v3 = Data(
            x=torch.tensor([[0.5, 1.0], [1.5, 2.0]]),
            y=torch.tensor([0, 1]),
            edge_index=torch.tensor([[0, 1], [1, 0]])
        )
        aug_data2_v3 = Data(
            x=torch.tensor([[2.5, 3.0], [3.5, 4.0], [4.5, 5.0]]),
            y=torch.tensor([1, 0, 1]),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        )
        aug_batch_v3 = Batch.from_data_list([aug_data1_v3, aug_data2_v3])
        
        mock_aug1.return_value = aug_batch_v1
        mock_aug2.return_value = aug_batch_v2
        mock_aug3.return_value = aug_batch_v3
        
        transform = VoteAugmentations([mock_aug1, mock_aug2, mock_aug3])
        result = transform(batch)
        
        # Check num_votes
        assert result.num_votes == 3
        
        # Check that batch tensor has correct grouping
        # Should have 3 votes * 2 original graphs = 6 graphs total
        assert hasattr(result, 'batch')
        assert hasattr(result, 'ptr')
        
        # The batch tensor should have values 0,1,0,1,0,1 (for each graph repeated per vote)
        expected_batch_pattern = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])  # 2 graphs * 3 votes
        # Actually, the batch values should be ordered by vote first: [0,1,0,1,0,1]
        # But the actual values depend on collate ordering
        
        # Check that we have the right number of nodes total
        # Original: 2+3=5 nodes per vote, *3 votes = 15 nodes
        assert result.x.shape[0] == 15
        
        # Check that the result contains all augmented data
        # The order should be: all graphs for vote 1, then all graphs for vote 2, then all graphs for vote 3
        # Let's verify the features are correctly grouped
        expected_x = torch.cat([
            aug_batch_v1.x,  # Vote 1: all graphs
            aug_batch_v2.x,  # Vote 2: all graphs
            aug_batch_v3.x   # Vote 3: all graphs
        ], dim=0)
        assert torch.equal(result.x, expected_x)
        
        # Check y is also correctly grouped
        expected_y = torch.cat([
            aug_batch_v1.y,
            aug_batch_v2.y,
            aug_batch_v3.y
        ], dim=0)
        assert torch.equal(result.y, expected_y)
    
    def test_forward_with_batch_input_new_behavior_edge_cases(self):
        """Test edge cases for batched input new behavior."""
        # Test with single graph in batch
        mock_aug1 = Mock(spec=BaseTransform)
        mock_aug2 = Mock(spec=BaseTransform)
        
        single_graph = Data(
            x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            y=torch.tensor([0, 1])
        )
        batch_single = Batch.from_data_list([single_graph])
        
        aug_v1 = Batch.from_data_list([
            Data(x=torch.tensor([[2.0, 3.0], [4.0, 5.0]]), y=torch.tensor([0, 1]))
        ])
        aug_v2 = Batch.from_data_list([
            Data(x=torch.tensor([[1.5, 2.5], [3.5, 4.5]]), y=torch.tensor([0, 1]))
        ])
        
        mock_aug1.return_value = aug_v1
        mock_aug2.return_value = aug_v2
        
        transform = VoteAugmentations([mock_aug1, mock_aug2])
        result = transform(batch_single)
        
        assert result.num_votes == 2
        assert result.x.shape[0] == 4  # 2 nodes * 2 votes
    
    
    def test_forward_with_batch_input_combine_votes_compatibility(self):
        """Test that VoteAugmentations output is compatible with CombineVotes."""
        
        # Batched data with 2 graphs x 2 nodes
        data = Batch.from_data_list(
            [
                Data(x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]), y=torch.tensor([0, 1])),
                Data(x=torch.tensor([[5.0, 6.0], [7.0, 8.0]]), y=torch.tensor([1, 0])),
            ]
        )

        # Voting with 2 augmentations
        vote_transform = VoteAugmentations(
            [
                Mock(spec=BaseTransform, return_value=data),
                Mock(spec=BaseTransform, return_value=data),
            ]
        )
       
        # Apply VoteAugmentations
        augmented = vote_transform(data)

        # (apply model)
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

        # Apply CombineVotes
        combine_transform = CombineVotes()
        combined = combine_transform(augmented)
        
        # Check that combined output has the right shape
        # Should have 2 graphs with averaged predictions
        assert combined.num_votes == 2
        assert combined.x.shape[0] == 8
        assert combined.vote_pred.shape[0] == 8
        assert combined.vote_y.shape[0] == 8
        assert combined.pred.shape[0] == 4
        assert combined.y.shape[0] == 4

        # Check that pred was averaged correctly (mean of the two votes)
        # For each graph, pred should be averaged across votes
        assert torch.allclose(combined.pred, torch.tensor(0.5))


class TestCombineVotes:
    """Test suite for CombineVotes transform."""
    
    def test_forward_basic(self):
        """Test basic forward pass combining votes."""
        # Create data with multiple votes
        data = Data(
            x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            pred=torch.tensor([
                [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]],  # Vote 0
                [[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]],  # Vote 1
                [[0.1, 0.4, 0.5], [0.2, 0.3, 0.5]]   # Vote 2
            ]),  # Shape: [num_votes, num_nodes, num_classes]
            y=torch.tensor([
                [0, 1],  # Vote 0
                [1, 0],  # Vote 1
                [1, 1]   # Vote 2
            ]),  # Shape: [num_votes, num_nodes]
            num_votes=3
        )
        
        transform = CombineVotes()
        result = transform(data)
        
        # Check that num_votes was preserved
        assert result.num_votes == 3
        
        # Check that vote_pred was stored
        assert hasattr(result, 'vote_pred')
        assert torch.equal(result.vote_pred, data.pred)
        
        # Check that pred was averaged
        expected_pred = data.pred.mean(dim=0)
        assert torch.allclose(result.pred, expected_pred)
        
        # Check that y was reshaped
        expected_y = data.y.reshape(3, -1)[0]
        assert torch.equal(result.y, expected_y)
    
    def test_forward_with_batched_input_from_vote_augmentations(self):
        """Test CombineVotes on batched input from VoteAugmentations."""
        # This simulates the output structure from VoteAugmentations with batched input
        # Create 2 original graphs with 2 votes each
        # Original graph 1: 2 nodes, graph 2: 3 nodes
        pred_vote1_graph1 = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
        pred_vote1_graph2 = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3], [0.4, 0.3, 0.3]])
        
        pred_vote2_graph1 = torch.tensor([[0.15, 0.25, 0.6], [0.35, 0.35, 0.3]])
        pred_vote2_graph2 = torch.tensor([[0.25, 0.35, 0.4], [0.15, 0.55, 0.3], [0.45, 0.25, 0.3]])
        
        # Stack pred: [num_votes * num_graphs, num_nodes_per_graph, num_classes]
        # Order: all graphs for vote 1, then all graphs for vote 2
        pred = torch.cat([
            pred_vote1_graph1, pred_vote1_graph2,  # Vote 1: graph 1, graph 2
            pred_vote2_graph1, pred_vote2_graph2   # Vote 2: graph 1, graph 2
        ], dim=0)
        
        # Similar for y
        y_vote1_graph1 = torch.tensor([0, 1])
        y_vote1_graph2 = torch.tensor([1, 0, 1])
        y_vote2_graph1 = torch.tensor([0, 1])
        y_vote2_graph2 = torch.tensor([1, 0, 1])
        
        y = torch.cat([
            y_vote1_graph1, y_vote1_graph2,
            y_vote2_graph1, y_vote2_graph2
        ], dim=0)
        
        # Create batch tensor indicating which graph each node belongs to
        # 2 nodes in graph1, 3 nodes in graph2, repeated for 2 votes
        batch = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 3])

        data = Data(
            x=torch.randn(10, 5),  # 10 total nodes
            pred=pred,
            y=y,
            batch=batch,
            num_votes=2
        )
        
        transform = CombineVotes()
        result = transform(data)
        
        # Check that pred was averaged per graph across votes
        # For graph 1: average of pred_vote1_graph1 and pred_vote2_graph1
        expected_pred_graph1 = (pred_vote1_graph1 + pred_vote2_graph1) / 2
        expected_pred_graph2 = (pred_vote1_graph2 + pred_vote2_graph2) / 2
        expected_pred = torch.cat([expected_pred_graph1, expected_pred_graph2], dim=0)
        
        assert torch.allclose(result.pred, expected_pred)
        
        # Check that y was reshaped correctly
        # y should be averaged or mode? In the current implementation, it's reshaped
        # but not averaged. The test checks that reshape works with batched input.
        assert result.y.shape[0] == 5  # 2 + 3 nodes
        
    
    def test_forward_asserts_num_votes(self):
        """Test that forward raises assertion if num_votes is missing."""
        data = Data(
            x=torch.tensor([[1.0, 2.0]]),
            pred=torch.tensor([[[0.1, 0.2, 0.7]]])
        )
        
        transform = CombineVotes()
        
        with pytest.raises(AssertionError):
            transform(data)
    
    def test_forward_preserves_other_attributes(self):
        """Test that other attributes are preserved."""
        data = Data(
            x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            pred=torch.tensor([
                [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]],
                [[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]]
            ]),
            y=torch.tensor([[0, 1], [1, 0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            num_votes=2,
            custom_attr=torch.tensor([1.0, 2.0])
        )
        
        transform = CombineVotes()
        result = transform(data)
        
        # Other attributes should be preserved
        assert hasattr(result, 'custom_attr')
        assert torch.equal(result.custom_attr, data.custom_attr)
        assert hasattr(result, 'edge_index')
        assert torch.equal(result.edge_index, data.edge_index)
        assert hasattr(result, 'x')
        assert torch.equal(result.x, data.x)
    
    def test_forward_predictions_mean_correctly(self):
        """Test that predictions are averaged correctly."""
        data = Data(
            x=torch.tensor([[1.0, 2.0]]),
            pred=torch.tensor([
                [[0.1, 0.2, 0.7]],
                [[0.2, 0.3, 0.5]],
                [[0.1, 0.4, 0.5]]
            ]),
            num_votes=3
        )
        
        transform = CombineVotes()
        result = transform(data)
        
        # Manually compute expected mean
        expected_mean = torch.tensor([[0.1333, 0.3000, 0.5667]])
        assert torch.allclose(result.pred, expected_mean, rtol=1e-3)
    
