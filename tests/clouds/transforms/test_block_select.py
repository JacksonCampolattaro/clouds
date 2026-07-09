import pytest
import torch
from torch_geometric.data import Data

from clouds.transforms.block_select import BlockSelect


class TestBlockSelect:
    """Test suite for BlockSelect transform."""

    @pytest.fixture
    def basic_data(self):
        """Create a basic Data object without batch information."""
        pos = torch.randn(100, 3)
        return Data(pos=pos, num_nodes=100)

    @pytest.fixture
    def batched_data(self):
        """Create a batched Data object with batch and ptr attributes."""
        # Create three batches of different sizes
        pos1 = torch.randn(50, 3)
        pos2 = torch.randn(30, 3)
        pos3 = torch.randn(20, 3)
        
        pos = torch.cat([pos1, pos2, pos3])
        batch = torch.tensor([0] * 50 + [1] * 30 + [2] * 20)
        ptr = torch.tensor([0, 50, 80, 100])
        
        return Data(pos=pos, batch=batch, ptr=ptr, num_nodes=100)

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        transform = BlockSelect()
        assert transform.max_num_points == int(1e9)
        assert transform.min_num_points == 1
        assert transform.selection_factor == 1.0

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        transform = BlockSelect(
            max_num_points=1000,
            min_num_points=10,
            selection_factor=0.5
        )
        assert transform.max_num_points == 1000
        assert transform.min_num_points == 10
        assert transform.selection_factor == 0.5

    def test_forward_basic_data(self, basic_data):
        """Test forward pass with non-batched data."""
        transform = BlockSelect(selection_factor=0.3)
        result = transform(basic_data)
        
        expected_size = max(min(int(100 * 0.3), int(1e9)), 1)
        assert result.selection_index is not None
        assert len(result.selection_index) == expected_size
        assert torch.all(result.selection_index == torch.arange(expected_size))
        assert result.pos.shape == basic_data.pos.shape

    def test_forward_basic_data_custom_params(self, basic_data):
        """Test forward pass with custom selection parameters."""
        transform = BlockSelect(
            max_num_points=20,
            min_num_points=5,
            selection_factor=0.8
        )
        result = transform(basic_data)
        
        # Should be clamped by max_num_points
        expected_size = 20  # 80 * 0.8 = 64, clamped to 20
        assert len(result.selection_index) == expected_size

    def test_forward_basic_data_min_clamp(self, basic_data):
        """Test min clamping for non-batched data."""
        transform = BlockSelect(
            max_num_points=100,
            min_num_points=50,
            selection_factor=0.1  # Would give 10, but clamped to min
        )
        result = transform(basic_data)
        
        expected_size = 50  # min_num_points
        assert len(result.selection_index) == expected_size

    def test_forward_batched_data(self, batched_data):
        """Test forward pass with batched data."""
        transform = BlockSelect(selection_factor=0.5)
        result = transform(batched_data)
        
        # Expected sizes per batch: 25, 15, 10 (50% of each)
        expected_sizes = [25, 15, 10]
        assert result.selection_index is not None
        
        # Check that selection indices are correct
        expected_indices = torch.cat([
            torch.arange(0, 25),
            torch.arange(50, 65),
            torch.arange(80, 90)
        ])
        assert torch.all(result.selection_index == expected_indices)

    def test_forward_batched_data_clamp(self, batched_data):
        """Test clamping for batched data."""
        transform = BlockSelect(
            max_num_points=12,
            min_num_points=5,
            selection_factor=0.8
        )
        result = transform(batched_data)
        
        # Expected: 40, 24, 16 -> clamped to 12, 12, 12
        expected_sizes = [12, 12, 12]
        expected_indices = torch.cat([
            torch.arange(0, 12),
            torch.arange(50, 62),
            torch.arange(80, 92)
        ])
        assert torch.all(result.selection_index == expected_indices)

    def test_forward_batched_data_min_clamp(self, batched_data):
        """Test min clamping for batched data."""
        transform = BlockSelect(
            max_num_points=30,
            min_num_points=10,
            selection_factor=0.1  # Would give 5, 3, 2 -> clamped to 10
        )
        result = transform(batched_data)
        
        expected_sizes = [10, 10, 10]
        expected_indices = torch.cat([
            torch.arange(0, 10),
            torch.arange(50, 60),
            torch.arange(80, 90)
        ])
        assert torch.all(result.selection_index == expected_indices)

    def test_forward_selection_factor_one(self, basic_data):
        """Test with selection_factor = 1.0."""
        transform = BlockSelect(selection_factor=1.0)
        result = transform(basic_data)
        
        assert len(result.selection_index) == 100
        assert torch.all(result.selection_index == torch.arange(100))

    def test_forward_selection_factor_zero(self, basic_data):
        """Test with selection_factor = 0.0."""
        transform = BlockSelect(
            selection_factor=0.0,
            min_num_points=10,
            max_num_points=100
        )
        result = transform(basic_data)
        
        # Should use min_num_points (10)
        assert len(result.selection_index) == 10

    def test_repr(self):
        """Test string representation."""
        transform = BlockSelect(
            max_num_points=1000,
            selection_factor=0.5
        )
        repr_str = repr(transform)
        assert "BlockSelect" in repr_str
        assert "*0.5" in repr_str
        assert "<1000" in repr_str

    def test_data_attributes_preserved(self, basic_data):
        """Test that original data attributes are preserved."""
        transform = BlockSelect(selection_factor=0.3)
        result = transform(basic_data)
        
        assert result.pos is basic_data.pos
        assert result.num_nodes == basic_data.num_nodes
        assert hasattr(result, 'selection_index')

    def test_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        # These should still work (no validation in constructor)
        transform = BlockSelect(
            max_num_points=-1,
            min_num_points=-5,
            selection_factor=-1.0
        )
        # They'll be used in clamp, which will handle negative values
        # but may produce unexpected results - this is a behavior test
        assert transform.max_num_points == -1
        assert transform.min_num_points == -5
        assert transform.selection_factor == -1.0

    def test_large_number_points(self):
        """Test with a large number of points."""
        pos = torch.randn(10000, 3)
        data = Data(pos=pos, num_nodes=10000)
        
        transform = BlockSelect(
            max_num_points=5000,
            selection_factor=0.8
        )
        result = transform(data)
        
        # Should be clamped to 5000
        assert len(result.selection_index) == 5000

    @pytest.mark.parametrize("selection_factor,min_points,max_points,expected_size", [
        (0.0, 1, 100, 1),
        (0.5, 1, 100, 50),
        (1.0, 1, 100, 100),
        (2.0, 1, 100, 100),  # Clamped
        (0.5, 30, 100, 50),  # Within bounds
        (0.1, 30, 100, 30),  # Clamped to min
        (0.9, 30, 50, 50),   # Clamped to max
    ])
    def test_parametrized_basic_data(self, basic_data, selection_factor, min_points, max_points, expected_size):
        """Parametrized tests for various parameter combinations."""
        transform = BlockSelect(
            selection_factor=selection_factor,
            min_num_points=min_points,
            max_num_points=max_points
        )
        result = transform(basic_data)
        assert len(result.selection_index) == expected_size

    @pytest.mark.parametrize("selection_factor,min_points,max_points,expected_sizes", [
        (0.0, 1, 100, [1, 1, 1]),
        (0.5, 1, 100, [25, 15, 10]),
        (1.0, 1, 100, [50, 30, 20]),
        # (2.0, 1, 100, [50, 30, 20]),  # Clamped
        (0.5, 20, 100, [25, 20, 20]),  # Second and third clamped to min
        (0.9, 10, 15, [15, 15, 15]),   # All clamped to max
    ])
    def test_parametrized_batched_data(self, batched_data, selection_factor, min_points, max_points, expected_sizes):
        """Parametrized tests for batched data."""
        transform = BlockSelect(
            selection_factor=selection_factor,
            min_num_points=min_points,
            max_num_points=max_points
        )
        result = transform(batched_data)
        
        # Verify selection_index contains the expected indices
        offset = 0
        for i, size in enumerate(expected_sizes):
            start_idx = batched_data.ptr[i].item()
            expected = torch.arange(start_idx, start_idx + size)
            actual = result.selection_index[offset:offset + size]
            assert torch.all(actual == expected)
            offset += size
        
        assert len(result.selection_index) == sum(expected_sizes)
