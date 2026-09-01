from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Data

from clouds.transforms.cut_select import CutSelect


@pytest.fixture
def sample_data():
    """Create a sample Data object for testing."""
    pos = torch.randn(100, 3)
    data = Data(pos=pos, num_nodes=100)
    return data


@pytest.fixture
def sample_data_2d():
    """Create a sample Data object with 2D coordinates."""
    pos = torch.randn(100, 2)
    data = Data(pos=pos, num_nodes=100)
    return data


class TestCutSelect:
    """Test suite for CutSelect class."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        cut_select = CutSelect(max_num_points=10)
        
        assert cut_select.max_num_points == 10
        assert cut_select.max_ratio == 1.0
        assert cut_select.sort_by_distance is False
        assert cut_select.dims is None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        cut_select = CutSelect(
            max_num_points=20,
            max_ratio=0.5,
            sort_by_distance=True,
            dims=[0, 1]
        )
        
        assert cut_select.max_num_points == 20
        assert cut_select.max_ratio == 0.5
        assert cut_select.sort_by_distance is True
        assert cut_select.dims == [0, 1]

    def test_forward_basic(self, sample_data):
        """Test basic forward pass."""
        cut_select = CutSelect(max_num_points=10)
        result = cut_select.forward(sample_data)
        
        assert hasattr(result, 'selection_index')
        assert len(result.selection_index) == 10
        assert torch.all(result.selection_index >= 0)
        assert torch.all(result.selection_index < sample_data.num_nodes)

    def test_forward_with_ratio(self, sample_data):
        """Test forward pass with ratio limiting."""
        cut_select = CutSelect(max_num_points=100, max_ratio=0.3)
        result = cut_select.forward(sample_data)
        
        expected_num = int(100 * 0.3)
        assert len(result.selection_index) == expected_num

    def test_forward_ratio_and_max_points(self, sample_data):
        """Test forward pass with both ratio and max points."""
        cut_select = CutSelect(max_num_points=5, max_ratio=0.8)
        result = cut_select.forward(sample_data)
        
        # max_num_points should take precedence
        assert len(result.selection_index) == 5

    def test_forward_with_sort_by_distance(self, sample_data):
        """Test forward pass with sorting by distance."""
        cut_select = CutSelect(max_num_points=10, sort_by_distance=True)
        result = cut_select.forward(sample_data)
        
        assert hasattr(result, 'selection_index')
        # When sort_by_distance is True, indices should be sorted by dot product
        assert len(result.selection_index) == 10

    def test_forward_with_dims(self, sample_data):
        """Test forward pass with dimension selection."""
        cut_select = CutSelect(max_num_points=10, dims=[0, 1])
        result = cut_select.forward(sample_data)
        
        assert hasattr(result, 'selection_index')
        assert len(result.selection_index) == 10

    def test_forward_with_single_dim(self, sample_data_2d):
        """Test forward pass with single dimension."""
        cut_select = CutSelect(max_num_points=10, dims=[0])
        result = cut_select.forward(sample_data_2d)
        
        assert hasattr(result, 'selection_index')
        assert len(result.selection_index) == 10

    def test_forward_keeps_original_data(self, sample_data):
        """Test that forward pass doesn't modify original data."""
        original_pos = sample_data.pos.clone()
        
        cut_select = CutSelect(max_num_points=10)
        result = cut_select.forward(sample_data)
        
        # Original data should be unchanged
        assert torch.equal(sample_data.pos, original_pos)
        assert sample_data.num_nodes == 100
        
        # Result should have additional attribute
        assert hasattr(result, 'selection_index')
        assert result.pos is sample_data.pos

    def test_forward_invalid_batch(self):
        """Test that forward raises assertion error when batch is present."""
        pos = torch.randn(10, 3)
        data = Data(pos=pos, num_nodes=10, batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]))
        
        cut_select = CutSelect(max_num_points=5)
        with pytest.raises(AssertionError):
            cut_select.forward(data)

    def test_forward_invalid_pos_type(self):
        """Test that forward raises assertion error when pos is not Tensor."""
        data = Data(pos=[1, 2, 3], num_nodes=3)
        
        cut_select = CutSelect(max_num_points=5)
        with pytest.raises(AssertionError):
            cut_select.forward(data)

    def test_repr(self):
        """Test string representation."""
        cut_select = CutSelect(
            max_num_points=20,
            max_ratio=0.5,
            sort_by_distance=True,
            dims=[0, 1]
        )
        
        expected = "CutSelect(max_num_points=20, max_ratio=0.5, sort_by_distance=True, dims=[0, 1])"
        assert repr(cut_select) == expected

    def test_repr_default(self):
        """Test string representation with default values."""
        cut_select = CutSelect(max_num_points=10)
        
        expected = "CutSelect(max_num_points=10, max_ratio=1.0, sort_by_distance=False, dims=None)"
        assert repr(cut_select) == expected

    @patch('torch.randn')
    def test_forward_vector_randomness(self, mock_randn, sample_data):
        """Test that random vector generation works correctly."""
        mock_randn.return_value = torch.tensor([1.0, 0.0, 0.0])
        
        cut_select = CutSelect(max_num_points=10)
        result = cut_select.forward(sample_data)
        
        # With vector [1,0,0], selection should be based on x-coordinates
        mock_randn.assert_called_once_with([3], device=sample_data.pos.device)
        assert hasattr(result, 'selection_index')

    def test_forward_edge_case_zero_points(self):
        """Test forward with max_num_points=0."""
        pos = torch.randn(10, 3)
        data = Data(pos=pos, num_nodes=10)
        
        cut_select = CutSelect(max_num_points=0)
        result = cut_select.forward(data)
        
        assert hasattr(result, 'selection_index')
        assert len(result.selection_index) == 0

    def test_forward_edge_case_max_ratio_zero(self):
        """Test forward with max_ratio=0."""
        pos = torch.randn(10, 3)
        data = Data(pos=pos, num_nodes=10)
        
        cut_select = CutSelect(max_num_points=100, max_ratio=0)
        result = cut_select.forward(data)
        
        assert hasattr(result, 'selection_index')
        assert len(result.selection_index) == 0

    def test_forward_different_devices(self):
        """Test forward on CPU (CUDA test would need GPU availability)."""
        pos = torch.randn(20, 3)
        data = Data(pos=pos, num_nodes=20)
        
        cut_select = CutSelect(max_num_points=5)
        result = cut_select.forward(data)
        
        assert result.selection_index.device == pos.device

    def test_forward_selection_index_properties(self, sample_data):
        """Test properties of selection_index."""
        cut_select = CutSelect(max_num_points=10)
        result = cut_select.forward(sample_data)
        
        # Check that selection_index contains valid indices
        assert torch.all(result.selection_index >= 0)
        assert torch.all(result.selection_index < sample_data.num_nodes)
        assert result.selection_index.dtype == torch.long

    @pytest.mark.parametrize("max_num_points, max_ratio, expected_count", [
        (10, 1.0, 10),
        (15, 0.5, 15),  # max_num_points takes precedence
        (200, 0.3, 30),  # ratio takes precedence
        (200, 1.0, 100),  # all points
        (50, 0.2, 20),  # ratio takes precedence
    ])
    def test_forward_count_calculation(self, sample_data, max_num_points, max_ratio, expected_count):
        """Test various combinations of max_num_points and max_ratio."""
        cut_select = CutSelect(max_num_points=max_num_points, max_ratio=max_ratio)
        result = cut_select.forward(sample_data)
        assert len(result.selection_index) == expected_count

    def test_forward_consistency_with_same_seed(self, sample_data):
        """Test that results are reproducible with same random seed."""
        torch.manual_seed(42)
        
        cut_select1 = CutSelect(max_num_points=10)
        result1 = cut_select1.forward(sample_data)
        
        torch.manual_seed(42)
        cut_select2 = CutSelect(max_num_points=10)
        result2 = cut_select2.forward(sample_data)
        
        assert torch.equal(result1.selection_index, result2.selection_index)

    def test_forward_3d_coordinates(self):
        """Test with 3D coordinates."""
        pos = torch.randn(50, 3)
        data = Data(pos=pos, num_nodes=50)
        
        cut_select = CutSelect(max_num_points=10)
        result = cut_select.forward(data)
        
        assert len(result.selection_index) == 10
        
    def test_forward_high_dimensions(self):
        """Test with high-dimensional coordinates."""
        pos = torch.randn(50, 128)
        data = Data(pos=pos, num_nodes=50)
        
        cut_select = CutSelect(max_num_points=10)
        result = cut_select.forward(data)
        
        assert len(result.selection_index) == 10
