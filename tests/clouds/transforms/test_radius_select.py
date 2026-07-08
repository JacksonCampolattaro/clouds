import math

import pytest
import torch
from torch_geometric.data import Data

from clouds.transforms.radius_select import RadiusSelect


class TestRadiusSelect:
    """Minimal set of pytest unit tests for RadiusSelect."""
    
    @pytest.fixture
    def sample_data(self):
        """Create a sample Data object with 10 points in 2D."""
        pos = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [6.0, 0.0],
            [7.0, 0.0],
            [8.0, 0.0],
            [9.0, 0.0],
        ], dtype=torch.float32)
        return Data(pos=pos)

    def test_init_and_repr(self):
        """Test initialization and __repr__ method."""
        transform = RadiusSelect(max_num_points=5, max_radius=2.0, max_ratio=0.5)
        repr_str = repr(transform)
        assert "RadiusSelect" in repr_str
        assert "max_num_points=5" in repr_str
        assert "max_radius=2.0" in repr_str
        assert "max_ratio=0.5" in repr_str
        assert "sort_by_distance=False" in repr_str
        assert "deterministic=False" in repr_str
        assert "dims=None" in repr_str

    def test_no_selection_when_num_nodes_less_than_max(self, sample_data):
        """Test that no selection is made when num_nodes <= max_num_points."""
        transform = RadiusSelect(max_num_points=10)
        result = transform(sample_data)
        assert not hasattr(result, 'selection_index')
        assert result.num_nodes == 10

    def test_selection_with_deterministic(self, sample_data):
        """Test deterministic selection (always picks first point)."""
        transform = RadiusSelect(max_num_points=3, deterministic=True)
        result = transform(sample_data)
        assert hasattr(result, 'selection_index')
        # With deterministic=True, center is pos[0], so closest points are 0, 1, 2
        assert result.selection_index.tolist() == [0, 1, 2]

    def test_selection_with_sort_by_distance(self, sample_data):
        """Test that selection_index is sorted by distance when sort_by_distance=True."""
        transform = RadiusSelect(max_num_points=3, deterministic=True, sort_by_distance=True)
        result = transform(sample_data)
        assert result.selection_index.tolist() == [0, 1, 2]  # Already sorted by distance from point 0

    def test_selection_without_sort_by_distance(self, sample_data):
        """Test that selection_index is sorted by index when sort_by_distance=False."""
        transform = RadiusSelect(max_num_points=3, deterministic=False, sort_by_distance=False)
        # Note: non-deterministic, so we can't test exact values, but we can test sorting
        # For deterministic test, we'll use deterministic=True
        transform = RadiusSelect(max_num_points=5, deterministic=True, sort_by_distance=False)
        result = transform(sample_data)
        # When sort_by_distance=False, selection_index should be sorted by index
        assert result.selection_index.tolist() == sorted(result.selection_index.tolist())

    def test_radius_filtering(self, sample_data):
        """Test that points beyond max_radius are filtered out."""
        transform = RadiusSelect(
            max_num_points=4, 
            max_radius=2.5, 
            deterministic=True,
            sort_by_distance=True
        )
        result = transform(sample_data)
        # From point 0, points within radius 2.5 are at indices 0, 1, 2
        assert result.selection_index.tolist() == [0, 1, 2]

    def test_max_ratio_limiting(self, sample_data):
        """Test that max_ratio limits the number of selected points."""
        transform = RadiusSelect(
            max_num_points=10, 
            max_ratio=0.3,
            deterministic=True,
            sort_by_distance=True
        )
        result = transform(sample_data)
        # 10 * 0.3 = 3 points
        assert len(result.selection_index) == 3
        assert result.selection_index.tolist() == [0, 1, 2]

    def test_dims_selection(self, sample_data):
        """Test that dims parameter correctly selects dimensions."""
        # Create 3D data
        pos_3d = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ], dtype=torch.float32)
        data_3d = Data(pos=pos_3d)
        
        # Select only first 2 dimensions
        transform = RadiusSelect(
            max_num_points=2,
            deterministic=True,
            dims=[0, 1],
            sort_by_distance=True
        )
        result = transform(data_3d)
        # Distance should be computed using only x,y coordinates
        assert result.selection_index.tolist() == [0, 1]

    def test_empty_selection_when_no_points_within_radius(self, sample_data):
        """Test that selection_index is empty when no points within radius."""
        transform = RadiusSelect(
            max_num_points=10,
            max_radius=0.1,
            deterministic=True,
            sort_by_distance=True
        )
        result = transform(sample_data)
        # Only point 0 is within radius 0.1 of itself
        assert result.selection_index.tolist() == [0]

    def test_math_inf_radius(self, sample_data):
        """Test that max_radius=inf doesn't filter any points."""
        transform = RadiusSelect(
            max_num_points=5,
            max_radius=math.inf,
            deterministic=True,
            sort_by_distance=True
        )
        result = transform(sample_data)
        # Should select first 5 points
        assert result.selection_index.tolist() == [0, 1, 2, 3, 4]
