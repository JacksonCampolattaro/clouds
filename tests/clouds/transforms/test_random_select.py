import pytest
import torch
from torch_geometric.data import Data

from clouds.transforms.random_select import RandomSelect


class TestRandomSelect:
    @pytest.fixture
    def simple_data(self):
        """Create a simple data object with 10 nodes."""
        pos = torch.randn(10, 3)
        x = torch.randn(10, 5)
        return Data(pos=pos, x=x, num_nodes=10)

    @pytest.fixture
    def batched_data(self):
        """Create a batched data object with 3 graphs."""
        pos = torch.randn(15, 3)
        x = torch.randn(15, 5)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        ptr = torch.tensor([0, 5, 10, 15])
        return Data(pos=pos, x=x, batch=batch, ptr=ptr, num_nodes=15)

    @pytest.fixture
    def batched_data_uneven(self):
        """Create a batched data object with uneven graph sizes."""
        pos = torch.randn(12, 3)
        x = torch.randn(12, 5)
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        ptr = torch.tensor([0, 4, 7, 12])
        return Data(pos=pos, x=x, batch=batch, ptr=ptr, num_nodes=12)

    def test_init_default(self):
        """Test default initialization."""
        transform = RandomSelect()
        assert transform.max_num_points == 1e7
        assert transform.min_num_points == 1
        assert transform.selection_factor == 1.0
        assert transform.replacement is False

    def test_init_custom(self):
        """Test custom initialization."""
        transform = RandomSelect(max_num_points=100, min_num_points=5, selection_factor=0.5, replacement=True)
        assert transform.max_num_points == 100
        assert transform.min_num_points == 5
        assert transform.selection_factor == 0.5
        assert transform.replacement is True

    def test_init_tuple_selection_factor(self):
        """Test initialization with tuple selection factor."""
        transform = RandomSelect(selection_factor=(0.2, 0.8))
        assert isinstance(transform.selection_factor, tuple)
        assert len(transform.selection_factor) == 2

    def test_forward_constant_factor(self, simple_data):
        """Test forward pass with constant selection factor."""
        transform = RandomSelect(selection_factor=0.5, max_num_points=100, min_num_points=1)
        result = transform(simple_data)

        # Should select exactly 5 nodes (10 * 0.5 = 5)
        assert hasattr(result, 'selection_index')
        assert len(result.selection_index) == 5
        assert result.selection_index.dtype == torch.long
        assert torch.all(result.selection_index >= 0)
        assert torch.all(result.selection_index < simple_data.num_nodes)
        assert torch.all(result.selection_index == result.selection_index.sort()[0])  # Check sorted

    def test_forward_min_num_points(self, simple_data):
        """Test forward pass with min_num_points constraint."""
        transform = RandomSelect(selection_factor=0.1, min_num_points=3, max_num_points=100)
        result = transform(simple_data)

        # Should select at least 3 nodes (min_num_points)
        assert len(result.selection_index) >= 3
        assert len(result.selection_index) <= simple_data.num_nodes

    def test_forward_max_num_points(self, simple_data):
        """Test forward pass with max_num_points constraint."""
        transform = RandomSelect(selection_factor=2.0, max_num_points=8, min_num_points=1)
        result = transform(simple_data)

        # Should select at most 8 nodes (max_num_points)
        assert len(result.selection_index) <= 8
        assert len(result.selection_index) <= simple_data.num_nodes

    def test_forward_with_replacement_single_graph(self, simple_data):
        """Test forward pass with replacement enabled for single graph."""
        transform = RandomSelect(selection_factor=1.5, replacement=True, max_num_points=100)
        result = transform(simple_data)

        # With replacement, we can have more than num_nodes
        expected_size = min(int(10 * 1.5), 100)
        assert len(result.selection_index) == expected_size
        assert result.selection_index.dtype == torch.long
        assert torch.all(result.selection_index >= 0)
        assert torch.all(result.selection_index < simple_data.num_nodes)

        # With replacement, duplicates should be possible
        # We can check for duplicates (not guaranteed but likely with size > num_nodes)
        if len(result.selection_index) > simple_data.num_nodes:
            unique_count = len(torch.unique(result.selection_index))
            assert unique_count < len(result.selection_index)  # Should have duplicates

    def test_forward_with_replacement_batched(self, batched_data):
        """Test forward pass with replacement enabled for batched data."""
        transform = RandomSelect(selection_factor=2, replacement=True, max_num_points=100)
        result = transform(batched_data)

        total_expected = sum(min((5 * 2), 100) for _ in range(3))
        assert len(result.selection_index) == total_expected
        assert result.selection_index.dtype == torch.long
        assert torch.all(result.selection_index >= 0)
        assert torch.all(result.selection_index < batched_data.num_nodes)

        # Check that indices are within each graph's range
        for start, end in zip([0, 5, 10], [5, 10, 15], strict=False):
            mask = (result.selection_index >= start) & (result.selection_index < end)
            count = mask.sum().item()
            expected_per_graph = min((5 * 2), 100)
            assert count == expected_per_graph

    def test_forward_with_replacement_batched_uneven(self, batched_data_uneven):
        """Test replacement with uneven graph sizes."""
        transform = RandomSelect(selection_factor=0.8, replacement=True, max_num_points=100)
        result = transform(batched_data_uneven)

        # Graph sizes: 4, 3, 5
        expected_counts = [
            min(int(4 * 0.8), 100),  # 3
            min(int(3 * 0.8), 100),  # 2
            min(int(5 * 0.8), 100),  # 4
        ]

        for i, (start, end) in enumerate(zip([0, 4, 7], [4, 7, 12])):
            mask = (result.selection_index >= start) & (result.selection_index < end)
            count = mask.sum().item()
            assert count == expected_counts[i]

    def test_forward_with_replacement_and_max_limit(self, simple_data):
        """Test forward pass with replacement and max limit."""
        transform = RandomSelect(selection_factor=5.0, replacement=True, max_num_points=20)
        result = transform(simple_data)

        # Should be limited by max_num_points=20
        assert len(result.selection_index) == 20

    def test_forward_batched_data_without_replacement(self, batched_data):
        """Test forward pass with batched data without replacement."""
        transform = RandomSelect(selection_factor=0.6, max_num_points=100, min_num_points=1)
        result = transform(batched_data)

        # Should select from each graph proportionally
        # Each graph has 5 nodes, 0.6*5 = 3 per graph -> total 9
        assert len(result.selection_index) == 9
        assert result.selection_index.dtype == torch.long
        assert torch.all(result.selection_index >= 0)
        assert torch.all(result.selection_index < batched_data.num_nodes)

        # Check that indices from each graph are within the correct range
        for i, (start, end) in enumerate(zip([0, 5, 10], [5, 10, 15])):
            mask = (result.selection_index >= start) & (result.selection_index < end)
            count = mask.sum().item()
            assert count == 3  # Each graph should contribute ~3 indices

        # Without replacement, should have no duplicates
        assert len(torch.unique(result.selection_index)) == len(result.selection_index)

    def test_forward_batched_data_min_num_points(self, batched_data):
        """Test batched data with min_num_points constraint."""
        transform = RandomSelect(selection_factor=0.1, min_num_points=2, max_num_points=100)
        result = transform(batched_data)

        # Each graph should have at least 2 nodes
        for i, (start, end) in enumerate(zip([0, 5, 10], [5, 10, 15])):
            mask = (result.selection_index >= start) & (result.selection_index < end)
            count = mask.sum().item()
            assert count >= 2

    def test_forward_batched_data_max_num_points(self, batched_data):
        """Test batched data with max_num_points constraint."""
        transform = RandomSelect(selection_factor=2.0, max_num_points=3, min_num_points=1)
        result = transform(batched_data)

        assert len(result.selection_index) == 3 * 3

    def test_forward_with_tuple_selection_factor(self, simple_data):
        """Test forward pass with tuple selection factor."""
        transform = RandomSelect(selection_factor=(0.3, 0.7), max_num_points=100, min_num_points=1)

        # Run multiple times to ensure the factor varies
        results = []
        for _ in range(10):
            result = transform(simple_data)
            results.append(len(result.selection_index))

        # Should have some variation
        assert len(set(results)) > 1
        assert all(3 <= r <= 7 for r in results)  # 0.3*10=3, 0.7*10=7

    def test_forward_preserves_other_attributes(self, simple_data):
        """Test that the transform preserves other data attributes."""
        transform = RandomSelect(selection_factor=0.5)
        original_x = simple_data.x.clone()
        original_pos = simple_data.pos.clone()

        result = transform(simple_data)

        # Original attributes should be preserved
        assert torch.equal(result.x, original_x)
        assert torch.equal(result.pos, original_pos)
        assert result.num_nodes == simple_data.num_nodes

    def test_sorting_of_indices(self, simple_data):
        """Test that selection_index is always sorted."""
        transform = RandomSelect(selection_factor=0.5)
        result = transform(simple_data)

        # Check that indices are sorted
        assert torch.all(result.selection_index == result.selection_index.sort()[0])

    def test_sorting_with_replacement(self, simple_data):
        """Test that selection_index is sorted with replacement."""
        transform = RandomSelect(selection_factor=2.0, replacement=True)
        result = transform(simple_data)

        # Check that indices are sorted
        assert torch.all(result.selection_index == result.selection_index.sort()[0])

    def test_edge_case_single_node(self):
        """Test with single node."""
        data = Data(pos=torch.randn(1, 3), x=torch.randn(1, 5), num_nodes=1)

        transform = RandomSelect(selection_factor=1.0, min_num_points=1, max_num_points=100)
        result = transform(data)
        assert len(result.selection_index) == 1
        assert result.selection_index[0].item() == 0

        # With replacement and factor > 1
        transform_replacement = RandomSelect(selection_factor=3.0, replacement=True, max_num_points=100)
        result = transform_replacement(data)
        assert len(result.selection_index) == 3
        assert torch.all(result.selection_index == 0)  # All zeros

    def test_edge_case_zero_selection(self):
        """Test when selection_size is 0."""
        data = Data(pos=torch.randn(10, 3), num_nodes=10)
        transform = RandomSelect(selection_factor=0.0, min_num_points=0, max_num_points=100)
        result = transform(data)
        assert len(result.selection_index) == 0

        # With replacement and factor 0
        transform_replacement = RandomSelect(selection_factor=0.0, replacement=True, min_num_points=0)
        result = transform_replacement(data)
        assert len(result.selection_index) == 0

    def test_batched_replacement_with_zero_selection(self, batched_data):
        """Test batched replacement when some graphs have zero selection."""
        transform = RandomSelect(selection_factor=0.0, replacement=True, min_num_points=0, max_num_points=100)
        result = transform(batched_data)
        assert len(result.selection_index) == 0

    def test_no_numpy_usage(self, simple_data):
        """Verify that numpy is not being used in replacement path."""
        transform = RandomSelect(selection_factor=1.5, replacement=True)

        # This should work without numpy
        result = transform(simple_data)
        assert result is not None

        # Check that selection_index is a torch tensor on the correct device
        assert isinstance(result.selection_index, torch.Tensor)
        assert result.selection_index.device == simple_data.pos.device
