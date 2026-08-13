import pytest
import torch
from torch import Tensor
from torch_geometric.data import Data

from clouds.transforms.partseg import CategoryClassMask, RefinePartSegmentation, class_map_to_table


class TestClassMapToTable:
    def test_basic_conversion(self):
        class_map = {
            'cat': [0, 1, 2],
            'dog': [1, 3, 4],
        }
        result = class_map_to_table(class_map)
        
        assert result.shape == (2, 5)
        assert result.dtype == torch.bool
        assert result[0, 0:3].all()
        assert result[0, 3:5].all() == False
        assert result[1, [1, 3, 4]].all()
        assert result[1, [0, 2]].all() == False

    def test_non_contiguous_classes(self):
        class_map = {
            'cat': [0, 2, 4],
            'dog': [1, 3, 5],
        }
        result = class_map_to_table(class_map)
        
        assert result.shape == (2, 6)
        assert result[0, [0, 2, 4]].all()
        assert result[1, [1, 3, 5]].all()


class TestCategoryClassMask:
    @pytest.fixture
    def sample_data(self):
        category = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        pred = torch.randn(4, 3)  # 4 nodes, 3 classes
        batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        return Data(category=category, pred=pred, batch=batch)

    @pytest.fixture
    def categories_to_classes(self):
        return torch.tensor([
            [True, True, False],  # category 0: classes 0,1
            [False, True, True],   # category 1: classes 1,2
        ], dtype=torch.bool)

    def test_initialization_with_tensor(self, categories_to_classes):
        transform = CategoryClassMask(categories_to_classes)
        assert isinstance(transform.categories_to_classes, Tensor)
        assert transform.categories_to_classes.shape == (2, 3)

    def test_initialization_with_dict(self):
        class_map = {
            'cat': [0, 1],
            'dog': [1, 2],
        }
        transform = CategoryClassMask(class_map)
        assert isinstance(transform.categories_to_classes, Tensor)
        assert transform.categories_to_classes.shape == (2, 3)

    def test_forward_basic(self, sample_data, categories_to_classes):
        transform = CategoryClassMask(categories_to_classes)
        result = transform(sample_data)
        
        # Check that invalid predictions are set to -inf
        # category 0: classes 0,1 valid; class 2 invalid
        # category 1: classes 1,2 valid; class 0 invalid
        invalid_mask_category0 = sample_data.category == 0
        invalid_mask_category1 = sample_data.category == 1
        
        assert torch.isinf(result.pred[invalid_mask_category0, 2]).all()
        assert torch.isinf(result.pred[invalid_mask_category1, 0]).all()
        
        # Check that valid predictions are unchanged
        assert torch.allclose(result.pred[invalid_mask_category0, 0:2], 
                            sample_data.pred[invalid_mask_category0, 0:2])
        assert torch.allclose(result.pred[invalid_mask_category1, 1:3], 
                            sample_data.pred[invalid_mask_category1, 1:3])

    def test_forward_with_dict_input(self, sample_data):
        class_map = {
            'cat': [0, 1],
            'dog': [1, 2],
        }
        transform = CategoryClassMask(class_map)
        result = transform(sample_data)
        
        assert hasattr(result, 'pred')
        assert result.pred.shape == sample_data.pred.shape

    def test_forward_device_handling(self):
        category = torch.tensor([0, 1], dtype=torch.long)
        pred = torch.randn(2, 3)
        batch = torch.tensor([0, 0], dtype=torch.long)
        data = Data(category=category, pred=pred, batch=batch)
        
        categories_to_classes = torch.tensor([
            [True, True, False],
            [False, True, True],
        ], dtype=torch.bool)
        
        transform = CategoryClassMask(categories_to_classes)
        result = transform(data)
        
        assert result.pred.device == data.category.device




class TestRefinePartSegmentation:
    @pytest.fixture
    def basic_class_map(self):
        return {'cat1': [0, 1], 'cat2': [2, 3]}

    @pytest.fixture
    def basic_data(self):
        data = Data(
            category=torch.tensor([0, 1]),
            pos=torch.randn(4, 3),
            batch=torch.tensor([0, 0, 1, 1]),
            pred=torch.randn(4, 3),
            y=torch.randint(0, 3, (4,)),
        )
        data.node_stores = [data]
        return data

    def test_init_with_tensor(self, basic_class_map):
        """Test initialization with tensor."""
        tensor = class_map_to_table(basic_class_map)
        transform = RefinePartSegmentation(tensor)
        assert torch.equal(transform.categories_to_classes, tensor)
        assert transform.k == 10
        assert transform.neighborhood is True

    def test_init_with_dict(self, basic_class_map):
        """Test initialization with dict."""
        transform = RefinePartSegmentation(basic_class_map)
        assert isinstance(transform.categories_to_classes, Tensor)
        assert transform.k == 10

    def test_init_custom_params(self, basic_class_map):
        """Test initialization with custom parameters."""
        transform = RefinePartSegmentation(basic_class_map, k=5, neighborhood=False)
        assert transform.k == 5
        assert transform.neighborhood is False

    def test_forward_basic_without_neighborhood(self, basic_class_map, basic_data):
        """Test forward pass without neighborhood refinement."""
        transform = RefinePartSegmentation(basic_class_map, neighborhood=False)
        result = transform(basic_data)
        
        # Check that invalid predictions were handled
        # For category 0, class 2 is invalid
        mask_cat0 = result.category == 0
        invalid_mask = result.pred[mask_cat0].argmax(dim=-1) == 2
        # The invalid class should have been set to the minimum value
        assert (result.pred[mask_cat0][invalid_mask, 2] == result.pred[mask_cat0][invalid_mask].amin(dim=-1)).all()

    def test_forward_with_neighborhood(self, basic_class_map, basic_data):
        """Test forward pass with neighborhood refinement."""
        transform = RefinePartSegmentation(basic_class_map, neighborhood=True, k=2)
        result = transform(basic_data)
        
        # Verify that all predicted labels are valid for their category
        for b in range(2):
            batch_mask = result.batch == b
            cat = result.category[b]
            valid_classes = transform.categories_to_classes[cat].nonzero().squeeze().tolist()
            pred_labels = result.pred[batch_mask].argmax(dim=-1)
            assert all(label in valid_classes for label in pred_labels)

    def test_forward_with_rare_labels_rejection(self, basic_class_map):
        """Test that rare labels are rejected."""
        transform = RefinePartSegmentation(basic_class_map, k=2, neighborhood=False)
        
        # Create data with a rare label
        data = Data(
            category=torch.tensor([0]),
            pos=torch.randn(10, 3),
            batch=torch.tensor([0] * 10),
            pred=torch.randn(10, 3),
            y=torch.randint(0, 3, (10,))
        )
        # Make one point predict class 2 (invalid for category 0)
        data.pred[0, 2] = 100.0
        data.pred[0, 0] = -100.0
        data.pred[0, 1] = -100.0
        
        data.node_stores = [data]
        
        result = transform(data)
        # The invalid prediction should be fixed
        assert result.pred[0].argmax() != 2

    def test_forward_handles_missing_attributes(self, basic_class_map):
        """Test forward when store is missing required attributes."""
        transform = RefinePartSegmentation(basic_class_map)
        
        data = Data(
            category=torch.tensor([0]),
            pred=torch.randn(2, 3)  # Missing pos, batch, y
        )
        data.node_stores = [data]
        
        result = transform(data)
        assert torch.equal(result.pred, data.pred)

    def test_forward_multiple_batches(self, basic_class_map):
        """Test forward with multiple batches."""
        transform = RefinePartSegmentation(basic_class_map, k=2, neighborhood=False)
        
        data = Data(
            category=torch.tensor([0, 1, 0, 1]),
            pos=torch.randn(20, 3),
            batch=torch.tensor([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5),
            pred=torch.randn(20, 3),
            y=torch.randint(0, 3, (20,))
        )
        data.node_stores = [data]
        
        result = transform(data)
        assert result.batch.max() == 3
        # All predictions should be valid
        for b in range(4):
            batch_mask = result.batch == b
            cat = result.category[b]
            valid_classes = transform.categories_to_classes[cat].nonzero().squeeze().tolist()
            pred_labels = result.pred[batch_mask].argmax(dim=-1)
            assert all(label in valid_classes for label in pred_labels)

    def test_forward_device_handling(self, basic_class_map):
        """Test that tensors are moved to correct device."""
        transform = RefinePartSegmentation(basic_class_map)
        
        data = Data(
            category=torch.tensor([0, 0], device='cpu'),
            pos=torch.randn(2, 3),
            batch=torch.tensor([0, 0]),
            pred=torch.randn(2, 3),
            y=torch.randint(0, 3, (2,))
        )
        data.node_stores = [data]
        
        result = transform(data)
        assert result.category.device == transform.categories_to_classes.device

    def test_forward_preserves_other_attributes(self, basic_class_map, basic_data):
        """Test that other attributes are preserved."""
        original_y = basic_data.y.clone()
        original_pos = basic_data.pos.clone()
        
        transform = RefinePartSegmentation(basic_class_map)
        result = transform(basic_data)
        
        assert torch.equal(result.y, original_y)
        assert torch.equal(result.pos, original_pos)

    def test_forward_with_all_valid_predictions(self, basic_class_map):
        """Test when all predictions are already valid."""
        transform = RefinePartSegmentation(basic_class_map, neighborhood=False)
        
        # Create data with all valid predictions
        data = Data(
            category=torch.tensor([0, 0]),
            pos=torch.randn(2, 3),
            batch=torch.tensor([0, 0]),
            pred=torch.zeros(2, 3),
            y=torch.randint(0, 3, (2,))
        )
        data.pred[0, 0] = 1.0  # Valid for category 0
        data.pred[1, 1] = 1.0  # Valid for category 0
        
        data.node_stores = [data]
        
        original_pred = data.pred.clone()
        result = transform(data)
        
        # Predictions should be unchanged
        assert torch.equal(result.pred, original_pred)
