import pytest
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.typing import WITH_KNN as HAS_PYG_KNN

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
            pred=torch.randn(4, 4),
            y=torch.randint(0, 3, (4,)),
        )
        return data

    def test_init_with_tensor(self, basic_class_map):
        """Test initialization with tensor."""
        tensor = class_map_to_table(basic_class_map)
        transform = RefinePartSegmentation(tensor)
        assert torch.equal(transform.categories_to_classes, tensor)
        assert transform.k == 10
        assert transform.replace_with_neighbors is True

    def test_init_with_dict(self, basic_class_map):
        """Test initialization with dict."""
        transform = RefinePartSegmentation(basic_class_map)
        assert isinstance(transform.categories_to_classes, Tensor)
        assert transform.k == 10

    def test_init_custom_params(self, basic_class_map):
        """Test initialization with custom parameters."""
        transform = RefinePartSegmentation(basic_class_map, k=5, replace_with_neighbors=False)
        assert transform.k == 5
        assert transform.replace_with_neighbors is False

    @pytest.mark.skipif(not HAS_PYG_KNN, reason="PyG kNN not installed")
    def test_forward_with_neighborhood(self, basic_class_map, basic_data):
        """Test forward pass with neighborhood refinement."""
        transform = RefinePartSegmentation(basic_class_map, replace_with_neighbors=True, k=2)
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
        transform = RefinePartSegmentation(basic_class_map, k=2, replace_with_neighbors=False)
        
        # Create data with a rare label
        data = Data(
            category=torch.tensor([0]),
            pos=torch.randn(10, 3),
            batch=torch.tensor([0] * 10),
            pred=torch.randn(10, 4),
            y=torch.randint(0, 3, (10,))
        )
        # Make one point predict class 2 (invalid for category 0)
        data.pred[0, 2] = 100.0
        data.pred[0, 0] = -100.0
        data.pred[0, 1] = -100.0
        
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
        
        result = transform(data)
        assert torch.equal(result.pred, data.pred)

    def test_forward_multiple_batches(self, basic_class_map):
        """Test forward with multiple batches."""
        transform = RefinePartSegmentation(basic_class_map, k=0, replace_with_neighbors=False)

        data = Data(
            category=torch.tensor([0, 1, 0, 1]),
            pos=torch.randn(20, 3),
            batch=torch.tensor([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5),
            pred=torch.randn(20, 4),
            y=torch.randint(0, 3, (20,))
        )
        
        result = transform(data)
        assert result.batch.max() == 3
        # All predictions should be valid
        for b in range(4):
            batch_indices = (result.batch == b).nonzero()
            cat = result.category[b]
            valid_classes = transform.categories_to_classes[cat].nonzero().squeeze().tolist()
            pred_labels = result.pred[batch_indices, :].argmax(dim=-1)
            assert all(label in valid_classes for label in pred_labels)

    def test_forward_with_all_valid_predictions(self, basic_class_map):
        """Test when all predictions are already valid."""
        transform = RefinePartSegmentation(basic_class_map, k=0, replace_with_neighbors=False)

        # Create data with all valid predictions
        data = Data(
            category=torch.tensor([0, 0]),
            pos=torch.randn(2, 3),
            batch=torch.tensor([0, 0]),
            pred=torch.zeros(2, 4),
            y=torch.randint(0, 3, (2,))
        )
        data.pred[0, 0] = 1.0  # Valid for category 0
        data.pred[1, 1] = 1.0  # Valid for category 0

        original_pred = data.pred.argmax(dim=-1)
        result_pred = transform(data).pred.argmax(dim=-1)

        # Predictions should be unchanged
        assert torch.equal(result_pred, original_pred)
