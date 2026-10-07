import pytest
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.typing import WITH_KNN as HAS_PYG_KNN

from clouds.transforms.partseg import CategoryClassMask, RefinePartSegmentation, class_map_to_table


def test_class_map_to_table_basic_conversion():
    result = class_map_to_table({'cat': [0, 1, 2], 'dog': [1, 3, 4]})

    assert result.shape == (2, 5)
    assert result.dtype == torch.bool
    assert result[0, 0:3].all()
    assert not result[0, 3:5].any()
    assert result[1, [1, 3, 4]].all()
    assert not result[1, [0, 2]].any()


def test_class_map_to_table_non_contiguous_classes():
    result = class_map_to_table({'cat': [0, 2, 4], 'dog': [1, 3, 5]})

    assert result.shape == (2, 6)
    assert result[0, [0, 2, 4]].all()
    assert result[1, [1, 3, 5]].all()


@pytest.fixture
def sample_data():
    return Data(
        category=torch.tensor([0, 1], dtype=torch.long),
        pred=torch.randn(4, 3),
        batch=torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )


@pytest.fixture
def categories_to_classes():
    return torch.tensor(
        [
            [True, True, False],
            [False, True, True],
        ]
    )


def test_category_class_mask_init_with_tensor(categories_to_classes):
    transform = CategoryClassMask(categories_to_classes)

    assert isinstance(transform.categories_to_classes, Tensor)
    assert transform.categories_to_classes.shape == (2, 3)


def test_category_class_mask_init_with_dict():
    transform = CategoryClassMask({'cat': [0, 1], 'dog': [1, 2]})

    assert isinstance(transform.categories_to_classes, Tensor)
    assert transform.categories_to_classes.shape == (2, 3)


def test_category_class_mask_forward_with_dict_input(sample_data):
    assert CategoryClassMask({'cat': [0, 1], 'dog': [1, 2]})(sample_data).pred.shape == sample_data.pred.shape


def test_category_class_mask_forward_device_handling():
    data = Data(category=torch.tensor([0, 1], dtype=torch.long), pred=torch.randn(2, 3), batch=torch.tensor([0, 0]))

    assert CategoryClassMask(torch.tensor([[True, True, False], [False, True, True]]))(data).pred.device == data.category.device


@pytest.fixture
def basic_class_map():
    return {'cat1': [0, 1], 'cat2': [2, 3]}


@pytest.fixture
def basic_data():
    return Data(
        category=torch.tensor([0, 1]),
        pos=torch.randn(4, 3),
        batch=torch.tensor([0, 0, 1, 1]),
        pred=torch.randn(4, 4),
        y=torch.randint(0, 3, (4,)),
    )


def test_refine_init_with_tensor(basic_class_map):
    tensor = class_map_to_table(basic_class_map)
    transform = RefinePartSegmentation(tensor)

    assert torch.equal(transform.categories_to_classes, tensor)
    assert transform.k == 10
    assert transform.replace_with_neighbors is True


def test_refine_init_with_dict(basic_class_map):
    transform = RefinePartSegmentation(basic_class_map)

    assert isinstance(transform.categories_to_classes, Tensor)
    assert transform.k == 10


def test_refine_init_custom_params(basic_class_map):
    transform = RefinePartSegmentation(basic_class_map, k=5, replace_with_neighbors=False)

    assert transform.k == 5
    assert transform.replace_with_neighbors is False


@pytest.mark.skipif(not HAS_PYG_KNN, reason='PyG kNN not installed')
def test_refine_forward_with_neighborhood(basic_class_map, basic_data):
    transform = RefinePartSegmentation(basic_class_map, replace_with_neighbors=True, k=1)
    result = transform(basic_data)

    for b in range(2):
        valid_classes = transform.categories_to_classes[result.category[b]].nonzero().squeeze().tolist()
        assert all(label in valid_classes for label in result.pred[result.batch == b].argmax(dim=-1))


def test_refine_forward_with_rare_labels_rejection(basic_class_map):
    data = Data(
        category=torch.tensor([0]),
        pos=torch.randn(10, 3),
        batch=torch.tensor([0] * 10),
        pred=torch.randn(10, 4),
        y=torch.randint(0, 3, (10,)),
    )
    data.pred[0, 2] = 100.0
    data.pred[0, 0] = -100.0
    data.pred[0, 1] = -100.0

    assert RefinePartSegmentation(basic_class_map, k=2, replace_with_neighbors=False)(data).pred[0].argmax() != 2


def test_refine_forward_handles_missing_attributes(basic_class_map):
    data = Data(category=torch.tensor([0]), pred=torch.randn(2, 3))

    assert torch.equal(RefinePartSegmentation(basic_class_map)(data).pred, data.pred)


def test_refine_forward_multiple_batches(basic_class_map):
    data = Data(
        category=torch.tensor([0, 1, 0, 1]),
        pos=torch.randn(20, 3),
        batch=torch.tensor([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5),
        pred=torch.randn(20, 4),
        y=torch.randint(0, 3, (20,)),
    )

    transform = RefinePartSegmentation(basic_class_map, k=0, replace_with_neighbors=False)
    result = transform(data)

    assert result.batch.max() == 3
    for b in range(4):
        valid = transform.categories_to_classes[result.category[b]].nonzero().squeeze().tolist()
        assert all(label in valid for label in result.pred[(result.batch == b).nonzero(), :].argmax(dim=-1))


def test_refine_forward_with_all_valid_predictions(basic_class_map):
    data = Data(
        category=torch.tensor([0, 0]),
        pos=torch.randn(2, 3),
        batch=torch.tensor([0, 0]),
        pred=torch.zeros(2, 4),
        y=torch.randint(0, 3, (2,)),
    )
    data.pred[0, 0] = 1.0
    data.pred[1, 1] = 1.0

    assert torch.equal(
        RefinePartSegmentation(basic_class_map, k=0, replace_with_neighbors=False)(data).pred.argmax(dim=-1),
        data.pred.argmax(dim=-1),
    )
