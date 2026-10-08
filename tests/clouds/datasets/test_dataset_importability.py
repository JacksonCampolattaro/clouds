import inspect

from torch_geometric.data import Dataset

import clouds.datasets as dataset_module


def test_all_datasets_importable():
    classes = [
        (name, obj)
        for name, obj in inspect.getmembers(dataset_module)
        if inspect.isclass(obj) and issubclass(obj, Dataset)
    ]

    assert classes

    for name, obj in classes:
        assert getattr(dataset_module, name) is obj
