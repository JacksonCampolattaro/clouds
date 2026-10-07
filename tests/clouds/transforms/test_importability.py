import inspect

from torch_geometric.transforms import BaseTransform

import clouds.transforms as transform_module


def test_all_transforms_importable():
    classes = [
        (name, obj)
        for name, obj in inspect.getmembers(transform_module)
        if inspect.isclass(obj) and issubclass(obj, BaseTransform)
    ]

    assert classes

    for name, obj in classes:
        assert getattr(transform_module, name) is obj
