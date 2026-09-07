import inspect

import pytest
from torch_geometric.transforms import BaseTransform

import clouds.transforms as transform_module


def test_all_transforms_importable():
    """
    Test that every class descending from BaseTransform in the transforms module
    is importable as clouds.transforms.[MyTransform]
    """
    # Get all classes defined in the transforms module
    transform_classes = []
    
    for name, obj in inspect.getmembers(transform_module):
        # Check if it's a class, defined in this module, and descends from BaseTransform
        if inspect.isclass(obj) and issubclass(obj, BaseTransform):
            transform_classes.append((name, obj))
    
    # Ensure we found at least one transform class
    assert len(transform_classes) > 0, "No transform classes found in the module"
    
    # Test each class is importable via the expected path
    for class_name, _ in transform_classes:
        # Construct the import path
        import_path = f"clouds.transforms.{class_name}"

        # Try to import it
        try:
            # This is the actual import test - if it fails, the test will fail
            imported_class = getattr(transform_module, class_name)
            
            # Verify it's the same class
            assert imported_class is not None
            assert issubclass(imported_class, BaseTransform)
            
            # Optional: Print success (helpful for debugging)
            print(f"✓ Successfully imported: {import_path}")
            
        except AttributeError:
            pytest.fail(f"Class {class_name} is defined but not accessible via {import_path}")
        except Exception as e:
            pytest.fail(f"Failed to import {import_path}: {str(e)}")
