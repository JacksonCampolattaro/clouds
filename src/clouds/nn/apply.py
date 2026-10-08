import inspect
import warnings
from collections.abc import Iterable
from typing import Any, get_args, get_origin

import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data


def is_tensor_type(annotation: Any) -> bool:
    """Check if a type annotation is or contains ``torch.Tensor``."""
    if annotation is torch.Tensor or (isinstance(annotation, type) and issubclass(annotation, torch.Tensor)):
        return True

    origin = get_origin(annotation)
    if origin is not None:
        return any(is_tensor_type(arg) for arg in get_args(annotation))

    return isinstance(annotation, type) and issubclass(annotation, torch.Tensor)


def get_param_names(module: Module, rewrite: dict[str, str] | None = None) -> list[str]:
    """Read the tensor-valued parameter names out of a module's ``forward`` signature."""
    param_names = []
    if hasattr(module, 'param_names'):
        param_names = [rewrite.get(name, name) if rewrite else name for name in module.param_names]

    else:
        sig = inspect.signature(module.forward)
        module_name = getattr(module, '__name__', module.__class__.__name__)

        if not sig.parameters:
            raise ValueError(f"Module {getattr(module, '__name__', module.__class__.__name__)} has no parameters")

        for name, param in sig.parameters.items():
            # Validate type annotation
            if name in ['return_emb']:
                continue
            elif param.annotation is inspect.Parameter.empty:
                warnings.warn(
                    f"Parameter '{name}' in module {module_name} has no type annotation. Assuming it's a tensor.", stacklevel=2
                )
            elif not is_tensor_type(param.annotation):
                # Optional non-tensor types can probably be safely ignored
                if type(None) in get_args(param.annotation):
                    continue

                raise TypeError(
                    f"Parameter '{name}' in module {module_name} must be torch.Tensor or Optional[Any], got {param.annotation}"
                )

            param_names.append(rewrite.get(name, name) if rewrite else name)

    return param_names


@torch.compiler.disable(recursive=False)
def apply_to_kwargs(module: Module, rewrite: dict[str, str] | None = None, **kwargs) -> Any:
    """Call ``module`` with the subset of ``kwargs`` matching its ``forward`` parameters."""
    if 'kwargs' in inspect.signature(module.forward).parameters:
        # If a module accepts kwargs, pass everything!
        return module(**kwargs)
    else:
        param_names: list[str] = get_param_names(module, rewrite)
        params = [kwargs.get(name) for name in param_names]
        return module(*params)


@torch.compiler.disable(recursive=False)
def apply_to_data(module: Module, data: Data, rewrite: dict[str, str] | None = None) -> Data:
    """Run ``module`` against a ``Data`` object, writing its output back onto it."""
    # Check if the module natively handles Data as its sole argument
    params = list(inspect.signature(module.forward).parameters.values())
    if params[-1].name == 'data':
        return module(data)

    # Otherwise, extract the necessary parameters
    param_names: list[str] = get_param_names(module, rewrite)
    params = [data.get(name, None) for name in param_names]
    out = module(*params)
    if hasattr(module, 'return_names') and isinstance(module.return_names, Iterable):
        out = [out] if isinstance(out, Tensor) else out
        for key, item in zip(module.return_names, out, strict=False):
            data[key] = item
    else:
        data.x = out

    return data
