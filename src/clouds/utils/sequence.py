from collections.abc import Callable
from itertools import pairwise
from types import NoneType
from typing import Any

import torch


def construct_object(expected_type: type, factory: Any | Callable | NoneType, **kwargs):
    """Construct an object from a factory, or pass through existing instances.

    ``expected_type`` instances are returned unchanged (shared objects!), callables
    are invoked with ``kwargs``, and ``None`` maps to ``None``.
    """
    if isinstance(factory, expected_type):
        return factory
    elif callable(factory):
        return factory(**kwargs)
    elif factory is None:
        return None
    else:
        raise NotImplementedError(f"construct_object() does not support '{factory}' factory type")


def _expand_ellipses(values: list[Any], n: int) -> list[Any]:
    """Expand a list containing ``...`` into a list of exactly ``n`` elements.

    A trailing/leading ellipsis repeats the adjacent value; a single interior
    ellipsis interpolates (linearly, or pairwise for tuples) between its neighbors.
    """
    if ... in values:
        # Pad with the last value
        if values[-1] is ... and len(values) > 1:
            return values[:-1] + [values[-2]] * (n - len(values) + 1)

        # Pad with the first value
        if values[0] is ... and len(values) > 1:
            return [values[1]] * (n - len(values) + 1) + values[1:]

        # Interpolate between values
        if values[1] is ... and len(values) == 3:
            start, _, end = values

            if isinstance(start, tuple) and isinstance(end, tuple):
                # Pairwise interpolation
                (start, _), (_, end) = start, end
                return list(pairwise(torch.linspace(start, end, steps=n + 1).tolist()))
            else:
                # Conventional interpolation
                # TODO: special case integers?
                return torch.linspace(start, end, steps=n).tolist()

        raise ValueError("unrecognized use of ellipses")

    else:
        return values
