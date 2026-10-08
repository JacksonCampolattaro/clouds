from collections.abc import Callable, Iterable
from itertools import pairwise, repeat, zip_longest
from types import NoneType
from typing import Any

import torch
from torch.nn import Module


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


def _normalize_list(v: Any | list[Any], n: int) -> Iterable[Any]:
    """Broadcast a scalar, list, or nested dict of values into ``n`` entries.

    Lists containing ``...`` are expanded (see ``_expand_ellipses``); a dict is
    expanded recursively via ``expand_sequence_kwargs``; anything else is
    repeated ``n`` times.
    """
    if isinstance(v, list):
        if ... in v:
            v = _expand_ellipses(v, n)

        assert len(v) == n
        return v
    elif isinstance(v, dict):
        # Recursive! This way e.g. conv_kwargs=dict(...) is handled correctly
        return expand_sequence_kwargs(sequence_length=n, **v)

    return repeat(v, n)


def expand_sequence_kwargs(sequence_length: int | None = None, **kwargs) -> list[dict]:
    """Broadcast per-step kwargs into an explicit list of ``{name: value}`` dicts.

    Values may be scalars (broadcast to every step), lists (one entry per step,
    optionally using ``...`` interpolation), or nested dicts (expanded
    recursively). If ``sequence_length`` is omitted it is inferred from the first
    concrete list.
    """
    # Infer the length from the first concrete list, if not provided
    if not sequence_length:
        lists = [v for v in kwargs.values() if isinstance(v, list) and ... not in v]
        sequence_length = len(lists[0]) if lists else 1

    # Convert all members to iterables
    columns = {k: _normalize_list(v, sequence_length) for k, v in kwargs.items()}

    # Transpose struct-of-lists to list-of-structs
    return [dict(zip(columns, row, strict=True)) for row in zip_longest(*columns.values())]


def construct_sequence_objects(
    expected_type: type,
    factory: Any | list[Any],
    **kwargs,
) -> list[Any]:
    """Instantiate one object per step from a factory and per-step kwargs.

    A list provides one factory per step; any other callable (or instance) is
    invoked once per step.
    """
    elem_kwargs = expand_sequence_kwargs(**kwargs)
    if isinstance(factory, list):
        factory = _normalize_list(factory, n=len(elem_kwargs))

        # If a list was passed, use the kwargs to instantiate the appropriate object for each scale
        return [construct_object(expected_type, f, **e_kwargs) for f, e_kwargs in zip(factory, elem_kwargs, strict=False)]
    else:
        # Otherwise, try to invoke the factory object for each scale
        return [construct_object(expected_type, factory, **e_kwargs) for e_kwargs in elem_kwargs]


def construct_sequential_modules(
    factory: Any | list[Any],
    in_channels: int | None = None,
    **kwargs,
) -> list[Any]:
    """Construct a chain of modules whose channel counts feed forward."""
    elem_kwargs = expand_sequence_kwargs(**kwargs)
    factory = _normalize_list(factory, n=len(elem_kwargs))

    modules = []
    for f, e_kwargs in zip(factory, elem_kwargs, strict=True):
        e_kwargs.setdefault('in_channels', in_channels)
        modules.append(construct_object(Module, f, **e_kwargs))
        in_channels = modules[-1].out_channels

    return modules
