from collections.abc import Iterable
from typing import Any

from torch.nn import Module

from .sequence import _expand_ellipses, construct_object


def _normalize_level_dict(v: Any | list[Any], levels: Iterable[int]) -> dict[int, Any]:
    if isinstance(v, list):
        if ... in v:
            v = _expand_ellipses(v, n=max(levels) + 1)

        return {level: v[level] for level in levels}

    elif isinstance(v, dict):
        # Recursive! This way e.g. conv_kwargs=dict(...) is handled correctly
        return expand_multigrid_kwargs(levels=levels, **v)

    return {level: v for level in levels}


def expand_multigrid_kwargs(levels: Iterable[int] | None = None, **kwargs) -> dict[int, dict]:
    """Broadcast per-level kwargs into an explicit ``{level: {name: value}}`` mapping.

    Values may be scalars (broadcast to every level), lists (one entry per level,
    optionally using ``...`` interpolation), or nested dicts (expanded recursively).
    If ``levels`` is omitted it is inferred from the first concrete list.
    """
    # Determine how many levels are present
    if levels is None:
        lists = [v for v in kwargs.values() if isinstance(v, list) and ... not in v]
        levels = list(range(len(lists[0]) if lists else 1))

    # TODO: this can be simplified, like expand_sequence_kwargs!
    # Convert all members to level dicts
    columns = {k: _normalize_level_dict(v, levels) for k, v in kwargs.items()}

    # Unpack the provided features for each level
    return {level: {k: columns[k][level] for k in columns} for level in levels}


def construct_multigrid_objects(
    expected_type: type,
    factory: Any | list[Any],
    levels: Iterable[int] | None = None,
    **kwargs,
) -> dict[int, Any]:
    """Instantiate one object per level from a factory and per-level kwargs.

    A single instance of ``expected_type`` is shared across all levels (weight
    sharing); a list provides one factory per level; any other callable is invoked
    once per level. Entries explicitly set to ``None`` are dropped.
    """
    scale_kwargs = expand_multigrid_kwargs(**kwargs, levels=levels).items()
    if issubclass(type(factory), expected_type):
        # If the user provided one complete object, we'll apply it to all scales (shared weights!)
        objects = {s: factory for s, _ in scale_kwargs}
    elif isinstance(factory, list):
        # If a list was passed, use the kwargs to instantiate the appropriate object for each scale
        objects = {
            s: construct_object(expected_type, f, **s_kwargs) for f, (s, s_kwargs) in zip(factory, scale_kwargs, strict=False)
        }
    else:
        # Otherwise, try to invoke the factory object for each scale
        objects = {s: construct_object(expected_type, factory, **s_kwargs) for s, s_kwargs in scale_kwargs}

    # Discard entries which were explicitly set to None
    return {s: m for s, m in objects.items() if m is not None}


def construct_multigrid_modules(factory: Any | list[Any], levels: Iterable[int] | None = None, **kwargs) -> dict[int, Module]:
    return construct_multigrid_objects(Module, factory, levels, **kwargs)
