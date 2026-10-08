"""Tests for the multigrid keyword-expansion helpers."""

import pytest

from clouds.utils import (
    _expand_ellipses,
    construct_multigrid_objects,
    construct_object,
    expand_multigrid_kwargs,
)


def test_expand_ellipses_pads_trailing_with_last_value():
    assert _expand_ellipses([1, 2, ...], 4) == [1, 2, 2, 2]


def test_expand_ellipses_pads_leading_with_first_value():
    assert _expand_ellipses([..., 3, 4], 4) == [3, 3, 3, 4]


def test_expand_ellipses_interpolates_scalars():
    assert _expand_ellipses([1, ..., 4], 4) == [1.0, 2.0, 3.0, 4.0]


def test_expand_ellipses_interpolates_pairs():
    out = _expand_ellipses([(0, 1), ..., (2, 4)], 3)
    assert out[0] == (0.0, pytest.approx(4 / 3))
    assert out[1] == (pytest.approx(4 / 3), pytest.approx(8 / 3))
    assert out[2] == (pytest.approx(8 / 3), 4.0)


def test_expand_ellipses_without_ellipsis_is_identity():
    assert _expand_ellipses([1, 2, 3], 3) == [1, 2, 3]


def test_expand_ellipses_rejects_unrecognized_pattern():
    with pytest.raises(ValueError):
        _expand_ellipses([1, 2, ..., 3, 4], 6)


def test_construct_object_passes_through_instances():
    obj = object()
    assert construct_object(object, obj) is obj


def test_construct_object_calls_factories():
    assert construct_object(dict, dict, a=1) == {'a': 1}


def test_construct_object_maps_none_to_none():
    assert construct_object(dict, None) is None


def test_construct_object_rejects_other_values():
    with pytest.raises(NotImplementedError):
        construct_object(dict, 'not callable')


def test_expand_multigrid_kwargs_broadcasts_scalars():
    assert expand_multigrid_kwargs(levels=[0, 1], a=[1, 2], b=0) == {
        0: {'a': 1, 'b': 0},
        1: {'a': 2, 'b': 0},
    }


def test_expand_multigrid_kwargs_infers_levels():
    assert expand_multigrid_kwargs(a=[1, 2, 3]) == {
        0: {'a': 1},
        1: {'a': 2},
        2: {'a': 3},
    }


def test_construct_multigrid_objects_instantiates_per_level():
    objects = construct_multigrid_objects(dict, dict, levels=[0, 1], a=[1, 2], b=0)
    assert objects == {0: {'a': 1, 'b': 0}, 1: {'a': 2, 'b': 0}}


def test_construct_multigrid_objects_shares_instances():
    shared = {'a': 1}
    objects = construct_multigrid_objects(dict, shared, levels=[0, 1])
    assert objects[0] is shared and objects[1] is shared


def test_construct_multigrid_objects_drops_none():
    objects = construct_multigrid_objects(dict, [dict, None], levels=[0, 1])
    assert objects == {0: {}}
