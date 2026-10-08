"""Tests for the sequence/kwargs expansion helpers."""

from clouds.utils.sequence import (
    _normalize_list,
    construct_sequence_objects,
    expand_sequence_kwargs,
)


def test_normalize_list_broadcasts_scalars():
    assert list(_normalize_list(5, 3)) == [5, 5, 5]


def test_normalize_list_returns_lists_unchanged():
    assert _normalize_list([1, 2, 3], 3) == [1, 2, 3]


def test_expand_sequence_kwargs_broadcasts_and_transposes():
    assert expand_sequence_kwargs(sequence_length=3, a=[1, 2, 3], b=0) == [
        {'a': 1, 'b': 0},
        {'a': 2, 'b': 0},
        {'a': 3, 'b': 0},
    ]


def test_expand_sequence_kwargs_infers_length_from_first_list():
    assert expand_sequence_kwargs(a=[1, 2, 3]) == [{'a': 1}, {'a': 2}, {'a': 3}]


def test_expand_sequence_kwargs_recurses_into_dicts():
    assert expand_sequence_kwargs(sequence_length=2, a=dict(b=[1, 2], c=0)) == [
        {'a': {'b': 1, 'c': 0}},
        {'a': {'b': 2, 'c': 0}},
    ]


def test_construct_sequence_objects_instantiates_one_per_step():
    assert construct_sequence_objects(dict, dict, sequence_length=2, a=[1, 2]) == [
        {'a': 1},
        {'a': 2},
    ]
