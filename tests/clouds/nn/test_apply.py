"""Sanity tests for the module-application helpers in clouds.nn.apply."""

import typing

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from clouds.nn.apply import apply_to_data, apply_to_kwargs, get_param_names, is_tensor_type


def test_is_tensor_type_accepts_tensors_and_optional_tensors():
    assert is_tensor_type(torch.Tensor)
    assert is_tensor_type(torch.Tensor | None)
    assert not is_tensor_type(int)


def test_get_param_names_reads_forward_signature():
    class Fwd(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

    assert get_param_names(Fwd()) == ['x', 'y']


def test_get_param_names_honours_rewrite():
    class Fwd(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    assert get_param_names(Fwd(), rewrite={'x': 'pos'}) == ['pos']


def test_apply_to_kwargs_passes_parameters_by_name():
    class Adder(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

    out = apply_to_kwargs(Adder(), x=torch.tensor(1.0), y=torch.tensor(2.0))
    assert out.item() == pytest.approx(3.0)


def test_apply_to_data_extracts_tensors_from_the_data_object():
    class Double(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2

    data = apply_to_data(Double(), Data(x=torch.ones(2, 2)))
    assert torch.equal(data.x, torch.full((2, 2), 2.0))


def test_apply_to_data_prefers_native_data_modules():
    class Native(nn.Module):
        def forward(self, data: Data) -> Data:
            data.x = data.x + 1
            return data

    data = apply_to_data(Native(), Data(x=torch.zeros(2, 2)))
    assert torch.equal(data.x, torch.ones(2, 2))


def test_apply_to_data_writes_return_names_back_onto_data():
    class Pair(nn.Module):
        return_names: typing.ClassVar[list[str]] = ['a', 'b']

        def forward(self, x: torch.Tensor):
            return x, x + 1

    data = apply_to_data(Pair(), Data(x=torch.zeros(2)))
    assert torch.equal(data['a'], torch.zeros(2))
    assert torch.equal(data['b'], torch.ones(2))
