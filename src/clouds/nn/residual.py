from collections.abc import Callable
from functools import partial
from typing import Any, Generic, TypeVar

import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn import Linear

from .merge import SumMerge
from .parallel import Parallel
from .resolver import normalization_resolver
from .sequential import Sequential


class Identity(Module):
    def __init__(self, in_channels: int, **kwargs) -> None:
        super().__init__()
        self.in_channels = self.out_channels = in_channels

    def forward(self, x: Tensor) -> Tensor:
        return x

    def extra_repr(self) -> str:
        return f"{self.in_channels}"


class ShapeMatch(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        weight_initializer: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        assert out_channels > in_channels
        self.lin = Linear(in_channels, out_channels - in_channels, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([x, self.lin(x)], dim=-1)


def make_shapematch(
    in_channels: int,
    out_channels: int,
    bias: bool = False,
    force_lin: bool = False,
    weight_initializer: str | None = None,
    **kwargs,
):
    if (in_channels == out_channels or in_channels == 0) and not force_lin:
        return Identity(in_channels=in_channels)
    else:
        return Linear(in_channels=in_channels, out_channels=out_channels, bias=bias, **kwargs)


def make_identity_path(
    in_channels: int,
    out_channels: int,
    force_lin: bool,
    norm: str | Callable | None,
    norm_kwargs: dict[str, Any] | None,
    weight_initializer: str | None = None,
):
    blocks = []

    blocks.append(
        make_shapematch(
            in_channels=in_channels,
            out_channels=out_channels,
            force_lin=force_lin,
            weight_initializer=weight_initializer,
        )
    )

    if norm and not isinstance(blocks[0], Identity):
        blocks.append(normalization_resolver(norm, in_channels=out_channels, **norm_kwargs))

    if len(blocks) == 1:
        return blocks[0]

    if len(blocks) == 0:
        return Identity(in_channels)

    return Sequential(*blocks)


def make_residual_path(
    module: Module | tuple,
    norm: str | Callable | None = None,
    norm_kwargs: dict[str, Any] | None = None,
    norm_init: float | None = None,
    dropout: type | None = None,
    dropout_p: float = 0,
):
    blocks = [module]
    out_channels = module[0].out_channels if isinstance(module, tuple) else module.out_channels

    if norm:
        blocks.append(normalization_resolver(norm, in_channels=out_channels, **norm_kwargs))

    if dropout_p > 0:
        assert isinstance(dropout, type)
        blocks.append(dropout(in_channels=out_channels, p=dropout_p))

    return blocks[0] if len(blocks) == 1 else Sequential(*blocks)


class Residual(Parallel):
    def __init__(
        self,
        module: Module | Callable,
        res_norm: str | Callable | None = None,
        res_norm_kwargs: dict | None = None,
        res_norm_init: float | None = None,
        res_dropout: type | None = None,
        res_dropout_p: float = 0.0,
        id_norm: str | Callable | None = None,
        id_norm_kwargs: dict | None = None,
        id_norm_init: float | None = None,
        force_lin: bool = False,
        **kwargs,
    ) -> None:
        if not isinstance(module, Module):
            module = module(**kwargs)
        self.in_channels = module.in_channels
        self.out_channels = getattr(module, 'out_channels', self.in_channels)

        super().__init__(
            (
                'id',
                make_identity_path(
                    self.in_channels,
                    self.out_channels,
                    force_lin=force_lin,
                    norm=id_norm,
                    norm_kwargs=(kwargs.get('norm_kwargs') or {}) | (id_norm_kwargs or {}),
                    weight_initializer=kwargs.get('weight_initializer'),
                ),
            ),
            (
                'res',
                make_residual_path(
                    module=module,
                    norm=res_norm,
                    norm_kwargs=(kwargs.get('norm_kwargs') or {}) | (res_norm_kwargs or {}),
                    dropout=res_dropout,
                    dropout_p=res_dropout_p,
                ),
            ),
            merge=SumMerge,
        )


def make_residual(module: type, **kwargs):
    return partial(Residual, module=module, **kwargs)


M = TypeVar("M", bound=Module)


class ResidualGeneric(Residual, Generic[M]):
    Module: type[M]

    def __init__(self, **kwargs) -> None:
        super().__init__(module=self.Module, **kwargs)


def make_residual_type(module: type[M]) -> type[ResidualGeneric[M]]:
    cls = type(f"Res{module.__name__}", (ResidualGeneric,), {"Module": module})
    return cls  # type: ignore[return-value]
