from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn.conv import SimpleConv as PyGSimpleConv
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
)

from ..apply import apply_to_kwargs
from ..dense.linear import Linear
from ..models.mlp import MLP
from ..resolver import normalization_resolver

# Number of source nodes aggregated per chunk for source-indexed edges. Chunking
# bounds the peak memory of the `(N, K, C)` gather to `(max_block_size, K, C)`,
# which matters for large (e.g. test-time voting) graphs.
MAX_BLOCK_SIZE = 2**15


# TODO: why does jit need to be diabled?
@torch.compiler.disable
class AggrConv(PyGSimpleConv):
    def __init__(self, in_channels: int, aggr: str = 'max', max_block_size: int = MAX_BLOCK_SIZE, **_) -> None:
        super().__init__(aggr=aggr)
        self.out_channels = self.in_channels = in_channels
        self.max_block_size = max_block_size

    def forward(
        self,
        x: Tensor | OptPairTensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        x = (x, x) if isinstance(x, Tensor) else x

        if isinstance(edge_index, Tensor) and edge_index.size(0) != 2:
            # More performant routine for source indices
            assert edge_weight is None
            return self.source_aggregate(x[0], edge_index)

        # Default to PyG's implementation
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

    def _aggregate(self, messages: Tensor, k: int) -> Tensor:
        out = messages.reshape(-1, k, self.out_channels)
        if self.aggr == 'max':
            # TODO: add maxpool support!
            return out.amax(dim=1)
        elif self.aggr == 'min':
            return out.amin(dim=1)
        elif self.aggr == 'sum':
            return out.sum(dim=1)
        elif self.aggr == 'mean':
            return out.mean(dim=1)
        else:
            raise NotImplementedError(f"Aggr '{self.aggr}' not supported here!")

    @torch.compiler.disable(recursive=False)
    def source_aggregate(self, x: Tensor, edge_index: Tensor) -> Tensor:
        num_nodes, k = edge_index.shape

        if num_nodes <= self.max_block_size:
            messages = self.message(x_j=x[edge_index, :].flatten(end_dim=1), edge_weight=None)
            return self._aggregate(messages, k)

        out = x.new_empty((num_nodes, self.out_channels))
        for start in range(0, num_nodes, self.max_block_size):
            end = min(start + self.max_block_size, num_nodes)
            messages = self.message(x_j=x[edge_index[start:end, :], :].flatten(end_dim=1), edge_weight=None)
            out[start:end] = self._aggregate(messages, k)
        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(aggr={self.aggr})'


class LinAggrConv(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'max',
        norm: str | Callable | type | None = 'batch',
        norm_kwargs: dict[str, Any] | None = None,
        bias: bool = False,
        **kwargs,
    ) -> None:
        norm_kwargs = norm_kwargs or {}
        super().__init__()
        self.norm = (
            norm(in_channels=in_channels, **norm_kwargs)
            if isinstance(norm, type)
            else normalization_resolver(norm, in_channels=in_channels, **norm_kwargs)
        )
        self.lin = Linear(in_channels, out_channels, bias=bias, **kwargs)
        self.aggr = AggrConv(out_channels, aggr=aggr)
        self.in_channels, self.out_channels = in_channels, self.aggr.out_channels

    def forward(
        self,
        x: Tensor | OptPairTensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        x: OptPairTensor = (x, None) if isinstance(x, Tensor) else x
        x = (self.norm(x[0]), x[1]) if self.norm else x
        x = (self.lin(x[0]), x[1])
        x = self.aggr(x, edge_index, edge_weight=edge_weight, size=size)
        return x


class AggrLinConv(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'max',
        norm: str | Callable | type | None = 'batch',
        norm_kwargs: dict[str, Any] | None = None,
        bias: bool = False,
        **kwargs,
    ) -> None:
        norm_kwargs = norm_kwargs or {}
        super().__init__()
        self.norm = (
            norm(in_channels=in_channels, **norm_kwargs)
            if isinstance(norm, type)
            else normalization_resolver(norm, in_channels=in_channels, **norm_kwargs)
        )
        self.aggr = AggrConv(in_channels, aggr=aggr)
        self.lin = Linear(in_channels, out_channels, bias=bias, **kwargs)
        self.in_channels, self.out_channels = in_channels, self.lin.out_channels

    def forward(
        self,
        x: Tensor | OptPairTensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        x: OptPairTensor = (x, None) if isinstance(x, Tensor) else x
        x = (self.norm(x[0]), x[1]) if self.norm else x
        x = self.aggr(x, edge_index, edge_weight=edge_weight, size=size)
        x = self.lin(x)
        return x


class ProjAggrConv(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'max',
        norm: str | Callable | type | None = 'batch',
        norm_kwargs: dict[str, Any] | None = None,
        bias: bool = False,
        **kwargs,
    ) -> None:
        norm_kwargs = norm_kwargs or {}
        super().__init__()
        self.lin = Linear(in_channels, out_channels, bias=bias, **kwargs)
        self.aggr = AggrConv(out_channels, aggr=aggr)
        self.norm = (
            norm(in_channels=out_channels, **norm_kwargs)
            if isinstance(norm, type)
            else normalization_resolver(norm, in_channels=out_channels, **norm_kwargs)
        )
        self.in_channels, self.out_channels = in_channels, self.aggr.out_channels

    def forward(
        self,
        x: Tensor | OptPairTensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        x: OptPairTensor = (x, None) if isinstance(x, Tensor) else x
        x = (self.lin(x[0]), x[1])
        x = self.aggr(x, edge_index, edge_weight=edge_weight, size=size)
        x = self.norm(x) if self.norm else x
        return x


class AggrProjConv(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        aggr: str = 'max',
        **kwargs,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.aggr = AggrConv(in_channels, aggr=aggr)
        self.lin = MLP([in_channels, out_channels], layout='ln', **kwargs)
        self.in_channels, self.out_channels = in_channels, out_channels

    def forward(
        self,
        x: Tensor | OptPairTensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        x: OptPairTensor = (x, None) if isinstance(x, Tensor) else x
        x = self.aggr(x, edge_index, edge_weight=edge_weight, size=size)
        return self.lin(x)


class ProjAggrProjConv(Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
        out_channels: int | None = None,
        aggr: str = 'max',
        **kwargs,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or min(in_channels, out_channels)
        self.pre = Linear(in_channels, hidden_channels, **kwargs)
        self.aggr = AggrConv(hidden_channels, aggr=aggr)
        self.post = MLP([hidden_channels, out_channels], layout='ln', **kwargs)
        self.in_channels, self.out_channels = in_channels, out_channels

    def forward(
        self,
        x: Tensor | OptPairTensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        x: OptPairTensor = (x, None) if isinstance(x, Tensor) else x
        x = (self.pre(x[0]), x[1])
        x = self.aggr(x, edge_index, edge_weight=edge_weight, size=size)
        return self.post(x)


class MLPAggrConv(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'max',
        **kwargs,
    ) -> None:
        super().__init__()
        self.mlp = MLP(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.aggr = AggrConv(out_channels, aggr=aggr)
        self.in_channels, self.out_channels = in_channels, self.aggr.out_channels

    def forward(
        self,
        x: Tensor | OptPairTensor,
        batch: Tensor | OptPairTensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        x: OptPairTensor = (x, None) if isinstance(x, Tensor) else x
        batch: OptPairTensor = (batch, None) if isinstance(batch, Tensor) else batch
        x = (apply_to_kwargs(self.mlp, x=x[0], batch=batch[0]), x[1])
        x = self.aggr(x, edge_index, edge_weight=edge_weight, size=size)
        return x

    def reset_parameters(self) -> None:
        self.mlp.reset_parameters()
