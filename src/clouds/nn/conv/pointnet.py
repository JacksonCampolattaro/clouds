from collections.abc import Iterable
from functools import partial
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import PointNetConv as PyGPointNetConv
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairOptTensor,
    PairTensor,
)

from ..models.mlp import MLP
from ..residual import make_residual_type


def slice(x: OptTensor | PairOptTensor, start: int, end: int) -> OptTensor | PairOptTensor:
    if x is None:
        return None
    if isinstance(x, tuple):
        x_j, x_i = x
        return (x_j, slice(x_i, start, end))

    assert isinstance(x, Tensor)
    return x[start:end, :]


def iter_blocks(
    edge_index: Tensor,
    *args: OptTensor | PairOptTensor,
    block_size: int,
) -> Iterable[tuple]:

    start = 0
    while start < edge_index.size(0):
        end = min(start + block_size, edge_index.size(0))
        yield (
            slice(edge_index, start, end),
            *(slice(x, start, end) for x in args),
        )
        start += block_size


class PointNetConv(PyGPointNetConv):
    # Number of source nodes propagated per chunk at inference. Lowering this
    # reduces peak memory (at some throughput cost) for large graphs; see
    # `AggrConv.MAX_BLOCK_SIZE` for the equivalent aggregation knob.
    max_block_size: int = 2**16

    def __init__(self, *args, edge_scaler: Module | None = None, grad_checkpoint: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.grad_checkpoint = grad_checkpoint
        self.edge_scaler = edge_scaler

    def forward(
        self,
        x: OptTensor | PairOptTensor,
        pos: Tensor | PairTensor,
        edge_index: Adj,
    ) -> Tensor:
        x = (x, None) if not isinstance(x, tuple) else x
        pos = (pos, pos) if isinstance(pos, Tensor) else pos

        if isinstance(edge_index, Tensor) and edge_index.size(0) != 2:
            max_block_size = self.max_block_size
            prop = (
                partial(self.block_source_propagate, max_block_size=max_block_size)
                if edge_index.size(0) > max_block_size and not self.training
                else self.source_propagate
            )

        else:
            prop = self.propagate

        if self.grad_checkpoint and self.training:
            prop = partial(checkpoint, prop, use_reentrant=False)

        out = prop(edge_index, x=x, pos=pos)

        # This part is kept in common
        if self.global_nn is not None:
            out = self.global_nn(out)

        return out

    def message(self, x_j: OptTensor, pos_i: Tensor, pos_j: Tensor) -> Tensor:
        msg = pos_j - pos_i
        if self.edge_scaler:
            msg = self.edge_scaler(msg)
        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=-1)
        if self.local_nn:
            msg = self.local_nn(msg)
        return msg

    # @torch.amp.autocast('cuda', enabled=False)
    @torch.compiler.disable(recursive=False)  # FIXME: why is this necessary?
    def source_propagate(
        self,
        edge_index: Tensor,
        x: tuple[Tensor, Tensor],
        pos: tuple[Tensor, Tensor],
        out: Tensor | None = None,
    ) -> Tensor:
        # More performant routine for source indices

        # Message passing
        out = self.message(
            x_j=None if x[0] is None else x[0][edge_index, :].flatten(end_dim=1),
            pos_i=pos[1].repeat_interleave(edge_index.size(1), dim=0),  # .contiguous(),
            pos_j=pos[0][edge_index, :].flatten(end_dim=1),
        )

        # Aggregation
        out = out.reshape(-1, edge_index.size(1), out.size(-1))
        if self.aggr == 'max':
            out = out.amax(dim=1)  # if self.training or out is None else torch.amax(unpooled, dim=1, out=out)
        elif self.aggr == 'min':
            out = out.amin(dim=1)  # if self.training or out is None else torch.amin(unpooled, dim=1, out=out)
        else:
            raise NotImplementedError(f"Aggr '{self.aggr}' not supported here!")

        return out

    @torch.compiler.disable(recursive=False)
    def block_source_propagate(
        self,
        edge_index: Tensor,
        x: tuple[Tensor, Tensor],
        pos: tuple[Tensor, Tensor],
        max_block_size: int | None = None,
        out: Tensor | None = None,
    ) -> Tensor:
        # TODO: this must be a tensor, so the loop is data-dependent
        # otherwise, torch.compile is smart enough to undo this optimization!

        start = torch.tensor(0).long()
        while (start < edge_index.size(0)).all():
            end = min(start + torch.tensor(max_block_size).long(), torch.tensor(edge_index.size(0)))
            block_out = self.source_propagate(
                slice(edge_index, start.item(), end.item()),
                slice(x, start.item(), end.item()),
                slice(pos, start.item(), end.item()),
            )
            if out is None:
                out = block_out.new_empty([edge_index.size(0), block_out.size(-1)])
            out[start.item() : end.item(), :] = block_out
            start += max_block_size

        assert isinstance(out, Tensor)
        return out

    def __repr__(self) -> str:
        if self.edge_scaler:
            return (
                f'{self.__class__.__name__}(local_nn={self.local_nn}, '
                f'global_nn={self.global_nn}, edge_scaler={self.edge_scaler})'
            )
        else:
            return super().__repr__()


class SimplePointNetConv(PointNetConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str | list[str] | Aggregation | None = 'max',
        aggr_kwargs: dict[str, Any] | None = None,
        decomposed_layers: int = 1,
        grad_checkpoint: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            local_nn=MLP(
                in_channels=in_channels + 3,
                out_channels=out_channels,
                plain_last=True,
                **kwargs,
            ),
            global_nn=None,
            aggr=aggr,
            aggr_kwargs=aggr_kwargs,
            decomposed_layers=decomposed_layers,
            grad_checkpoint=grad_checkpoint,
        )
        self.in_channels, self.out_channels = in_channels, out_channels


ResSimplePointNetConv = make_residual_type(SimplePointNetConv)
