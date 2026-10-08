from collections.abc import Callable

import torch
from pool import maxpool
from torch import Tensor
from torch.nn import Module
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.conv import SimpleConv
from torch_geometric.typing import Adj, OptPairTensor, PairTensor, Size

from ..dense.linear import Linear
from ..models.dela import ResDeLAMLP
from ..models.mlp import MLP
from ..residual import make_residual_type
from ..resolver import normalization_resolver
from ..sequential import Sequential
from .selection import ProjSelectionConv


class DeLAConv(SimpleConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        hidden_factor: float | None = None,
        bias: bool = False,
        **kwargs,
    ) -> None:
        out_channels = out_channels or in_channels
        super().__init__(combine_root=None, aggr='max')
        hidden_channels = int(out_channels * hidden_factor) if hidden_factor else out_channels
        self.lin = Linear(
            in_channels=in_channels,
            out_channels=hidden_channels,
            bias=bias and not hidden_factor,
            **kwargs,
        )
        self.post_lin = (
            Linear(
                in_channels=hidden_channels,
                out_channels=out_channels,
                bias=bias,
                **kwargs,
            )
            if hidden_factor
            else None
        )
        self.in_channels, self.out_channels = in_channels, out_channels

    # @torch.amp.autocast('cuda', enabled=False) # FIXME
    def forward(
        self,
        x: Tensor | PairTensor,
        edge_index: Adj,
        selection_index: OptPairTensor = None,
        size: Size = None,
    ) -> Tensor:
        # TODO: Self-edges????
        x = x if not isinstance(x, tuple) else x[0]
        selection_index = selection_index if not isinstance(selection_index, tuple) else selection_index[0]

        # Apply linear layer
        x = self.lin(x)

        if isinstance(edge_index, Tensor) and edge_index.size(0) != 2:
            # More performant routine for source indices
            assert self.aggr == 'max'
            # TODO: if size is present, check that it matches!
            # TODO: this ought to support torch.compile()!
            if x.dtype == torch.bfloat16 and torch.is_grad_enabled() and x.requires_grad:
                # NOTE: pool.maxpool implements no bf16 backward, so run the
                # aggregation in fp32 (as the original DeLA implementation did).
                with torch.amp.autocast('cuda', enabled=False):
                    out = maxpool(x.to(torch.float32), edge_index).to(x.dtype)
            else:
                # fp32/fp16 (and bf16 forward-only, e.g. under no_grad) are supported.
                out = maxpool(x, edge_index)
        else:
            # Default to PyG's implementation
            out = self.propagate(x=x, edge_index=edge_index, edge_weight=None, size=size)

        if out.shape != x.shape:
            assert isinstance(selection_index, Tensor)
            x = x[selection_index, :]

        out = out - x
        return self.post_lin(out) if self.post_lin else out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(pre={self.lin}, post={self.post_lin})'


ResDeLAConv = make_residual_type(DeLAConv)


class SimpleDeLABlock(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        num_convs_per_block: int = 2,
        res_norm_init: float = 0.2,
        res_dropout_p: float | tuple[float, float] | list[float] = 0.0,
        norm: str | Callable | None = None,
        norm_kwargs: dict | None = None,
        mlp_hidden_factor: float | None = 2,
        conv_hidden_factor: float | None = None,
        **kwargs,
    ) -> None:
        norm_kwargs = norm_kwargs or {}
        out_channels = out_channels if out_channels is not None else in_channels
        if isinstance(res_dropout_p, float):
            res_dropout_p = torch.linspace(0, res_dropout_p, num_convs_per_block).tolist()
        elif isinstance(res_dropout_p, tuple):
            res_dropout_p = torch.linspace(res_dropout_p[0], res_dropout_p[1], num_convs_per_block).tolist()
        assert isinstance(res_dropout_p, list)

        blocks = [
            ResDeLAMLP(
                in_channels=in_channels,
                out_channels=out_channels,
                norm=norm,
                norm_kwargs=norm_kwargs,
                hidden_factor=mlp_hidden_factor,
                # Residual path
                res_norm=norm,
                res_norm_kwargs=dict(init_weight=res_norm_init) | norm_kwargs,
                res_dropout_p=res_dropout_p[0],
                # Identity path
                id_norm=norm,
                id_norm_kwargs=dict(init_weight=(1.0 - res_norm_init)) | norm_kwargs,
                **kwargs,
            )
        ]

        for dropout_p in res_dropout_p:
            blocks.append(
                ResDeLAConv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                    res_norm=norm,
                    res_norm_kwargs=dict(init_weight=0.0) | norm_kwargs,
                    res_dropout_p=dropout_p,
                    hidden_factor=conv_hidden_factor,
                    **kwargs,
                ),
            )

        # FIXME: this should probably be optional

        super().__init__(*blocks)


class DeLAMLPConv(Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
        aggr_channels: int | None = None,
        out_channels: int | None = None,
        norm_kwargs: dict | None = None,
        out_norm_init: float = 0.2,
        **kwargs,
    ) -> None:
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        aggr_channels = aggr_channels or hidden_channels // 2
        super().__init__(
            DeLAConv(in_channels=in_channels, out_channels=aggr_channels, **kwargs),
            MLP(
                channel_list=[aggr_channels, hidden_channels, out_channels],
                layout='lnaln',
                norm_kwargs=[(norm_kwargs or {}), (norm_kwargs or {}) | dict(init_weight=out_norm_init)],
                **kwargs,
            ),
        )


class DeLABlock(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        num_convs_per_block: int = 4,
        # TODO: not implemented!
        res_dropout_p: float | tuple[float, float] | list[float] = 0.0,
        norm: type | None = None,
        res_norm_init: float = 0.2,
        mlp_hidden_factor: float | None = 2,
        conv_hidden_factor: float | None = None,
        grad_checkpoint: bool = False,
        **kwargs,
    ) -> None:
        out_channels = out_channels if out_channels is not None else in_channels
        if isinstance(res_dropout_p, float):
            res_dropout_p = torch.linspace(0, res_dropout_p, num_convs_per_block).tolist()
        elif isinstance(res_dropout_p, tuple):
            res_dropout_p = torch.linspace(res_dropout_p[0], res_dropout_p[1], num_convs_per_block).tolist()
        assert isinstance(res_dropout_p, list) and len(res_dropout_p) == num_convs_per_block

        blocks = [
            ResDeLAMLP(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_factor=mlp_hidden_factor,
                res_norm=norm,
                res_norm_kwargs=dict(init_weight=res_norm_init) | (kwargs.get('norm_kwargs') or {}),
                res_dropout_p=res_dropout_p[0],
                id_norm=norm,
                id_norm_kwargs=dict(init_weight=1.0 - res_norm_init) | (kwargs.get('norm_kwargs') or {}),
                **kwargs,
            ),
        ]
        for i in range(num_convs_per_block):
            blocks.append(
                ResDeLAConv(
                    in_channels=out_channels,
                    res_dropout_p=res_dropout_p[i],
                    res_norm=norm,
                    res_norm_kwargs=dict(init_weight=0.0) | (kwargs.get('norm_kwargs') or {}),
                    hidden_factor=conv_hidden_factor,
                    **kwargs,
                )
            )
            if i % 2 == 1:
                blocks.append(
                    ResDeLAMLP(
                        in_channels=out_channels,
                        hidden_factor=mlp_hidden_factor,
                        res_dropout_p=res_dropout_p[i],
                        res_norm=norm,
                        res_norm_kwargs=dict(init_weight=0.0) | (kwargs.get('norm_kwargs') or {}),
                        **kwargs,
                    )
                )

        self.grad_checkpoint = grad_checkpoint
        super().__init__(*blocks)

    def forward(self, *args):
        if self.grad_checkpoint:
            return checkpoint(super().forward, *args, use_reentrant=False)
        else:
            return super().forward(*args)


class DeLADownsample(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        norm: str | Callable | None = None,
        norm_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        norm_kwargs = norm_kwargs or {}
        super().__init__()
        self.res = DeLAConv(
            in_channels=in_channels,
            out_channels=out_channels,
            norm=norm,
            norm_kwargs=norm_kwargs,
            **kwargs,
        )
        self.res_norm = normalization_resolver(
            norm,
            in_channels=out_channels,
            **(dict(init_weight=0.3) | norm_kwargs),
        )
        self.id = ProjSelectionConv(
            in_channels=in_channels,
            out_channels=out_channels,
            # TODO: yuck.
            norm=norm,
            **(dict(norm_kwargs=dict(init_weight=0.3) | norm_kwargs) | kwargs),
        )
        self.in_channels, self.out_channels = in_channels, out_channels

    def forward(
        self,
        x: Tensor | PairTensor,
        edge_index: Adj,
        selection_index: OptPairTensor = None,
        size: Size = None,
        **kwargs,
    ) -> Tensor:
        res = self.res_norm(self.res(x, edge_index, selection_index, size))
        id = self.id(x, selection_index=selection_index, size=size)
        return id + res
