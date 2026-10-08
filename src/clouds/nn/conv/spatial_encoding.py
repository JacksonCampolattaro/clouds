from typing import Any

import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairOptTensor,
    PairTensor,
    SparseTensor,
)

from ..models.mlp import MLP
from ..resolver import normalization_resolver
from .pointnet import PointNetConv


class SpatialEncodingConv(PointNetConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr_channels: int | None = None,
        hidden_channels: int | list[int] | None = None,
        init_weight: float | None = None,
        edge_scaler: Module | None = None,
        edge_mlp_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        rel_channels = getattr(edge_scaler, 'out_channels', 3)
        self.in_channels, self.out_channels = in_channels, out_channels
        hidden_channels = hidden_channels or (aggr_channels or out_channels) // 2  # TODO: does this make sense?
        if not isinstance(hidden_channels, list):
            hidden_channels = [hidden_channels // 2, hidden_channels]
        super().__init__(
            local_nn=MLP(
                channel_list=[
                    self.in_channels + rel_channels,
                    *hidden_channels,
                    aggr_channels or out_channels,
                ],
                plain_last=True,
                **(kwargs | (edge_mlp_kwargs or {})),
            ),
            global_nn=(
                MLP(
                    channel_list=[aggr_channels, self.out_channels],
                    plain_last=False,
                    layout='ln',
                    **kwargs,
                )
                if aggr_channels
                else normalization_resolver(
                    kwargs['norm'],
                    in_channels=out_channels,
                    **(dict(init_weight=init_weight) | (kwargs.get('norm_kwargs') or {})),
                )
                if kwargs.get('norm')
                else None
            ),
            edge_scaler=edge_scaler,
        )


class SplitSpatialEncodingConv(Module):
    def __init__(
        self,
        in_channels: int,
        aggr_channels: int | None = None,
        out_channels: int = 96,
        hidden_channels: int | list[int] | None = None,
        init_weight: float | None = None,
        edge_scaler: Module | None = None,
        edge_mlp_kwargs: dict[str, Any] | None = None,
        node_mlp_kwargs: dict[str, Any] | None = None,
        # TODO: merge option
        **kwargs,
    ) -> None:
        rel_channels = getattr(edge_scaler, 'out_channels', 3)
        self.in_channels, self.out_channels = in_channels, out_channels
        hidden_channels = hidden_channels or (aggr_channels or out_channels) // 2  # TODO: does this make sense?
        if not isinstance(hidden_channels, list):
            hidden_channels = [hidden_channels // 2, hidden_channels]
        super().__init__()
        self.edge_nn = PointNetConv(
            local_nn=MLP(
                channel_list=[
                    rel_channels,
                    *hidden_channels,
                    aggr_channels or out_channels,
                ],
                plain_last=True,
                **(kwargs | (edge_mlp_kwargs or {})),
            ),
            global_nn=None,
            edge_scaler=edge_scaler,
        )
        self.node_nn = MLP(
            channel_list=[
                self.in_channels,
                *hidden_channels,
                aggr_channels or out_channels,
            ],
            plain_last=True,
            **(kwargs | (node_mlp_kwargs or {})),
        )
        self.post_nn = (
            MLP(
                channel_list=[aggr_channels, self.out_channels],
                plain_last=False,
                layout='ln',
                **kwargs,
            )
            if aggr_channels
            else normalization_resolver(
                kwargs['norm'],
                in_channels=out_channels,
                **(dict(init_weight=init_weight) | (kwargs.get('norm_kwargs') or {})),
            )
            if kwargs.get('norm')
            else None
        )
        self.out_channels = out_channels

    def forward(
        self,
        x: OptTensor | PairOptTensor,
        pos: Tensor | PairTensor,
        edge_index: Adj,
    ) -> Tensor:
        node_out = self.node_nn(x)
        edge_out = self.edge_nn(x=None, pos=pos, edge_index=edge_index)
        return self.post_nn(node_out + edge_out)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('  #
            f'node_nn={self.node_nn}, '
            f'edge_nn={self.edge_nn}, '
            f'post_nn={self.post_nn}'
            ')'
        )


class RelativePositionRegularizer(Module):
    return_names = ('rel_pos_loss',)

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        edge_scaler: Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.out_channels = self.in_channels = in_channels
        rel_channels = getattr(edge_scaler, 'out_channels', 3)
        self.mlp = MLP(
            [in_channels, 32, rel_channels],
            plain_last=True,
            bias=False,
            **kwargs,
        )
        self.scaler = edge_scaler

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj) -> Tensor:
        num_nodes = pos.size(0)

        # Select some random edges
        if isinstance(edge_index, SparseTensor):
            # Sparse edge matrix
            row, col, _ = edge_index.coo()
            coo_edge_index = torch.stack([row, col], dim=0)
            chosen_edge_ids = torch.randperm(coo_edge_index.shape[-1])[:num_nodes]
            chosen_edges = coo_edge_index[:, chosen_edge_ids].contiguous()
        elif edge_index.size(0) == 2:
            # Pairwise (PyG-style) edges
            chosen_edge_ids = torch.randperm(edge_index.shape[-1])[:num_nodes]
            chosen_edges = edge_index[:, chosen_edge_ids].contiguous()
        else:
            # Source-indexed edges
            chosen_edges_per_node = torch.randint(
                0, edge_index.size(-1), size=[num_nodes, 1], dtype=edge_index.dtype, device=edge_index.device
            )
            chosen_edges = torch.stack(
                [
                    torch.gather(edge_index, 1, chosen_edges_per_node).flatten(),
                    torch.arange(num_nodes, device=edge_index.device, dtype=edge_index.dtype),
                ]
            )

        # Compute true vectors
        true_vectors = pos[chosen_edges[1], :] - pos[chosen_edges[0], :]
        if self.scaler:
            # Don't encourage the network to learn an easier encoding!
            with torch.no_grad():
                true_vectors = self.scaler(true_vectors).detach()

        # Compute predicted vectors from node feature pairs
        rel_features = x[chosen_edges[1], :] - x[chosen_edges[0], :]
        pred_vectors = self.mlp(rel_features)

        # Compute loss
        return torch.nn.functional.mse_loss(pred_vectors, true_vectors)

    def extra_repr(self) -> str:
        return f"{self.in_channels} -> rel_pos_loss"
