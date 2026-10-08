from torch import Tensor
from torch.nn import Module
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj, OptPairTensor, PairTensor, Size

from ..apply import apply_to_kwargs
from ..models.mlp import MLP
from .pointnet import PointNetConv, SimplePointNetConv
from .selection import ProjSelectionConv


class SetAbstraction(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.res = SimplePointNetConv(
            in_channels,
            out_channels,
            hidden_factor=1 / 2,
            **kwargs,
        )
        self.id = ProjSelectionConv(
            in_channels,
            out_channels,
            **(kwargs | dict(norm=None)),
        )
        self.act = activation_resolver(kwargs.get('act'), **(kwargs.get('act_kwargs') or {}))
        self.in_channels, self.out_channels = in_channels, out_channels

    def forward(
        self,
        x: Tensor | PairTensor,
        pos: Tensor | PairTensor,
        edge_index: Adj,
        selection_index: OptPairTensor = None,
        size: Size = None,
        **kwargs,
    ) -> Tensor:
        res = apply_to_kwargs(self.res, x=x, pos=pos, edge_index=edge_index, selection_index=selection_index, size=size)
        id = apply_to_kwargs(self.id, x=x, pos=pos, selection_index=selection_index, size=size)
        # Interesting that PointNeXt applies activation after merging res stream!
        return self.act(id + res)


class InvResConv(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.res = PointNetConv(
            local_nn=MLP(
                in_channels=in_channels + 3,
                out_channels=out_channels,
                **kwargs,
            ),
            global_nn=MLP(
                in_channels=out_channels,
                out_channels=out_channels,
                hidden_factor=4,
                num_layers=2,
                plain_last=True,
                **kwargs,
            ),
            # FIXME: this should use PointNetConv with a global (post-pooling) MLP!
        )
        self.act = activation_resolver(kwargs.get('act'), **(kwargs.get('act_kwargs') or {}))
        self.in_channels, self.out_channels = in_channels, out_channels

    def forward(
        self,
        x: Tensor | PairTensor,
        pos: Tensor | PairTensor,
        edge_index: Adj,
        selection_index: OptPairTensor = None,
        size: Size = None,
    ) -> Tensor:
        res = apply_to_kwargs(self.res, x=x, pos=pos, edge_index=edge_index, selection_index=selection_index, size=size)
        return self.act(x + res)
