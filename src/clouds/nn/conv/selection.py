from torch import Tensor
from torch.nn import Module
from torch_geometric.typing import OptPairTensor, PairOptTensor, PairTensor, Size

from ..models.mlp import MLP


class SelectionConv(Module):
    def __init__(self, in_channels: int, out_channels: int | None = None, **kwargs) -> None:
        assert out_channels in (in_channels, None)
        super().__init__()
        self.in_channels = self.out_channels = in_channels

    def forward(self, x: Tensor | PairTensor, selection_index: PairOptTensor = None, size: Size = None) -> Tensor:
        x = x if not isinstance(x, tuple) else x[0]
        selection_index = selection_index if not isinstance(selection_index, tuple) else selection_index[0]
        return x[selection_index, :]


class ProjSelectionConv(Module):
    def __init__(self, in_channels: int, out_channels: int | None = None, **kwargs) -> None:
        self.in_channels, self.out_channels = in_channels, out_channels or in_channels
        super().__init__()
        # TODO: norm?
        self.proj = MLP([self.in_channels, self.out_channels], layout='ln', **kwargs)

    def forward(
        self,
        x: Tensor | PairTensor,
        selection_index: PairOptTensor = None,
        batch: OptPairTensor = None,
        size: Size = None,
    ) -> Tensor:
        x = x if not isinstance(x, tuple) else x[0]
        selection_index = selection_index if not isinstance(selection_index, tuple) else selection_index[0]
        if size and size[0] < size[1]:
            # if upsampling, perform the proj first
            return self.proj(x)[selection_index, :]
        else:
            return self.proj(x[selection_index, :])
