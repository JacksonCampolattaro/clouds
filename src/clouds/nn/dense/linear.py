import math

import torch
from torch import Tensor
from torch_geometric.nn import Linear as PyGLinear
from torch_geometric.nn.dense.linear import reset_bias_ as pyg_reset_bias_
from torch_geometric.nn.dense.linear import reset_weight_ as pyg_reset_weight_


def reset_weight_(weight: Tensor, in_channels: int, initializer: str | None = None) -> Tensor:
    if initializer == 'trunc_normal':
        torch.nn.init.trunc_normal_(weight, std=0.02)
    elif initializer == 'finer0':
        torch.nn.init.uniform_(
            weight,
            -1 / weight.size(-1),
            1 / weight.size(-1),
        )
    elif initializer == 'finer':
        # FIXME: how to specify omega in config?
        omega_0 = 1
        torch.nn.init.uniform_(
            weight,
            -math.sqrt(6 / weight.size(-1)) / omega_0,
            math.sqrt(6 / weight.size(-1)) / omega_0,
        )
    else:
        return pyg_reset_weight_(weight, in_channels, initializer=initializer)

    return weight


def reset_bias_(
    bias: Tensor | None,
    in_channels: int,
    initializer: str | None = None,
) -> Tensor | None:
    if bias is None or in_channels <= 0:
        pass
    elif initializer and initializer.startswith('uniformk_'):
        k = float(initializer.lstrip('uniformk_'))
        torch.nn.init.uniform_(bias, -k, k)
    else:
        return pyg_reset_bias_(bias, in_channels)

    return bias


class Linear(PyGLinear):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        weight_initializer: str | None = None,
        bias_initializer: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            bias=bias,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )

    def reset_parameters(self) -> None:
        reset_weight_(self.weight, self.in_channels, self.weight_initializer)
        reset_bias_(self.bias, self.in_channels, self.bias_initializer)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}({self.in_channels}, '  #
            f'{self.out_channels}, bias={self.bias is not None} '  #
            f'init={self.weight_initializer}'  #
            + (f'+{self.bias_initializer}' if self.bias is not None else '')  #
            + ')'
        )
