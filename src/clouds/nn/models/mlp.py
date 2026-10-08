import itertools
from collections.abc import Callable
from typing import Any

from torch.nn import Dropout
from torch.utils.checkpoint import checkpoint

from ...utils.sequence import _normalize_list
from ..dense.linear import Linear
from ..resolver import activation_resolver, normalization_resolver
from ..sequential import Sequential


class MLP(Sequential):
    def __init__(
        self,
        # Explicit construction
        channel_list: list[int] | None = None,
        # Implicit construction
        in_channels: int | None = None,
        hidden_channels: int | None = None,
        out_channels: int | None = None,
        out_factor: float | None = None,
        hidden_factor: float | None = None,
        num_layers: int | None = None,
        # Layout
        layout: str = 'LinNormActDrop',
        plain_last: bool = False,
        # Other properties
        bias: bool | list[bool] = False,
        weight_initializer: str | None = None,
        bias_initializer: str | None = None,
        act: str | Callable | type | None = 'relu',
        act_kwargs: dict[str, Any] | None = None,
        norm: str | Callable | type | None = 'batch',
        norm_kwargs: dict[str, Any] | None = None,
        dropout: type | None = Dropout,
        dropout_p: float | list[float] = 0,
        dropout_kwargs: dict[str, Any] | None = None,
        grad_checkpoint: bool = False,
        **kwargs,
    ) -> None:
        self.grad_checkpoint = grad_checkpoint

        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        dropout_kwargs = dropout_kwargs or {}
        num_layers = num_layers or (len(channel_list) if channel_list else 1)

        # Determine channel counts, if not provided explicitly
        if not channel_list:
            assert isinstance(in_channels, int)
            if out_channels is None and isinstance(out_factor, (float, int)):
                out_channels = int(in_channels * out_factor)
            assert isinstance(out_channels, int)
            if hidden_channels is None and isinstance(hidden_factor, (float, int)):
                hidden_channels = int(out_channels * hidden_factor)
            assert num_layers == 1 or isinstance(hidden_channels, int)

            assert isinstance(num_layers, int)
            channel_list = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]

        # Determine layout, if not provided explicitly
        if any(c.isupper() for c in layout):
            layout = ''.join([c.lower() for c in layout if c.isupper()])
        if num_lins := layout.count('l') != len(channel_list) - 1:
            assert num_lins == 1
            layout *= len(channel_list) - 1
        if plain_last:
            # only dropout is allowed after the last linear layer
            blocks = layout.split('l')
            blocks[-1] = 'd' if 'd' in blocks[-1] else ''
            layout = 'l'.join(blocks)

        # Expand other parameters
        dropout_p = list(_normalize_list(dropout_p, layout.count('d')))
        if not isinstance(dropout_kwargs, list):
            dropout_kwargs = [dropout_kwargs] * layout.count('d')
        bias = list(_normalize_list(bias, layout.count('l')))
        bias_initializer = list(_normalize_list(bias_initializer, layout.count('l')))
        weight_initializer = list(_normalize_list(weight_initializer, layout.count('l')))
        act = list(_normalize_list(act, layout.count('a')))
        if not isinstance(act_kwargs, list):
            act_kwargs = [act_kwargs] * layout.count('a')
        norm = list(_normalize_list(norm, layout.count('n')))
        if not isinstance(norm_kwargs, list):
            norm_kwargs = [norm_kwargs] * layout.count('n')

        assert len(norm) == len(norm_kwargs)

        # Construct each element of the layout
        channel_pairs = list(itertools.pairwise([*channel_list, None]))
        in_c, out_c = channel_pairs.pop(0)
        modules = []
        for m in layout:
            if m == 'l':
                modules.append(
                    Linear(
                        in_channels=in_c,
                        out_channels=out_c,
                        bias=bias.pop(0),
                        weight_initializer=weight_initializer.pop(0),
                        bias_initializer=bias_initializer.pop(0),
                    )
                )
                in_c, out_c = channel_pairs.pop(0)
            elif m == 'a':
                if a := act.pop(0):
                    a_kwargs = act_kwargs.pop(0)
                    modules.append(
                        a(**a_kwargs)  #
                        if isinstance(act, type)
                        else activation_resolver(a, **a_kwargs)
                    )
            elif m == 'n':
                if n := norm.pop(0):
                    n_kwargs = norm_kwargs.pop(0)
                    modules.append(
                        n(in_channels=in_c, **n_kwargs)
                        if isinstance(norm, type)
                        else normalization_resolver(n, in_channels=in_c, **n_kwargs)
                    )
            elif m == 'd':
                if p := dropout_p.pop(0):
                    modules.append(dropout(p=p, **dropout_kwargs.pop(0)))
            else:
                raise ValueError(f"Unrecognized block type '{m}'!")

        super().__init__(*modules, rewrite={'input': 'x'})

        # Modules like Dropout don't provide out_channels, so this hack is sometimes necessary
        if self.out_channels is None:
            self.out_channels = self[-2].out_channels

    def forward(self, *args) -> Any:
        if self.grad_checkpoint:
            return checkpoint(super().forward, *args, use_reentrant=False)
        else:
            return super().forward(*args)
