from typing import Any

from torch.nn import Module, ModuleList
from torch_geometric.data import Data

from .apply import get_param_names


class Sequential(ModuleList):
    """A name-aware container that chains modules by their ``forward`` signatures."""

    def __init__(self, *modules: Module, rewrite: dict[str, str] | None = None) -> None:
        super().__init__(modules)
        self.rewrite = rewrite or {}

        self.param_names, self.return_names = [], []
        self.child_signatures = []
        for module in self:
            module_param_names = get_param_names(module, rewrite=rewrite)
            module_return_names = getattr(module, 'return_names', ['x'])
            self.child_signatures.append((module_param_names, module_return_names))

            self.param_names.extend(name for name in module_param_names if name not in self.param_names)
            self.return_names.extend(name for name in module_return_names if name not in self.return_names)

            if not hasattr(self, 'in_channels') and hasattr(module, 'in_channels'):
                self.in_channels = module.in_channels
            if hasattr(module, 'out_channels'):
                self.out_channels = module.out_channels

    def forward(self, *args) -> Any:
        value_dict = dict(zip(self.param_names, args, strict=True))
        for module, (param_names, return_names) in zip(self, self.child_signatures, strict=True):
            outs = module(*[value_dict[name] for name in param_names])

            if len(return_names) == 1:
                value_dict[return_names[0]] = outs[0] if isinstance(outs, tuple) else outs
            else:
                for name, out in zip(return_names, outs, strict=True):
                    value_dict[name] = out

        if len(value_dict) == 1:
            return next(iter(value_dict.values()))
        elif len(self.return_names) == 1:
            return value_dict[return_names[0]]
        else:
            return tuple(value_dict[name] for name in self.return_names)


class DataSequential(ModuleList):
    """A container that applies each module to a ``Data`` object in turn."""

    def __init__(self, *modules: Module) -> None:
        super().__init__(modules)
        self.in_channels, self.out_channels = modules[0].in_channels, modules[-1].out_channels

    def forward(self, data: Data) -> Data:
        for m in self:
            data = m(data)
        return data
