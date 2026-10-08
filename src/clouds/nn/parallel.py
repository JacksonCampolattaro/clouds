import string
from typing import Any

from torch.nn import Module, ModuleDict

from .apply import get_param_names
from .merge import SumMerge


class Parallel(ModuleDict):
    """A container that runs several branches and merges their outputs."""

    def __init__(
        self,
        *modules: Module | tuple[str, Module],
        rewrite: dict[str, str] | None = None,
        merge: type = SumMerge,
    ) -> None:
        module_dict = dict()
        for i, m in enumerate(modules):
            if isinstance(m, Module):
                module_dict[string.ascii_letters[i]] = m
            else:
                name, m = m
                module_dict[name] = m

        super().__init__(module_dict)

        self.param_names, self.return_names = [], []
        self.child_signatures = []
        module_in_channels, module_out_channels = [], []
        for module in self.values():
            module_param_names = get_param_names(module, rewrite=rewrite)
            module_return_names = getattr(module, 'return_names', ['x'])
            self.child_signatures.append((module_param_names, module_return_names))

            self.param_names.extend(name for name in module_param_names if name not in self.param_names)
            self.return_names.extend(name for name in module_return_names if name not in self.return_names)

            module_in_channels.append(getattr(module, 'in_channels', None))
            module_out_channels.append(getattr(module, 'out_channels', None))

        self.in_channels = next(c for c in module_in_channels if c is not None)
        self.merge = merge(in_channels=module_out_channels)  # todo: use module dict!
        self.out_channels = self.merge.out_channels

    def forward(self, *args) -> Any:
        value_dict = dict(zip(self.param_names, args, strict=True))
        modules = list(self.values())[:-1]
        module_args = [[value_dict[name] for name in param_names] for param_names, _ in self.child_signatures]
        outputs = [module(*m_args) for module, m_args in zip(modules, module_args, strict=True)]
        return self.merge(*outputs)
