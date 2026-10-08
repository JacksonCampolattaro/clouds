from collections.abc import Callable, Iterable

from torch.nn import Module, ModuleDict

from ...data import MultiGridData
from ...utils import construct_multigrid_modules
from ..apply import apply_to_data


class MultiGridModule(ModuleDict):
    def __init__(
        self,
        module: Module | Callable | list[Callable],
        in_channels: list[int | None],
        levels: Iterable[int] | None = None,
        **kwargs,
    ) -> None:
        assert module
        modules = construct_multigrid_modules(factory=module, levels=levels, in_channels=in_channels, **kwargs)
        super().__init__({f"x{s}": m for s, m in modules.items()})
        self.in_channels = in_channels
        self.out_channels = [
            getattr(self[f'x{s}'], 'out_channels', None) if hasattr(self, f'x{s}') else None  #
            for s in range(len(in_channels))
        ]

    def forward(self, data: MultiGridData) -> MultiGridData:
        for level, module in self.items():
            data.set_level(level, apply_to_data(module, data.get_level(level)))
        return data
