import string

from torch.nn import Module, ModuleDict

from ...data import MultiGridData
from ..merge import IdentityMerge, SumMerge
from .multigrid import MultiGridModule


class MultiGridParallel(ModuleDict):
    def __init__(
        self,
        *modules: Module | tuple[str, Module],
        merge: type = SumMerge,
        merge_kwargs: dict | None = None,
    ) -> None:
        super().__init__()

        # Add modules
        for i, m in enumerate(modules):
            if isinstance(m, Module):
                self[string.ascii_letters[i]] = m
            else:
                name, m = m
                self[name] = m

        # Determine input channels
        self.in_channels: list[int] = next(iter(self.values())).in_channels
        for m in self.values():
            assert m.in_channels == self.in_channels

        # Determine module output channels
        out_channels = [m.out_channels for m in self.values()]

        # Construct mergers
        merge_in_channels = [
            [c for c in level_channels if c]  #
            for level_channels in zip(*out_channels, strict=True)
        ]
        self.merge = MultiGridModule(
            [merge if len(l_c) > 1 else IdentityMerge for l_c in merge_in_channels],
            in_channels=merge_in_channels,
            **(merge_kwargs or {}),
        )

        # Determine output channels (based on merges)
        self.out_channels = self.merge.out_channels

    def forward(self, data: MultiGridData) -> MultiGridData:
        # NOTE: this trusts that no modules will modify data!
        modules = list(self.values())[:-1]
        outputs = [module(data) for module in modules]

        # NOTE: this trusts that every scale has a merge (this should be fine?)
        for level, merge in self.merge.items():
            level_outs = [out[level].x for out in outputs if hasattr(out[level], 'x')]
            data[level].x = merge(*level_outs)

        return data
