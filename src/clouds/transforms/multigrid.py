from collections.abc import Callable, Iterable

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from ..data import MultiGridData
from ..utils import construct_multigrid_objects
from .apply_selection import ApplySelection


class MultiGridTransform(BaseTransform):
    """Apply a (per-level) transform to every level of a :class:`MultiGridData`.

    ``transform`` may be a single ``BaseTransform`` (shared across all levels), a
    list of transforms (one per level), or a class/callable invoked once per level
    with the level-specific ``kwargs`` (see :func:`construct_multigrid_objects`).
    """

    def __init__(self, transform: Callable | list[Callable], levels: Iterable[int] | None = None, **kwargs) -> None:
        super().__init__()
        modules = construct_multigrid_objects(expected_type=BaseTransform, factory=transform, levels=levels, **kwargs)
        assert modules
        self.modules = {f"x{s}": m for s, m in modules.items()}

    def forward(self, data: MultiGridData) -> MultiGridData:
        for level, transform in self.modules.items():
            data.set_level(level, transform(data.get_level(level)))

        return data

    def __repr__(self) -> str:
        lines = [f"  ({level}): {transform!r}" for level, transform in self.modules.items()]
        body = "\n".join(lines)
        return f"{self.__class__.__name__}(\n{body}\n)"


class BuildSelectionMultiGrid(BaseTransform):
    """Build multigrid levels by repeatedly selecting a coarser subset of points.

    Each selector follows the ``*Select`` protocol (it writes
    ``data.selection_index``); the coarser level is produced by applying
    :class:`ApplySelection`.
    """

    def __init__(self, selectors: list[BaseTransform]) -> None:
        super().__init__()
        self.selectors = selectors

    def forward(self, data: Data) -> MultiGridData:
        # Copy the first level(s) over
        if isinstance(data, MultiGridData):
            mg_data = data
        else:
            # todo: maybe to_heterogeneous() can be used here?
            mg_data = MultiGridData()
            mg_data.set_level('x0', data)

        # Build levels one by one
        for i, selector in enumerate(self.selectors):
            fine, coarse = f'x{i}', f'x{i + 1}'

            # If a scale already exists, there's no need to create it
            if mg_data[coarse]:
                continue

            # Assign parents in the finer level
            fine_data = mg_data.get_level(fine)
            fine_data = selector(fine_data)

            # Produce the coarser level
            coarse_data = ApplySelection()(fine_data)

            # Write back both levels
            mg_data.set_level(fine, fine_data)
            mg_data.set_level(coarse, coarse_data)

        return mg_data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.selectors!r})"


class BuildClusterMultiGrid(BaseTransform):
    """Build multigrid levels by clustering the finer level and coarsening it.

    Pairs of ``(clusterer, coarsener)`` are applied level by level: the clusterer
    assigns ``data.cluster_index`` on the finer level, and the coarsener (e.g.
    :class:`ClusterSelect` + :class:`ApplySelection`) produces the coarse level.
    """

    def __init__(self, clusterers: list[BaseTransform], coarseners: list[BaseTransform]) -> None:
        super().__init__()
        self.clusterers, self.coarseners = clusterers, coarseners

    def forward(self, data: Data) -> MultiGridData:
        # Copy the first level(s) over
        if isinstance(data, MultiGridData):
            mg_data = data
        else:
            # todo: maybe to_heterogeneous() can be used here?
            mg_data = MultiGridData()
            mg_data.set_level('x0', data)

        # Build levels one by one
        for i, (cluster, coarsen) in enumerate(zip(self.clusterers, self.coarseners, strict=True)):
            fine, coarse = f'x{i}', f'x{i + 1}'

            # If a scale already exists, there's no need to create it
            if mg_data[coarse]:
                continue

            # Assign clusters in the finer level
            fine_data = mg_data.get_level(fine)
            fine_data = cluster(fine_data)

            # Produce the coarser level
            coarse_data = coarsen(fine_data)

            # Write back both levels
            mg_data.set_level(fine, fine_data)
            mg_data.set_level(coarse, coarse_data)

        return mg_data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(clusterers={self.clusterers!r}, coarseners={self.coarseners!r})"


class BuildPhantomMultigrid(BaseTransform):
    """Wrap a single-level ``Data`` into a two-level :class:`MultiGridData`.

    The fine level ``x0`` is an empty placeholder; the data lands on the coarse
    level ``x1``.
    """

    def forward(self, data: Data) -> MultiGridData:
        mg_data = MultiGridData()
        mg_data.set_level('x0', Data())
        mg_data.set_level('x1', data)
        return mg_data


class FlattenMultigrid(BaseTransform):
    """Collapse a :class:`MultiGridData` into a single flat ``Data`` object.

    Node attributes are concatenated across levels (records each level's node
    ``offset``), a ``level`` column tags each node with its source level, and the
    inter-level edges are stitched into a single pairwise ``(2, E)`` ``edge_index``.
    """

    def forward(self, data: MultiGridData | Data) -> Data:
        if not hasattr(data, 'get_level'):
            return data

        out = None
        for level in data.node_types:
            level_data = data.get_level(level)
            if out is not None:
                if not hasattr(out, 'level') and not hasattr(level_data, 'level'):
                    out.level = torch.zeros(out.num_nodes, 1)

                # Append data
                offset = data[level].offset = out.num_nodes
                for key, item in level_data.items():
                    if 'cluster' in key or 'index' in key:
                        pass
                    elif level_data.is_node_attr(key) and (key in out or not out.num_nodes):
                        if hasattr(out, key) and isinstance(getattr(out, key, None), Tensor):
                            item = torch.cat([out[key], item], dim=0)
                        out[key] = item
                        if 'pos' not in out and key != 'level':
                            out.num_nodes = out[key].size(0)
                    elif level_data.is_edge_attr(key) or key in ['ptr']:
                        pass
                    else:
                        out[key] = item

                if 'x' not in level_data:
                    # FIXME: this is a hack!
                    del out.x

                # Pad levels
                if out.level.size(0) == offset:
                    next_level = out.level[-1, 0] + 1
                    out.level = torch.cat([out.level, next_level * torch.ones(out.num_nodes - out.level.size(0), 1)], dim=0)

            elif getattr(level_data, 'num_nodes', None):
                data[level].offset = 0
                out = level_data

        # Add edges between levels
        assert isinstance(out, Data)
        out.edge_index = torch.zeros(2, 0, dtype=torch.long)
        for (source_level, _, dest_level), edge_store in data.edge_items():
            if not hasattr(edge_store, 'edge_index'):
                continue
            assert isinstance(edge_store.edge_index, Tensor)
            edge_index = edge_store.edge_index
            if edge_index.size(0) != 2:
                edge_index = torch.stack(
                    [
                        edge_index.flatten(),
                        torch.arange(edge_index.size(0), dtype=edge_index.dtype, device=edge_index.device).repeat_interleave(
                            edge_index.size(1)
                        ),
                    ]
                )

            edge_index[0] += data[source_level].offset
            edge_index[1] += data[dest_level].offset

            assert isinstance(out.edge_index, Tensor)
            out.edge_index = torch.cat([out.edge_index, edge_index], dim=-1)

        return out
