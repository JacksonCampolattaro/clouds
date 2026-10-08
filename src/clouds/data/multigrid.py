from collections.abc import Iterable
from typing import Any

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.hetero_data import NodeOrEdgeStorage
from torch_geometric.data.storage import EdgeStorage, NodeStorage


class MultiGridData(HeteroData):
    """A heterogeneous ``Data`` object whose node types are the levels of a multigrid.

    Each level is stored as a node type (by convention ``x0``, ``x1``, ...), with
    intra-level edges stored under the ``(level, 'to', level)`` edge type. Attributes
    which are neither node- nor edge-level are kept as shared globals and copied onto
    every level by :meth:`get_level`.
    """

    def _level_name(self, level: str | int) -> str:
        return self.node_types[level] if isinstance(level, int) else level

    def get_globals(self) -> dict:
        return {k: v for k, v in self._global_store.items()}

    def get_level(self, level: str | int) -> Data:
        level = self._level_name(level)
        return Data(**self[level], **self[level, 'to', level], **self.get_globals())

    def set_level(self, level: str | int, data: Data) -> None:
        level = self._level_name(level)

        # Clear old values
        for key in list(self[level]):
            self[level][key] = None

        # Set new values
        for key, item in data.items():
            if data.is_node_attr(key):
                self[level][key] = item
            elif data.is_edge_attr(key) and key not in ['y', 'pred']:
                # TODO: is_edge_attr breaks when K == batch size!
                # checking for keys is a poor workaround
                self[level, 'to', level][key] = item
            elif key in ['num_nodes'] and hasattr(data, 'pos'):
                pass
            elif key in ['face', 'ptr', 'selection_index', 'num_nodes']:
                # Faces always connect nodes of the same scale
                # for now, it's simplest to just treat them as a node attribute
                self[level][key] = item
            elif key.endswith('_loss'):
                # Losses are confined to their scale, and get accumulated!
                self[level][key] = (self[level][key] + item) if hasattr(self[level], key) else item
            else:
                # Global attributes
                self[key] = item

    def get_levels(self, levels: Iterable[str | int], rename: bool = False) -> 'MultiGridData':
        keep = {self._level_name(level) for level in levels}
        unknown = keep - set(self.node_types)
        if unknown:
            raise KeyError(f'Unknown level(s): {sorted(unknown)}')

        # Preserve the original level ordering
        kept = [name for name in self.node_types if name in keep]
        names = {name: (f'x{i}' if rename else name) for i, name in enumerate(kept)}

        out = MultiGridData()
        out._global_store.update(self.get_globals())

        for name in kept:
            for key, value in self[name].items():
                out[names[name]][key] = value

        # Keep every edge type whose endpoints are both retained
        for (src, rel, dst), store in self.edge_items():
            if src in keep and dst in keep:
                for key, value in store.items():
                    out[names[src], rel, names[dst]][key] = value

        return out

    def __inc__(self, key: str, value: Any, store: NodeOrEdgeStorage | None = None, *args, **kwargs) -> Any:
        if isinstance(store, EdgeStorage) and 'index' in key and value.size(0) != 2:
            return value.max() + 1
        elif isinstance(store, NodeStorage) and key == 'cluster_index':
            return store.selection_index.size(0)
        elif isinstance(store, NodeStorage) and 'index' in key:
            return store.num_nodes  # FIXME: breaks on cluster!
        return super().__inc__(key, value, store, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, store: NodeOrEdgeStorage | None = None, *args, **kwargs) -> Any:
        if isinstance(store, EdgeStorage) and 'index' in key and value.size(0) != 2:
            return 0
        return super().__cat_dim__(key, value, store, *args, **kwargs)
