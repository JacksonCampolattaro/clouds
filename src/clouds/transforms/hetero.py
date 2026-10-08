from collections.abc import Iterator

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import EdgeStorage, NodeStorage


def require_homogeneous(data: Data, transform: str) -> None:
    """Raise if ``data`` is heterogeneous (e.g. a ``MultiGridData``).

    Transforms that need cross-level coordination are not implemented for
    multigrids; the supported alternative is to apply them per level with
    :class:`MultiGridTransform`.
    """
    if isinstance(data, HeteroData):
        raise NotImplementedError(
            f"{transform} does not support heterogeneous (multi-level) data; apply it per level with MultiGridTransform instead"
        )


def intra_level_stores(data: Data) -> Iterator[tuple[NodeStorage, EdgeStorage]]:
    """Yield ``(node_store, edge_store)`` pairs for every level of ``data``.

    A flat ``Data`` yields a single pair. A ``MultiGridData`` (or any
    ``HeteroData``) yields one pair per node type, with the intra-level edges
    stored under the ``(type, 'to', type)`` edge type.
    """
    if isinstance(data, HeteroData):
        for node_type in data.node_types:
            yield data[node_type], data[node_type, 'to', node_type]
    else:
        yield data.node_stores[0], data.edge_stores[0]
