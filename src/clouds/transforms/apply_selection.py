import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


def select_knn_edges(edge_index: Tensor, selection_index: Tensor) -> Tensor:
    """Remap a (N, K) kNN edge-index tensor after dropping unselected nodes.

    Args:
        edge_index: (N, K) tensor of neighbor indices for each of the N nodes.
        selection_index: Boolean mask of shape (N,) or a sorted integer index
            tensor identifying which nodes are kept.

    Returns:
        (M, K) tensor of remapped neighbor indices, where M is the number of
        kept nodes. Neighbors that were dropped are replaced with a self-edge
        (i.e. the node points to itself).
    """
    num_nodes = edge_index.size(0)
    num_kept = selection_index.sum() if selection_index.dtype == torch.bool else selection_index.size(0)

    # Map old node index -> new node index, -1 if the node was dropped.
    old_to_new = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_index.device)
    old_to_new[selection_index] = torch.arange(num_kept, dtype=torch.long, device=edge_index.device)

    # Keep only the rows for selected nodes, then remap their neighbor indices.
    kept_rows = edge_index[selection_index]  # (M, K)
    remapped = old_to_new[kept_rows]  # (M, K), -1 where neighbor was dropped

    # Replace dropped neighbors (-1) with a self-edge.
    self_idx = torch.arange(remapped.size(0), dtype=torch.long, device=edge_index.device).unsqueeze(1)
    remapped = torch.where(remapped == -1, self_idx.expand_as(remapped), remapped)

    return remapped


class ApplySelection(BaseTransform):
    """Restrict a `Data` object to a subset of nodes given by `selection_index`.

    Node-level attributes are filtered down to the selected nodes. Edge-level
    attributes and index/cluster bookkeeping fields are dropped, since they no
    longer make sense after selection. Optionally, `edge_index` can be
    remapped (rather than dropped) via `select_knn_edges`, assuming it is
    stored in dense (N, K) kNN format.
    """

    def forward(self, data: Data) -> Data:
        selection_index = data.get('selection_index')
        if not isinstance(selection_index, Tensor):
            return data
        if selection_index.size(0) == 0:
            raise ValueError("Empty selection!")

        num_nodes = data.num_nodes
        out = Data()

        for key, item in data.items():
            if key == 'edge_index':
                assert item.size(0) == num_nodes  # Only works on kNN-formatted edges
                out[key] = select_knn_edges(item, selection_index=selection_index)
            elif 'cluster' in key or 'index' in key:
                # Drop stale index/cluster bookkeeping fields
                pass
            elif data.is_node_attr(key) and item.size(0) == num_nodes:
                out[key] = item[selection_index]
                if 'pos' not in data:
                    out.num_nodes = out[key].size(0)
            elif data.is_edge_attr(key) or key in ('ptr',):
                # Drop invalidated edge attributes
                # TODO: handle this correctly
                pass
            else:
                out[key] = item

        if isinstance(out.batch, Tensor):
            # NOTE: assumes batches are contiguous.
            counts = torch.bincount(out.batch, minlength=data.batch_size)
            out.ptr = torch.cat([torch.tensor([0], device=out.batch.device), torch.cumsum(counts, dim=0)])

        del out.selection_index
        return out


apply_selection = ApplySelection()
