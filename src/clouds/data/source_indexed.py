from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.data.data import Index


class SourceIndexedData(Data):
    def __cat_dim__(self, key: str, value: Any, *args, **kwargs):
        if key == 'edge_index' and value.size(0) != 2:
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


def unpack_source_indexed_data(data: Data) -> Data:
    data.edge_index = Index(
        torch.stack(
            [
                data.edge_index.flatten(),
                torch.arange(
                    data.edge_index.size(0),
                    dtype=data.edge_index.dtype,
                    device=data.edge_index.device,
                ).repeat_interleave(data.edge_index.size(1)),
            ]
        )
    )
    # TODO: handle edge attributes
    return data
