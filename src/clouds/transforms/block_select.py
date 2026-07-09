import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from clouds.transforms.apply_selection import apply_selection


class BlockSelect(BaseTransform):
    def __init__(
        self,
        max_num_points: int = int(1e9),
        min_num_points: int = 1,
        selection_factor: float = 1.0,
    ):
        super().__init__()
        self.max_num_points, self.min_num_points = max_num_points, min_num_points
        self.selection_factor = selection_factor

    def forward(self, data: Data) -> Data:
        if isinstance(data.batch, Tensor):
            assert isinstance(data.ptr, Tensor)
            batch_sizes = data.ptr[1:] - data.ptr[:-1]
            selection_sizes = torch.clamp(  # TODO: prefer clamp
                batch_sizes.float() * self.selection_factor,
                min=self.min_num_points,
                max=self.max_num_points,
            ).long()
            data.selection_index = torch.cat(
                [
                    torch.arange(offset, offset + count, device=data.pos.device)
                    for offset, count in zip(data.ptr[:-1], selection_sizes, strict=True)
                ]
            )
        else:
            selection_size = max(min(int(data.num_nodes * self.selection_factor), self.max_num_points), self.min_num_points)
            data.selection_index = torch.arange(selection_size, device=data.pos.device)
        return data

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(*{self.selection_factor}, <{self.max_num_points})"


class BlockSample(BlockSelect):
    def forward(self, data: Data) -> Data:
        return apply_selection(super(data))
