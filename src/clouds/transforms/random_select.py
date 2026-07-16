import itertools
import random

import numpy
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from clouds.transforms.apply_selection import apply_selection


class RandomSelect(BaseTransform):
    def __init__(
        self,
        max_num_points: int = int(1e7),
        selection_factor: float | tuple[float, float] = 1.0,
        min_num_points: int = 1,
        replacement: bool = False,
    ):
        super().__init__()
        self.max_num_points = max_num_points
        self.min_num_points = min_num_points
        self.selection_factor = selection_factor
        self.replacement = replacement

    def forward(self, data: Data) -> Data:
        def _selection_size(n: int):
            selection_factor = (
                random.uniform(*self.selection_factor)  #
                if isinstance(self.selection_factor, tuple)
                else self.selection_factor
            )
            return numpy.clip(int(n * selection_factor), min=self.min_num_points, max=self.max_num_points)

        if self.replacement:
            data.selection_index = (
                torch.cat(
                    [
                        torch.randint(
                            start,
                            end,
                            (_selection_size(end - start),),
                            dtype=torch.long,
                            device=data.pos.device,
                        )
                        for start, end in itertools.pairwise(data.ptr)
                    ],
                    dim=0,
                )
                if isinstance(data.batch, Tensor)
                else torch.randint(
                    0,
                    data.num_nodes,
                    (_selection_size(data.num_nodes),),
                    dtype=torch.long,
                    device=data.pos.device,
                )
            )
        else:
            data.selection_index = (
                torch.cat(
                    [
                        torch.randperm(
                            end - start,
                            dtype=torch.long,
                            device=data.pos.device,
                        )[: _selection_size(end - start)]
                        + start
                        for start, end in itertools.pairwise(data.ptr)
                    ],
                    dim=0,
                )
                if isinstance(data.batch, Tensor)
                else torch.randperm(
                    data.num_nodes,
                    dtype=torch.long,
                    device=data.pos.device,
                )[: _selection_size(data.num_nodes)]
            )

        data.selection_index = data.selection_index.sort()[0]

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(*{self.selection_factor}, <{self.max_num_points}, replace={self.replacement})"


class RandomSample(RandomSelect):
    def forward(self, data: Data) -> Data:
        return apply_selection(super().forward(data))


class RandomPointDropout(BaseTransform):
    def __init__(self, max_dropout=0.9, p: float = 1.0):
        super().__init__()
        self.max_dropout = max_dropout
        self.p = p

    def forward(self, data: Data) -> Data:
        if self.p != 1 and random.random() > self.p:
            return data

        keep_ratio = 1.0 - (torch.rand(1, device=data.pos.device) * self.max_dropout)

        return RandomSample(selection_factor=keep_ratio)(data)

    def __repr__(self):
        return f"{self.__class__.__name__}(<{self.max_dropout}, p={self.p})"
