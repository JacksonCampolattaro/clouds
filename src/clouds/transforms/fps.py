import math

import numpy
import torch_geometric
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from .apply_selection import apply_selection

try:
    HAS_TORCH_FPSAMPLE = True
    import torch_fpsample
except (ImportError, OSError):
    HAS_TORCH_FPSAMPLE = False


def _fpsample_fps(
    pos: Tensor,
    n: int | None = None,
    ratio: float | None = None,
    deterministic: bool = False,
) -> Tensor:
    if not n:
        assert ratio
        n = numpy.clip(
            int(ratio * pos.size(0)),
            min=1,
            max=pos.size(0),
        )
    return torch_fpsample.fps.sample(
        pos,
        n,
        h=min(math.floor(math.log2(pos.size(0))) - 1, 6),
        start_idx=0 if deterministic else None,
    )[1]


def _pyg_fps(
    pos: Tensor,
    n: int | None = None,
    ratio: float | None = None,
    deterministic: bool = False,
    batch: Tensor | None = None,
    batch_size: int | None = None,
) -> Tensor:
    if not ratio:
        assert batch is None
        assert n
        ratio = n / pos.size(0)
    return torch_geometric.nn.pool.fps(
        pos,
        ratio=ratio,
        batch=batch,
        random_start=not deterministic,
        batch_size=batch_size,
    )


def fps(
    pos: Tensor,
    n: int | None = None,
    ratio: float | None = None,
    deterministic: bool = False,
    batch: Tensor | None = None,
    batch_size: int | None = None,
) -> Tensor:
    assert (n or ratio) and not (n and ratio)
    if pos.device.type == "cpu" and batch is None and HAS_TORCH_FPSAMPLE:
        # Use torch_fpsample only if it is available AND data is on CPU
        return _fpsample_fps(pos, n=n, ratio=ratio, deterministic=deterministic)
    else:
        # PyG's fps works on CPU and GPU, and handles batches
        return _pyg_fps(pos, n=n, ratio=ratio, deterministic=deterministic, batch=batch, batch_size=batch_size)


class FurthestPointSelect(BaseTransform):
    def __init__(
        self,
        max_num_points: int = int(1e7),
        selection_factor: float = 1.0,
        min_num_points: int = 1,
        deterministic: bool = False,
    ):
        super().__init__()
        self.max_num_points = max_num_points
        self.selection_factor = selection_factor
        self.min_num_points = min_num_points
        self.deterministic = deterministic

    def forward(self, data: Data) -> Data:
        assert not isinstance(data.batch, Tensor)
        selection_size = int(data.num_nodes * self.selection_factor)
        selection_size = numpy.clip(
            selection_size,
            min=self.min_num_points,
            max=self.max_num_points,
        )

        data.selection_index = fps(
            data.pos,
            selection_size,
            deterministic=self.deterministic,
            batch=data.batch,
            batch_size=data.batch_size,
        )
        return data

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(*{self.selection_factor}, <{self.max_num_points}, deterministic={self.deterministic})"


class FurthestPointSample(FurthestPointSelect):
    def forward(self, data: Data) -> Data:
        return apply_selection(super().forward(data))
