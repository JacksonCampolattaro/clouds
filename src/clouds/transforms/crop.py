import math
import random

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from .apply_selection import apply_selection


class CylinderSelect(BaseTransform):
    """Select points within a cylinder aligned with the Z-axis.

    A center point is chosen (randomly or deterministically from the first
    point), and points are ranked by their XY-plane distance from that center.
    The cylinder has infinite height by default; set ``max_half_height`` to
    also clip along Z.

    Args:
        max_num_points: Hard cap on the number of selected points.
        max_radius: Maximum XY radius of the cylinder. Points beyond this
            distance are dropped even if they would fit within ``max_num_points``.
        max_half_height: Half-height of the cylinder along Z. Points with
            ``|z - center_z| > max_half_height`` are dropped. Defaults to
            ``math.inf`` (no height clipping).
        max_ratio: Upper bound on the fraction of the original cloud to keep.
        deterministic: If ``True``, always use the first point as center.
        sort_by_distance: If ``True`` (default), output indices are ordered by
            ascending XY distance. If ``False``, they are sorted in their
            original order.
    """

    def __init__(
        self,
        max_num_points: int,
        max_radius: float = math.inf,
        max_half_height: float = math.inf,
        max_ratio: float = 1.0,
        deterministic: bool = False,
        sort_by_distance: bool = True,
    ) -> None:
        super().__init__()
        self.max_num_points = max_num_points
        self.max_radius = max_radius
        self.max_half_height = max_half_height
        self.max_ratio = max_ratio
        self.deterministic = deterministic
        self.sort_by_distance = sort_by_distance

    def forward(self, data: Data) -> Data:
        assert isinstance(data.pos, Tensor) and isinstance(data.num_nodes, int)
        assert data.pos.shape[-1] >= 3, "CylinderSelect requires 3-D positions (x, y, z)"

        if data.num_nodes <= self.max_num_points:
            return data

        center_idx = 0 if self.deterministic else random.randrange(data.num_nodes)
        center = data.pos[center_idx, :]

        num_points = min(int(data.num_nodes * self.max_ratio), self.max_num_points)

        # Rank by XY distance only (cylinder axis == Z)
        xy_diff = data.pos[:, :2] - center[:2]
        xy_dist = torch.linalg.vector_norm(xy_diff, dim=-1)
        selected_indices = xy_dist.argsort()[:num_points]

        if not self.sort_by_distance:
            selected_indices, _ = selected_indices.sort()

        # Drop points outside the finite XY radius
        if math.isfinite(self.max_radius):
            mask = xy_dist[selected_indices] < self.max_radius
            selected_indices = selected_indices[mask]

        # Drop points outside the finite half-height along Z
        if math.isfinite(self.max_half_height):
            z_diff = (data.pos[selected_indices, 2] - center[2]).abs()
            selected_indices = selected_indices[z_diff < self.max_half_height]

        data.selection_index = selected_indices
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r{self.max_radius}, <{self.max_num_points}, deterministic={self.deterministic})"


class CylinderSample(CylinderSelect):
    """Select + apply: restrict the point cloud to a cylinder aligned with Z."""

    def forward(self, data: Data) -> Data:
        return apply_selection(super().forward(data))


class SlabSelect(BaseTransform):
    """Select points within a slab between two parallel planes.

    A center point is chosen (randomly or deterministically from the first point),
    and a plane normal is sampled (or fixed along the last axis when
    ``deterministic``). Points are ranked by their distance from the plane, and
    optionally clipped to ``half_thickness``.
    """

    def __init__(
        self,
        max_num_points: int,
        half_thickness: float = math.inf,
        max_ratio: float = 1.0,
        deterministic: bool = False,
        sort_by_distance: bool = True,
    ) -> None:
        super().__init__()
        self.max_num_points = max_num_points
        self.half_thickness = half_thickness
        self.max_ratio = max_ratio
        self.deterministic = deterministic
        self.sort_by_distance = sort_by_distance

    @staticmethod
    def _random_unit_vector(dim: int, device: torch.device) -> Tensor:
        """Sample a uniformly random unit vector of length ``dim``."""
        v = torch.randn(dim, device=device)
        return v / torch.linalg.vector_norm(v)

    def forward(self, data: Data) -> Data:
        assert isinstance(data.pos, Tensor) and isinstance(data.num_nodes, int)

        if data.num_nodes <= self.max_num_points:
            return data

        center_idx = 0 if self.deterministic else random.randrange(data.num_nodes)
        center = data.pos[center_idx, :2]

        if self.deterministic:
            normal = torch.zeros(2, device=data.pos.device)
            normal[-1] = 1.0  # (0, …, 0, 1)
        else:
            normal = self._random_unit_vector(2, data.pos.device)

        # Signed distance of every point from the plane
        signed_dist = (data.pos[:, :2] - center) @ normal  # shape: (N,)
        abs_dist = signed_dist.abs()

        # Keep only points within the slab thickness
        if math.isfinite(self.half_thickness):
            in_slab = abs_dist < self.half_thickness
            candidate_indices = in_slab.nonzero(as_tuple=False).squeeze(-1)
        else:
            candidate_indices = torch.arange(data.num_nodes, device=data.pos.device)

        # Rank candidates by closeness to the plane, apply count/ratio cap
        num_points = min(int(data.num_nodes * self.max_ratio), self.max_num_points)
        order = abs_dist[candidate_indices].argsort()
        selected_indices = candidate_indices[order[:num_points]]

        if not self.sort_by_distance:
            selected_indices, _ = selected_indices.sort()

        data.selection_index = selected_indices
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(d<{self.half_thickness}, <{self.max_num_points}, deterministic={self.deterministic})"


class SlabSample(SlabSelect):
    """Select + apply: restrict the point cloud to a slab between two planes."""

    def forward(self, data: Data) -> Data:
        return apply_selection(super().forward(data))
