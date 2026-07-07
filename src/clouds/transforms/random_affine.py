import random

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, LinearTransformation


def random_scaling_matrix(
    dim: int, scales: tuple[float, float], uniform_scaling: bool = False, device: str | torch.device = 'cpu'
) -> torch.Tensor:
    """
    Generate a random scaling matrix.

    Args:
        dim (int): Dimensionality (2 or 3).
        scales (tuple): (min_scale, max_scale)
        uniform_scaling (bool): If True, all axes are scaled equally.
        device (torch.device): Device to create the tensor on.

    Returns:
        torch.Tensor: A (dim, dim) diagonal scaling matrix.
    """
    scale = torch.rand(dim, device=device) * (scales[1] - scales[0]) + scales[0]
    if uniform_scaling:
        scale[:] = scale[0]
    return torch.diag(scale)


def random_rotation_matrix(
    dim: int,
    degrees: tuple[float, float] | list[float],
    axis: list[int],
    device: str | torch.device = 'cpu',
) -> torch.Tensor:
    """
    Generate a random rotation matrix.

    Args:
        dim (int): Dimensionality (2 or 3).
        degrees (tuple): Rotation angle range in degrees.
        axis (list): Axes to rotate around (0=X, 1=Y, 2=Z).
        device (torch.device): Device to create the tensor on.

    Returns:
        torch.Tensor: A (dim, dim) rotation matrix.
    """
    if isinstance(degrees, tuple):
        angle = torch.deg2rad(torch.FloatTensor(1).uniform_(*degrees))
    else:
        angle = torch.deg2rad(torch.tensor([random.choice(degrees)]))

    sin, cos = torch.sin(angle), torch.cos(angle)

    if dim == 2:
        return torch.tensor([[cos, sin], [-sin, cos]], device=device)

    # TODO: this should work for arbitrary dimensionality!
    rotation_matrix = torch.eye(dim, device=device)
    for ax in axis:
        rot = torch.eye(3, device=device)
        if ax == 0:
            rot = torch.tensor([[1, 0, 0], [0, cos, sin], [0, -sin, cos]], device=device)
        elif ax == 1:
            rot = torch.tensor([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]], device=device)
        elif ax == 2:
            rot = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]], device=device)
        rotation_matrix = rot @ rotation_matrix

    return rotation_matrix


def transform_normals(normals: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """
    Applies a general linear transformation to normal vectors.
    The correct method is to use the **inverse transpose** of the transformation matrix.

    Args:
        normals (Tensor): The (N, D) tensor of normal vectors.
        transform (Tensor): The (D, D) transformation matrix applied to positions.

    Returns:
        Tensor: Transformed and re-normalized normals of shape (N, D).
    """
    assert normals.dim() == 2 and transform.dim() == 2
    assert normals.size(1) == transform.size(0) == transform.size(1)

    # inverse for correct normal transformation
    normal_transform = torch.linalg.inv(transform)
    # todo: why isn't this transposed?

    transformed_normals = normals @ normal_transform

    # Re-normalize to unit vectors
    transformed_normals = transformed_normals / torch.linalg.vector_norm(transformed_normals, dim=-1, keepdim=True)

    return transformed_normals


class RandomScale(BaseTransform):
    """Randomly scales node positions.

    Args:
        scales (tuple): Scaling range (min_scale, max_scale).
        uniform_scaling (bool, optional): Whether to apply the same scale to all axes.
        correct_norm (bool, optional): Whether to correct normal vectors after scaling.
        p (float, optional): Probability of applying the scaling.
    """

    def __init__(
        self,
        scales: tuple[float, float] = (0.8, 1.2),
        uniform_scaling: bool = False,
        correct_norm: bool = True,
        p: float = 1.0,
    ) -> None:
        super().__init__()
        self.low, self.high = scales
        self.uniform_scaling = uniform_scaling
        self.correct_norm = correct_norm
        self.p = p

    def forward(self, data: Data) -> Data:
        if random.random() >= self.p:
            return data

        dim = data.node_stores[-1].pos.size(-1)
        device = data.node_stores[0].pos.device

        scaling_matrix = random_scaling_matrix(dim, (self.low, self.high), self.uniform_scaling, device)
        data = LinearTransformation(scaling_matrix)(data)

        for store in data.node_stores:
            if self.correct_norm and hasattr(store, 'norm'):
                store.norm = transform_normals(store.norm, scaling_matrix)

        return data

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(scales={self.low, self.high}, '
            f'uniform_scaling={self.uniform_scaling}, '
            f'correct_norm={self.correct_norm}, '
            f'p={self.p})'
        )


class RandomRotate(BaseTransform):
    """Randomly rotates node positions around given axis/axes.

    Args:
        degrees (tuple or float): Rotation angle range in degrees.
        axis (int or list of int): Axis or axes to rotate around (0=X, 1=Y, 2=Z).
        correct_norm (bool, optional): Whether to correct normal vectors after scaling.
        p (float, optional): Probability of applying the rotation.
    """

    def __init__(
        self,
        degrees: tuple[float, float] | float = 360.0,
        axis: int | list[int] = 2,
        correct_norm: bool = True,
        p: float = 1.0,
    ) -> None:
        super().__init__()

        self.degrees = degrees if isinstance(degrees, tuple) else (-abs(degrees), abs(degrees))
        self.axis = axis if isinstance(axis, list) else [axis]
        self.correct_norm = correct_norm
        self.p = p

    def forward(self, data: Data) -> Data:
        if random.random() >= self.p:
            return data

        dim = data.node_stores[-1].pos.size(-1)
        device = data.node_stores[0].pos.device

        rotation_matrix = random_rotation_matrix(dim, self.degrees, self.axis, device)
        data = LinearTransformation(rotation_matrix)(data)

        for store in data.node_stores:
            if self.correct_norm and hasattr(store, 'norm'):
                store.norm = transform_normals(store.norm, rotation_matrix)

        return data

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(degrees={self.degrees}, correct_norm={self.correct_norm}, axis={self.axis}, p={self.p})'
        )


class RandomScaleAndRotate(BaseTransform):
    """Scales node positions by a randomly sampled factor and rotates node positions
    around a specific axis by a randomly sampled angle, using a single combined transformation.

    Args:
        scales (tuple or list): Scaling factors from which the scaling is sampled.
            Expected format is (min_scale, max_scale).
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by `[-degrees, degrees]`.
        axis (int, optional): The rotation axis. (default: :obj:`2`)
        correct_norm (bool, optional): Whether to correct normal vectors after
            transformation. (default: :obj:`True`)
        rotate_prob (float, optional): Probability of applying rotation.
            (default: :obj:`1.0`)
        scale_prob (float, optional): Probability of applying scaling.
            (default: :obj:`1.0`)
    """

    def __init__(
        self,
        scales: tuple[float, float] | list[float] = (0.8, 1.2),
        degrees: tuple[float, float] | list[float] | float = 360.0,
        axis: int | list[int] = 2,
        uniform_scaling: bool = False,
        correct_norm: bool = True,
        rotate_prob: float = 1.0,
        scale_prob: float = 1.0,
    ) -> None:
        super().__init__()
        self.low, self.high = scales
        self.correct_norm = correct_norm
        self.scale_prob = scale_prob
        self.uniform_scaling = uniform_scaling

        if isinstance(degrees, (int, float)):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list))
        self.degrees = degrees
        self.axis = axis if isinstance(axis, list) else [axis]
        self.rotate_prob = rotate_prob

    def forward(self, data: Data) -> Data:
        # Check if we should apply transformations
        apply_scale = random.random() < self.scale_prob
        apply_rotate = random.random() < self.rotate_prob

        if not (apply_scale or apply_rotate):
            return data

        # Get the dimensionality of the point cloud
        assert data.node_stores[-1].pos is not None
        dim = data.node_stores[-1].pos.size(-1)
        device = data.node_stores[0].pos.device

        # Start with identity matrix
        transform_matrix = torch.eye(dim, device=device)

        # Generate scaling matrix if needed
        if apply_scale:
            transform_matrix = random_scaling_matrix(dim, (self.low, self.high), self.uniform_scaling, device)

        # Generate rotation matrix if needed
        if apply_rotate:
            transform_matrix = random_rotation_matrix(dim, self.degrees, self.axis, device) @ transform_matrix

        # Apply the combined transformation
        data = LinearTransformation(transform_matrix)(data)

        # Correct normal vectors if necessary
        for store in data.node_stores:
            if self.correct_norm and hasattr(store, 'norm'):
                store.norm = transform_normals(store.norm, transform_matrix)

        return data

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'scales={self.low, self.high}, '
            f'degrees={self.degrees}, axis={self.axis}, '
            f'correct_norm={self.correct_norm}, '
            f'rotate_prob={self.rotate_prob}, '
            f'scale_prob={self.scale_prob}'
            f')'
        )
