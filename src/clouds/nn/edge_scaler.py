import math

import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn import BatchNorm


def gini_coefficient(x: torch.Tensor) -> float:
    x = x.flatten().sort()[0]
    n = x.numel()
    x = x / x.sum()
    weighted_sum = (x * torch.arange(1, n + 1, device=x.device)).sum()
    return (2 * weighted_sum / n - (n + 1) / n).item()


class ScaledVector(Module):
    def __init__(self) -> None:
        super().__init__()
        self.out_channels = 3  # Assumes 3d points
        self.register_buffer('scale', torch.tensor(0.0))

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.scale == 0:
            self.scale = 1 / torch.std(x)
        return x * self.scale


class QuantileScaledVector(ScaledVector):
    def __init__(self, q: float = 0.5) -> None:
        super().__init__()
        self.q = q

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.scale == 0:
            r = torch.linalg.vector_norm(x, dim=-1)
            self.scale = 1 / torch.quantile(r, self.q)
        return x * self.scale

    def extra_repr(self) -> str:
        return f"q={self.q}"


class HyperbolicSine(QuantileScaledVector):
    def forward(self, x: Tensor) -> Tensor:
        return torch.asinh(super().forward(x))


class LearnableVectorArctan(Module):
    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.z = torch.nn.Parameter(torch.tensor([0.0]))  # Initialize with 0
        self.eps = eps
        self.out_channels = 3

    def forward(self, x: Tensor) -> Tensor:
        a = torch.sigmoid(self.z)
        r = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        r_scaled = torch.arctan(r * a) / a
        print(self.z, a)
        return (x / torch.clamp(r, min=self.eps)) * r_scaled

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


class LearnableVectorTanh(LearnableVectorArctan):
    def forward(self, x: Tensor) -> Tensor:
        a = torch.sigmoid(self.z)
        r = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        r_scaled = torch.tanh(r * a) / a
        print(self.z, a)
        return (x / torch.clamp(r, min=self.eps)) * r_scaled


class LearnableVectorSoftClip(Module):
    def __init__(self, eps: float = 1e-12, n: float = 2) -> None:
        super().__init__()
        self.z = torch.nn.Parameter(torch.tensor([0.0]))  # Initialize with 0
        self.eps, self.n = eps, n
        self.out_channels = 3

    def forward(self, x: Tensor) -> Tensor:
        a = torch.sigmoid(self.z)
        r = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        r_scaled = r / (1 + (r * a) ** (2 * self.n)) ** (1 / (2 * self.n))
        print(self.z, a)
        return (x / torch.clamp(r, min=self.eps)) * r_scaled

    def extra_repr(self) -> str:
        return f"eps={self.eps}, n={self.n}"


class VectorSoftsign(Module):
    def __init__(self, k: float = 1) -> None:
        super().__init__()
        self.k = k

    def forward(self, x: Tensor) -> Tensor:
        x = x * self.k
        return x / (1.0 + torch.linalg.vector_norm(x, dim=-1, keepdim=True))

    def extra_repr(self) -> str:
        return f"k={self.k}"


class VectorTanh(Module):
    def __init__(self, k: float = 1, eps: float = 1e-12) -> None:
        super().__init__()
        self.k = k
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        r = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        return torch.tanh(r * self.k) * x / torch.clamp(r, min=self.eps) / self.k

    def extra_repr(self) -> str:
        return f"k={self.k}"


class VectorSignedLog(Module):
    def __init__(self, k: float = 1, eps: float = 1e-12) -> None:
        super().__init__()
        self.k = k
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        r = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        return torch.log1p(r * self.k) * x / torch.clamp(r, min=self.eps) / self.k

    def extra_repr(self) -> str:
        return f"k={self.k}"


class DirectionMagnitude(Module):
    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps
        self.out_channels = 4  # Assumes 3d points
        self.register_buffer('scale', torch.tensor(0, dtype=torch.float))

    def forward(self, x: Tensor) -> Tensor:
        r = torch.linalg.vector_norm(x, dim=-1, keepdim=True) + self.eps  # FIXME: redundant
        x = x / torch.clamp(r, min=self.eps)
        if self.training and self.scale == 0:
            print(f"Setting edge scale based on mean length of {torch.mean(r)}")
            self.scale = 1 / torch.mean(r)
        return torch.cat([x, r * self.scale], dim=-1)


class DirectionZMagnitude(Module):
    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps
        self.out_channels = 4  # Assumes 3d points

        self.register_buffer('mean', torch.tensor(0, dtype=torch.float))
        self.register_buffer('std', torch.tensor(0, dtype=torch.float))

    def forward(self, x: Tensor) -> Tensor:
        r = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        x = x / torch.clamp(r, min=self.eps)
        if self.training and self.std == 0:
            self.std, self.mean = torch.std_mean(r)
        return torch.cat([x, (r - self.mean) / self.std], dim=-1)


class VectorDirectionMagnitude(Module):
    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps
        self.out_channels = 7  # Assumes 3d points
        self.register_buffer('vector_scale', torch.tensor(0.0))
        self.register_buffer('magnitude_scale', torch.tensor(0.0))

    def forward(self, x: Tensor) -> Tensor:
        r = torch.linalg.vector_norm(x, dim=-1, keepdim=True) + self.eps
        if self.training and self.vector_scale == 0:
            self.vector_scale = 1 / torch.std(x)
            self.magnitude_scale = 1 / torch.mean(r)
        return torch.cat(
            [
                x * self.vector_scale,
                x / torch.clamp(r, min=self.eps),
                r * self.magnitude_scale,
            ],
            dim=-1,
        )


class VectorInvVector(ScaledVector):
    def __init__(self, eps: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eps = eps
        self.out_channels = 6

    def forward(self, x: Tensor) -> Tensor:
        x = super().forward(x)
        r = torch.linalg.vector_norm(x, dim=-1, keepdim=True) + self.eps
        inv_x = x / (r**2 + self.eps)
        return torch.cat([x, inv_x], dim=-1)

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


class SpectralEncoding(Module):
    def __init__(self, num_frequencies: int = 48, skip: bool = True, in_channels: int = 3) -> None:
        super().__init__()
        self.skip = skip

        # Set min and max frequencies
        self.register_buffer('scale', torch.tensor(0.0))
        self.out_channels = (in_channels if skip else 0) + 2 * num_frequencies
        self.frequencies = torch.nn.Parameter(torch.zeros((in_channels, num_frequencies)))

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.scale == 0:
            distances = torch.linalg.vector_norm(x, dim=-1)
            nonzero_indices = distances.nonzero()

            # Determine scaling
            nonzero_x = x[nonzero_indices]
            self.scale = 1 / torch.std(nonzero_x)
            distances = distances * self.scale

        # Normalize scale
        x = x * self.scale

        # Determine frequencies
        if self.training and (self.frequencies == 0).all():
            distances = torch.linalg.vector_norm(x, dim=-1)
            nonzero_distances = distances[distances.nonzero()]
            min_dist, max_dist = torch.quantile(nonzero_distances, 0.1), torch.quantile(nonzero_distances, 0.99)
            magnitudes = torch.logspace(
                math.log10(math.pi / (max_dist * 2)),
                math.log10(math.pi / min_dist),
                self.frequencies.size(-1),
                device=self.frequencies.device,
            )
            if x.size(-1) > 1:
                directions = torch.randn(x.size(-1), magnitudes.size(0), device=self.frequencies.device)
                directions = directions / directions.norm(dim=0, keepdim=True).clamp(min=1e-8)
                self.frequencies.data = directions * magnitudes
            else:
                self.frequencies.data = magnitudes.unsqueeze(0)

        # Project onto frequency bands: (..., F)
        proj = x @ self.frequencies

        # Fourier features
        encoded = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        return torch.cat([x, encoded], dim=-1) if self.skip else encoded


class DirectionSpectralMagnitude(Module):
    def __init__(self, eps: float = 1e-12, num_frequencies: int = 16, skip: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.spectral = SpectralEncoding(num_frequencies=num_frequencies, skip=skip, in_channels=1)
        self.out_channels = 3 + self.spectral.out_channels  # Assumes 3d points

    def forward(self, x: Tensor) -> Tensor:
        r = torch.linalg.vector_norm(x, dim=-1, keepdim=True) + self.eps
        return torch.cat([x / r, self.spectral(r)], dim=-1)


class DirectionMagnitudeInvMagnitude(Module):
    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps
        self.out_channels = 5  # Assumes 3d points
        self.a = torch.nn.Parameter(torch.tensor(0.0))
        self.norm, self.inv_norm = BatchNorm(1), BatchNorm(1)

    def forward(self, x: Tensor) -> Tensor:
        r = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        x = x / torch.clamp(r, min=self.eps)
        inv_r = 1 / (torch.sigmoid(self.a) + r)
        print(self.a.item())
        return torch.cat([x, self.norm(r), self.inv_norm(inv_r)], dim=-1)


class DirectionLogMagnitude(Module):
    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps
        self.out_channels = 4  # Assumes 3d points

    def forward(self, x: Tensor) -> Tensor:
        r = torch.linalg.vector_norm(x, dim=-1, keepdim=True) + self.eps
        x = x / torch.clamp(r, min=self.eps)
        r = torch.log1p(10 * r)  # TODO: 10?
        return torch.cat([x, r], dim=-1)


class VectorDirection(Module):
    # Inspired by https://arxiv.org/pdf/2507.12602
    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps
        self.out_channels = 6  # Assumes 3d points
        self.register_buffer('scale', torch.tensor(0.0))

    def forward(self, x: Tensor) -> Tensor:
        r = torch.linalg.vector_norm(x, dim=-1, keepdim=True) + self.eps
        if self.training and self.scale == 0:
            self.scale = 1 / torch.std(x)
        return torch.cat([x * self.scale, x / r], dim=-1)


class DirectionMagnitudeRBF(Module):
    def __init__(self, num_channels: int = 32, dim: int = 3, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps
        self.num_channels = num_channels
        self.out_channels = dim + num_channels + 1

        self.centers = torch.nn.Parameter(torch.zeros(num_channels))
        self.log_q = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: Tensor) -> Tensor:
        r = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        direction = x / torch.clamp(r, min=self.eps)

        if self.training and (self.centers == 0).all():
            with torch.no_grad():
                nonzero_r = r.flatten()[r.flatten().nonzero()]
                min_dist, max_dist = torch.quantile(nonzero_r, 0.1), torch.quantile(nonzero_r, 0.99)
                self.centers.data = torch.logspace(
                    math.log10(min_dist / 2),
                    math.log10(max_dist * 2),
                    self.centers.size(-1),
                    device=self.centers.device,
                )
                spacing = self.centers.clone()
                spacing[1:] = self.centers[1:] - self.centers[:-1]
                self.log_q.data = torch.log(1 / spacing**2)

        q = torch.exp(self.log_q)

        rbf = torch.exp(-q * (r - self.centers) ** 2)
        return torch.cat(
            [
                direction,
                rbf / rbf.sum(dim=-1, keepdim=True).clamp(min=self.eps),
                rbf.sum(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
