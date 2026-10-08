import torch
from torch import Tensor
from torch.nn import Module


class FINER(Module):
    """The FINER activation, ``sin(omega_0 * (|x| + 1) * x)``."""

    def __init__(self, omega: float = 1, **kwargs) -> None:
        super().__init__()
        self.omega_0 = omega

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            alpha = torch.abs(x) + 1
        return torch.sin(self.omega_0 * alpha * x)


class FINERGauss(FINER):
    def forward(self, x: Tensor) -> Tensor:
        scale = 1.0
        with torch.no_grad():
            alpha = torch.abs(x) + 1
        x = torch.sin(self.omega_0 * alpha * x)
        return torch.exp(-((scale * x) ** 2))


class FINERWavelet(FINER):
    def forward(self, x: Tensor) -> Tensor:
        scale = 1.0
        with torch.no_grad():
            alpha = torch.abs(x) + 1
        x = torch.sin(self.omega_0 * alpha * x)
        return torch.exp(self.omega_0 * x - torch.abs(scale * x) ** 2)
