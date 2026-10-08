import torch
from torch.nn import RMSNorm
from torch_geometric.nn.norm import BatchNorm, LayerNorm


class DeLABatchNorm(BatchNorm):
    def __init__(self, in_channels: int, init_weight: float = 1.0, **kwargs) -> None:
        self.init_weight = init_weight
        super().__init__(in_channels, **kwargs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if self.init_weight is not None:
            torch.nn.init.constant_(self.module.weight, self.init_weight)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.module.extra_repr()}, init_weight={self.init_weight})'


class DeLALayerNorm(LayerNorm):
    def __init__(self, in_channels: int, init_weight: float = 1.0, **kwargs) -> None:
        self.init_weight = init_weight
        super().__init__(in_channels, **kwargs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if self.init_weight is not None:
            torch.nn.init.constant_(self.weight, self.init_weight)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(init_weight={self.init_weight})'


class DeLARMSNorm(RMSNorm):
    def __init__(self, in_channels: int, init_weight: float = 1.0, **kwargs) -> None:
        self.init_weight = init_weight
        super().__init__(in_channels, **kwargs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if self.init_weight is not None:
            torch.nn.init.constant_(self.weight, self.init_weight)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()}, init_weight={self.init_weight})'
