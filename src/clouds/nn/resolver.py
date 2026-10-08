from typing import Any

from torch.nn import Module
from torch_geometric.nn.resolver import activation_resolver as pyg_activation_resolver
from torch_geometric.nn.resolver import normalization_resolver as pyg_normalization_resolver
from torch_geometric.nn.resolver import normalize_string

from .activation import FINER, FINERGauss, FINERWavelet


def activation_resolver(query: Any | str = 'relu', *args, **kwargs) -> Module:
    """Resolve an activation name/class, extending PyG's resolver with the FINER family."""
    query = normalize_string(query) if isinstance(query, str) else query
    if query == 'finer':
        return FINER(*args, **kwargs)
    if query == 'finergauss':
        return FINERGauss(*args, **kwargs)
    if query == 'finerwavelet':
        return FINERWavelet(*args, **kwargs)
    else:
        return pyg_activation_resolver(query, *args, **kwargs)


def normalization_resolver(query: type | Any | str, *args, **kwargs) -> Any:
    """Resolve a normalization layer, passing classes straight through."""
    if isinstance(query, type):
        return query(*args, **kwargs)
    else:
        return pyg_normalization_resolver(query, *args, **kwargs)
