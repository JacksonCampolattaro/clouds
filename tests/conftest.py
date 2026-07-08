import random

import pytest
import torch


@pytest.fixture(autouse=True)
def seed():
    """Seed all RNGs before every test for reproducible randomized behavior."""
    random.seed(0)
    torch.manual_seed(0)
