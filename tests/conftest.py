import random

import pytest
import torch


@pytest.fixture(autouse=True)
def seed():
    random.seed(0)
    torch.manual_seed(0)
