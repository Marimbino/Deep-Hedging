from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)
    np.random.seed(0)
    yield
