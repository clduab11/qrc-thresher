"""Shared pytest fixtures."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def config_path():
    return Path('configs/alpha_lite.yaml')


@pytest.fixture
def default_config(config_path):
    from qrc_thresher.config import load_config

    return load_config(config_path)
