import numpy as onp
import pytest
from dm_control import suite


def flat_obs(o):
    return onp.concatenate([o[k].flatten() for k in o])


@pytest.fixture
def cartpole_environment(seed: int = 42):
    env = suite.load("cartpole", "swingup", {"random": seed})
    return env
