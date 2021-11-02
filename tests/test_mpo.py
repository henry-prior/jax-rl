import numpy as onp
import pytest
from dm_control import suite
from jax_rl import MPO
from jax_rl import utils

from .utils import cartpole_environment
from .utils import flat_obs


@pytest.fixture
def MPO_policy(cartpole_environment, discount: float = 0.99):
    temp_timestep = cartpole_environment.reset()
    state_dim = flat_obs(temp_timestep.observation).shape[0]
    action_dim = cartpole_environment.action_spec().shape[0]
    max_action = float(cartpole_environment.action_spec().maximum[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": discount,
    }

    return MPO.MPO(**kwargs)


def test_sac_runs(cartpole_environment, MPO_policy):
    env = cartpole_environment
    policy = MPO_policy

    timestep = env.reset()

    state = flat_obs(timestep.observation)

    action = (policy.select_action(state)).clip(-policy.max_action, policy.max_action)

    timestep = env.step(action)

    # only testing that this runs without errors
    assert True
