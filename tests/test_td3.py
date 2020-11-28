import numpy as onp
import pytest
from dm_control import suite
from .utils import cartpole_environment
from .utils import flat_obs

from jax_rl import TD3
from jax_rl import SAC
from jax_rl import MPO
from jax_rl import utils


@pytest.fixture
def TD3_policy(
    cartpole_environment,
    discount: float = 0.99,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    policy_freq: int = 2,
    expl_noise: float = 0.1,
    tau: float = 0.005,
):
    temp_timestep = cartpole_environment.reset()
    state_dim = flat_obs(temp_timestep.observation).shape[0]
    action_dim = cartpole_environment.action_spec().shape[0]
    max_action = float(cartpole_environment.action_spec().maximum[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": discount,
        "policy_noise": policy_noise * max_action,
        "noise_clip": noise_clip * max_action,
        "policy_freq": policy_freq,
        "expl_noise": expl_noise,
        "tau": tau,
    }

    return TD3.TD3(**kwargs)


def test_td3_runs(cartpole_environment, TD3_policy):
    env = cartpole_environment
    policy = TD3_policy

    timestep = env.reset()

    state = flat_obs(timestep.observation)

    action = (policy.select_action(state)).clip(-policy.max_action, policy.max_action)

    timestep = env.step(action)

    # only testing that this runs without errors
    assert True
