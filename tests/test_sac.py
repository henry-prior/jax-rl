import pytest
from jax_rl import SAC

from .utils import cartpole_environment
from .utils import flat_obs


@pytest.fixture
def SAC_policy(
    cartpole_environment,
    discount: float = 0.99,
    policy_freq: int = 2,
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
        "policy_freq": policy_freq,
        "tau": tau,
    }

    return SAC.SAC(**kwargs)


def test_sac_runs(cartpole_environment, SAC_policy):
    env = cartpole_environment
    policy = SAC_policy

    timestep = env.reset()

    state = flat_obs(timestep.observation)

    action = (policy.select_action(state)).clip(-policy.max_action, policy.max_action)

    timestep = env.step(action)

    # only testing that this runs without errors
    assert True
