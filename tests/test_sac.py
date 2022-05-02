import jax
import numpy as np

from .utils import *
from jax_rl import SAC


def SAC_policy(
    env,
    backend: Backend,
    discount: float = 0.99,
    policy_freq: int = 2,
    tau: float = 0.005,
):
    state_dim, action_dim, max_action = get_policy_args(env, backend)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": discount,
        "policy_freq": policy_freq,
        "tau": tau,
    }

    return SAC.SAC(**kwargs)


def test_sac_runs_dmc(all_dmc_environments):
    for env in all_dmc_environments:
        policy = SAC_policy(env, backend=Backend.DMC)

        timestep = env.reset()

        state = flat_obs(timestep.observation)

        action = (policy.select_action(state)).clip(
            -policy.max_action, policy.max_action
        )

        env.step(action)

    # only testing that this runs without errors
    assert True


def test_sac_runs_gym(all_gym_environments):
    for env in all_gym_environments:
        policy = SAC_policy(env, backend=Backend.GYM)

        state = env.reset()

        action = policy.select_action(np.array(state))
        env.step(action)
        action = policy.sample_action(jax.random.PRNGKey(0), np.array(state))
        env.step(action)

    # only testing that this runs without errors
    assert True
