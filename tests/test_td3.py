import jax
import numpy as np

from .utils import *
from jax_rl import TD3


def TD3_policy(
    env,
    backend: Backend,
    discount: float = 0.99,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    policy_freq: int = 2,
    expl_noise: float = 0.1,
    tau: float = 0.005,
):
    state_dim, action_dim, max_action = get_policy_args(env, backend)

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


def test_td3_runs_dmc(all_dmc_environments):
    for env in all_dmc_environments:
        policy = TD3_policy(env, backend=Backend.DMC)

        timestep = env.reset()

        state = flat_obs(timestep.observation)

        action = (policy.select_action(state)).clip(
            -policy.max_action, policy.max_action
        )

        env.step(action)

    # only testing that this runs without errors
    assert True


def test_td3_runs_gym(all_gym_environments):
    for env in all_gym_environments:
        policy = TD3_policy(env, backend=Backend.GYM)

        state = env.reset()

        action = policy.select_action(np.array(state))
        env.step(action)
        action = policy.sample_action(jax.random.PRNGKey(0), np.array(state))
        env.step(action)

    # only testing that this runs without errors
    assert True
