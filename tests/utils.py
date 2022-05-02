from enum import auto
from enum import Enum

import gym
import numpy as onp
import pytest
from dm_control import suite


class Backend(Enum):
    GYM = auto()
    DMC = auto()


def get_policy_args(env, backend: Backend):
    if backend == Backend.DMC:
        temp_timestep = env.reset()
        state_dim = flat_obs(temp_timestep.observation).shape[0]
        action_dim = env.action_spec().shape[0]
        max_action = float(env.action_spec().maximum[0])
    elif backend == Backend.GYM:
        state_dim = env.observation_space.shape[0]
        print(env.action_space.shape)
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
    else:
        raise ValueError(f"Unsupported env backend {backend}")

    return state_dim, action_dim, max_action


def flat_obs(o):
    return onp.concatenate([o[k].flatten() for k in o])


@pytest.fixture
def cartpole_swingup(seed: int = 42):
    env = suite.load("cartpole", "swingup", {"random": seed})
    return env


@pytest.fixture
def pendulum_swingup(seed: int = 42):
    env = suite.load("pendulum", "swingup", {"random": seed})
    return env


@pytest.fixture
def cheetah_run(seed: int = 42):
    env = suite.load("cheetah", "run", {"random": seed})
    return env


@pytest.fixture
def all_dmc_environments(cartpole_swingup, pendulum_swingup, cheetah_run):
    return cartpole_swingup, pendulum_swingup, cheetah_run


# @pytest.fixture
# def CartPole(seed: int = 42):
#    env = gym.make("CartPole-v0")
#    env.seed(seed)
#    return env


@pytest.fixture
def Pendulum(seed: int = 42):
    env = gym.make("Pendulum-v1")
    env.seed(seed)
    return env


@pytest.fixture
def HalfCheetah(seed: int = 42):
    env = gym.make("HalfCheetah-v3")
    env.seed(seed)
    return env


@pytest.fixture
def all_gym_environments(Pendulum, HalfCheetah):
    return Pendulum, HalfCheetah
