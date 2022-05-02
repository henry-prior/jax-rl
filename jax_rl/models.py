from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from haiku import PRNGSequence
from jax import random

from jax_rl.utils import gaussian_likelihood


class TD3Actor(nn.Module):
    action_dim: int
    max_action: float

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.action_dim)(x)
        return self.max_action * nn.tanh(x)


class TD3Critic(nn.Module):
    @nn.compact
    def __call__(self, state, action, Q1=False):
        state_action = jnp.concatenate([state, action], axis=-1)

        q1 = nn.Dense(features=256)(state_action)
        q1 = nn.relu(q1)
        q1 = nn.Dense(features=256)(q1)
        q1 = nn.relu(q1)
        q1 = nn.Dense(features=1)(q1)

        if Q1:
            return q1

        q2 = nn.Dense(features=256)(state_action)
        q2 = nn.relu(q2)
        q2 = nn.Dense(features=256)(q2)
        q2 = nn.relu(q2)
        q2 = nn.Dense(features=1)(q2)

        return q1, q2


class DoubleCritic(nn.Module):
    @nn.compact
    def __call__(self, state, action, Q1=False):
        state_action = jnp.concatenate([state, action], axis=-1)

        q1 = nn.Dense(features=500)(state_action)
        q1 = nn.LayerNorm()(q1)
        q1 = nn.tanh(q1)
        q1 = nn.Dense(features=500)(q1)
        q1 = nn.elu(q1)
        q1 = nn.Dense(features=1)(q1)

        if Q1:
            return q1

        q2 = nn.Dense(features=500)(state_action)
        q2 = nn.LayerNorm()(q2)
        q2 = nn.tanh(q2)
        q2 = nn.Dense(features=500)(q2)
        q2 = nn.elu(q2)
        q2 = nn.Dense(features=1)(q2)

        return q1, q2


class GaussianPolicy(nn.Module):
    action_dim: int
    max_action: float
    log_sig_min: float = -20.0
    log_sig_max: float = None

    @nn.compact
    def __call__(self, x, key=None, sample=False, MPO=False):
        x = nn.Dense(features=200)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        x = nn.Dense(features=200)(x)
        x = nn.elu(x)
        x = nn.Dense(features=2 * self.action_dim)(x)

        mu, log_sig = jnp.split(x, 2, axis=-1)
        log_sig = jnp.clip(log_sig, self.log_sig_min, self.log_sig_max)

        if MPO:
            return mu, log_sig

        if not sample:
            return self.max_action * nn.tanh(mu), log_sig
        else:
            sig = jnp.exp(log_sig)
            pi = mu + random.normal(key, mu.shape) * sig
            log_pi = gaussian_likelihood(pi, mu, log_sig)
            pi = nn.tanh(pi)
            log_pi -= jnp.sum(
                jnp.log(nn.relu(1 - pi ** 2) + 1e-6), axis=1, keepdims=True,
            )
            return self.max_action * pi, log_pi


class Constant(nn.Module):
    start_value: float
    absolute: bool = False

    @nn.compact
    def __call__(self, dtype=jnp.float32):
        value = self.param(
            "value", lambda key, shape: jnp.full(shape, self.start_value, dtype), (1,)
        )
        if self.absolute:
            value = nn.softplus(value)
        return jnp.asarray(value, dtype)


def build_constant_model(
    start_value: float, init_rng: PRNGSequence, absolute: bool = False
) -> FrozenDict:
    constant = Constant(start_value=start_value, absolute=absolute)
    init_variables = constant.init(init_rng)

    return init_variables["params"]


@partial(jax.jit, static_argnums=(1, 2))
def apply_constant_model(
    params: FrozenDict, start_value: float, absolute: bool,
) -> jnp.ndarray:
    return Constant(start_value=start_value, absolute=absolute).apply(
        dict(params=params)
    )


def build_td3_actor_model(
    input_shapes: float, action_dim: int, max_action: float, init_rng: PRNGSequence
) -> FrozenDict:
    init_batch = jnp.ones(input_shapes, jnp.float32)
    actor = TD3Actor(action_dim=action_dim, max_action=max_action)
    init_variables = actor.init(init_rng, init_batch)

    return init_variables["params"]


@partial(jax.jit, static_argnums=(1, 2))
def apply_td3_actor_model(
    params: FrozenDict, action_dim: int, max_action: float, state: jnp.ndarray,
) -> jnp.ndarray:
    return TD3Actor(action_dim=action_dim, max_action=max_action).apply(
        dict(params=params), state
    )


def build_td3_critic_model(input_shapes, init_rng):
    init_batch = [jnp.ones(shape, jnp.float32) for shape in input_shapes]
    critic = TD3Critic()
    init_variables = critic.init(init_rng, *init_batch)

    return init_variables["params"]


@partial(jax.jit, static_argnums=3)
def apply_td3_critic_model(
    params: FrozenDict, state: jnp.ndarray, action: jnp.ndarray, Q1: bool
) -> jnp.ndarray:
    return TD3Critic().apply(dict(params=params), state, action, Q1=Q1)


def build_double_critic_model(input_shapes, init_rng):
    init_batch = [jnp.ones(shape, jnp.float32) for shape in input_shapes]
    critic = DoubleCritic()
    init_variables = critic.init(init_rng, *init_batch)

    return init_variables["params"]


@partial(jax.jit, static_argnums=3)
def apply_double_critic_model(
    params: FrozenDict, state: jnp.ndarray, action: jnp.ndarray, Q1: bool
) -> jnp.ndarray:
    return DoubleCritic().apply(dict(params=params), state, action, Q1=Q1)


def build_gaussian_policy_model(input_shapes, action_dim, max_action, init_rng):
    init_batch = jnp.ones(input_shapes, jnp.float32)
    policy = GaussianPolicy(action_dim=action_dim, max_action=max_action)
    init_variables = policy.init(init_rng, init_batch)

    return init_variables["params"]


@partial(jax.jit, static_argnums=(1, 2, 5, 6))
def apply_gaussian_policy_model(
    params: FrozenDict,
    action_dim: int,
    max_action: float,
    state: jnp.ndarray,
    key: PRNGSequence,
    sample: bool,
    MPO: bool,
) -> jnp.ndarray:
    return GaussianPolicy(action_dim=action_dim, max_action=max_action).apply(
        dict(params=params), state, key=key, sample=sample, MPO=MPO
    )
