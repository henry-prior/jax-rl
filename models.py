from flax import nn
import jax
from jax import random
import jax.numpy as jnp
import numpy as onp

from utils import gaussian_likelihood


class TD3Actor(nn.Module):
    def apply(self, x, action_dim, max_action):
        x = nn.Dense(x, features=256)
        x = nn.relu(x)
        x = nn.Dense(x, features=256)
        x = nn.relu(x)
        x = nn.Dense(x, features=action_dim)
        return max_action * nn.tanh(x)


class TD3Critic(nn.Module):
    def apply(self, state, action, Q1=False):
        state_action = jnp.concatenate([state, action], axis=1)

        q1 = nn.Dense(state_action, features=256)
        q1 = nn.relu(q1)
        q1 = nn.Dense(q1, features=256)
        q1 = nn.relu(q1)
        q1 = nn.Dense(q1, features=1)

        if Q1: return q1

        q2 = nn.Dense(state_action, features=256)
        q2 = nn.relu(q2)
        q2 = nn.Dense(q2, features=256)
        q2 = nn.relu(q2)
        q2 = nn.Dense(q2, features=1)

        return q1, q2


class DoubleCritic(nn.Module):
    def apply(self, state, action, Q1=False):
        state_action = jnp.concatenate([state, action], axis=1)

        q1 = nn.Dense(state_action, features=500)
        q1 = nn.LayerNorm(q1)
        q1 = nn.tanh(q1)
        q1 = nn.Dense(q1, features=500)
        q1 = nn.elu(q1)
        q1 = nn.Dense(q1, features=1)

        if Q1: return q1

        q2 = nn.Dense(state_action, features=500)
        q2 = nn.LayerNorm(q2)
        q2 = nn.tanh(q2)
        q2 = nn.Dense(q2, features=500)
        q2 = nn.elu(q2)
        q2 = nn.Dense(q2, features=1)

        return q1, q2


class GaussianPolicy(nn.Module):
    def apply(self, x, action_dim, max_action, key=None, MPO=False,
              sample=False, log_sig_min=-20, log_sig_max=2):
        x = nn.Dense(x, features=200)
        x = nn.LayerNorm(x)
        x = nn.tanh(x)
        x = nn.Dense(x, features=200)
        x = nn.elu(x)
        x = nn.Dense(x, features=2*action_dim)

        mu, log_sig = jnp.split(x, 2, axis=-1)
        log_sig = nn.softplus(log_sig)
        log_sig = jnp.clip(log_sig, log_sig_min, log_sig_max)

        if MPO:
            return mu, log_sig

        if not sample:
            return max_action * nn.tanh(mu), log_sig
        else:
            sig = jnp.exp(log_sig)
            pi = mu + random.normal(key, mu.shape) * sig
            log_pi = gaussian_likelihood(pi, mu, log_sig)
            pi = nn.tanh(pi)
            log_pi -= jnp.sum(jnp.log(nn.relu(1 - pi ** 2) + 1e-6), axis=1)
            return max_action * pi, log_pi


class Constant(nn.Module):
    def apply(self, start_value, dtype=jnp.float32):
        value = self.param('value', (1,), nn.initializers.ones)
        return start_value * jnp.asarray(value, dtype)


def build_constant_model(start_value, init_rng):
    constant = Constant.partial(start_value=start_value)
    _, init_params = constant.init(init_rng)

    return nn.Model(constant, init_params)


def build_td3_actor_model(input_shapes, action_dim, max_action, init_rng):
    actor = TD3Actor.partial(action_dim=action_dim, max_action=max_action)
    _, init_params = actor.init_by_shape(init_rng, input_shapes)

    return nn.Model(actor, init_params)


def build_td3_critic_model(input_shapes, init_rng):
    critic = TD3Critic.partial()
    _, init_params = critic.init_by_shape(init_rng, input_shapes)

    return nn.Model(critic, init_params)


def build_double_critic_model(input_shapes, init_rng):
    critic = DoubleCritic.partial()
    _, init_params = critic.init_by_shape(init_rng, input_shapes)

    return nn.Model(critic, init_params)


def build_gaussian_policy_model(input_shapes, action_dim, max_action, init_rng):
    actor = GaussianPolicy.partial(action_dim=action_dim, max_action=max_action)
    _, init_params = actor.init_by_shape(init_rng, input_shapes)

    return nn.Model(actor, init_params)
