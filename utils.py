from typing import Iterator, Union
import jax
import jax.numpy as jnp
from jax import random
import numpy as onp


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(2e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = onp.zeros((max_size, state_dim))
        self.action = onp.zeros((max_size, action_dim))
        self.next_state = onp.zeros((max_size, state_dim))
        self.reward = onp.zeros((max_size, 1))
        self.not_done = onp.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, rng, batch_size):
        ind = random.randint(rng, (batch_size,), 0, self.size)

        return (
            jax.device_put(self.state[ind]),
            jax.device_put(self.action[ind]),
            jax.device_put(self.next_state[ind]),
            jax.device_put(self.reward[ind]),
            jax.device_put(self.not_done[ind])
        )


@jax.jit
def copy_params(model, model_target, tau):
    update_params = jax.tree_multimap(
        lambda m1, mt: tau * m1 + (1 - tau) * mt,
        model.params, model_target.params)

    return model_target.replace(params=update_params)


@jax.vmap
def double_mse(q1, q2, qt):
    return jnp.square(qt - q1).mean() + jnp.square(qt - q2).mean()


@jax.vmap
def mse(pred, true):
    return jnp.square(true - pred).mean()


@jax.jit
def apply_model(model, x, *args, **kwargs):
    return model(x.reshape(1, -1), *args, **kwargs)


@jax.jit
def sample_from_multivariate_normal(rng, mean, cov, shape=None):
    return random.multivariate_normal(rng, mean, cov, shape=shape)


@jax.jit
def gaussian_likelihood(sample, mu, log_sig):
    pre_sum = -0.5 * (((sample - mu) / (jnp.exp(log_sig) + 1e-6)) ** 2 + 2 * log_sig + jnp.log(2 * onp.pi))
    return jnp.sum(pre_sum, axis=1)


@jax.jit
def kl_divergence(p, q):
    return (p * jnp.log(p / q)).sum()
