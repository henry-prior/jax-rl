import jax
import jax.numpy as jnp
from jax import random
from flax import serialization
import numpy as onp
from typing import Iterator, Union


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


class PRNGSequence(Iterator[random.PRNGKey]):
    """Iterator of PRNGKeys.
        >>> seq = hk.PRNGSequence(42)  # OR pass a jax.random.PRNGKey
        >>> key1 = next(seq)
        >>> key2 = next(seq)
        >>> assert key1 is not key2
    """

    def __init__(self, key_or_seed: Union[random.PRNGKey, int]):
        if isinstance(key_or_seed, int):
            key = random.PRNGKey(key_or_seed)
        else:
            assert_is_prng_key(key_or_seed)
            key = key_or_seed
        self._key = key

    def peek(self):
        return self._key

    def replace(self, key: random.PRNGKey):
        self._key = key

    def __next__(self) -> random.PRNGKey:
        key, subkey = random.split(self._key)
        self._key = key
        return subkey

    next = __next__


@jax.vmap
def double_mse(q1, q2, qt):
    return jnp.square(qt - q1).mean() + jnp.square(qt - q2).mean()


@jax.vmap
def mse(pred, true):
    return jnp.square(true - pred).mean()
