import jax
import numpy as onp
from haiku import PRNGSequence
from jax import random


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int = int(2e6),
        episode_length: int = None,
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        state_buffer_size = (
            (max_size, state_dim)
            if episode_length is None
            else (max_size, episode_length, state_dim)
        )
        action_buffer_size = (
            (max_size, action_dim)
            if episode_length is None
            else (max_size, episode_length, action_dim)
        )
        next_state_buffer_size = (
            (max_size, state_dim)
            if episode_length is None
            else (max_size, episode_length, state_dim)
        )
        reward_buffer_size = (
            (max_size, 1) if episode_length is None else (max_size, episode_length, 1)
        )
        not_done_buffer_size = (
            (max_size, 1) if episode_length is None else (max_size, episode_length, 1)
        )

        self.state = onp.empty(state_buffer_size)
        self.action = onp.empty(action_buffer_size)
        self.next_state = onp.empty(next_state_buffer_size)
        self.reward = onp.empty(reward_buffer_size)
        self.not_done = onp.empty(not_done_buffer_size)

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, rng: PRNGSequence, batch_size: int):
        ind = random.randint(rng, (batch_size,), 0, self.size)

        return (
            jax.device_put(self.state[ind]),
            jax.device_put(self.action[ind]),
            jax.device_put(self.next_state[ind]),
            jax.device_put(self.reward[ind]),
            jax.device_put(self.not_done[ind]),
        )
