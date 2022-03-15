from functools import partial

import jax
import jax.numpy as jnp
from flax import optim
from flax.core.frozen_dict import FrozenDict
from haiku import PRNGSequence
from jax import random

from jax_rl.models import apply_td3_actor_model
from jax_rl.models import apply_td3_critic_model
from jax_rl.models import build_td3_actor_model
from jax_rl.models import build_td3_critic_model
from jax_rl.saving import load_model
from jax_rl.saving import save_model
from jax_rl.utils import copy_params
from jax_rl.utils import double_mse


@partial(jax.jit, static_argnums=(6, 7, 8, 9, 10))
def get_td_target(
    rng: PRNGSequence,
    state: jnp.ndarray,
    action: jnp.ndarray,
    next_state: jnp.ndarray,
    reward: jnp.ndarray,
    not_done: jnp.ndarray,
    discount: float,
    policy_noise: float,
    noise_clip: float,
    max_action: float,
    action_dim: int,
    actor_target_params: FrozenDict,
    critic_target_params: FrozenDict,
) -> jnp.ndarray:
    noise = jnp.clip(
        random.normal(rng, action.shape) * policy_noise, -noise_clip, noise_clip
    )

    next_action = jnp.clip(
        apply_td3_actor_model(actor_target_params, action_dim, max_action, next_state)
        + noise,
        -max_action,
        max_action,
    )

    target_Q1, target_Q2 = apply_td3_critic_model(
        critic_target_params, next_state, next_action, False
    )
    target_Q = jnp.minimum(target_Q1, target_Q2)
    target_Q = reward + not_done * discount * target_Q

    return target_Q


@jax.jit
def critic_step(
    optimizer: optim.Optimizer,
    state: jnp.ndarray,
    action: jnp.ndarray,
    target_Q: jnp.ndarray,
) -> optim.Optimizer:
    def loss_fn(critic_params):
        current_Q1, current_Q2 = apply_td3_critic_model(
            critic_params, state, action, False
        )
        critic_loss = double_mse(current_Q1, current_Q2, target_Q)
        return jnp.mean(critic_loss)

    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


@partial(jax.jit, static_argnums=(2, 3))
def actor_step(
    optimizer: optim.Optimizer,
    critic_params: FrozenDict,
    max_action: float,
    action_dim: int,
    state: jnp.ndarray,
) -> optim.Optimizer:
    def loss_fn(actor_params):
        actor_loss = -apply_td3_critic_model(
            critic_params,
            state,
            apply_td3_actor_model(actor_params, action_dim, max_action, state),
            True,
        )
        return jnp.mean(actor_loss)

    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


class TD3:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        expl_noise: float = 0.1,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        seed: int = 0,
    ):
        self.rng = PRNGSequence(seed)

        actor_input_dim = (1, state_dim)

        init_rng = next(self.rng)

        actor_params = build_td3_actor_model(
            actor_input_dim, action_dim, max_action, init_rng
        )
        self.actor_target_params = build_td3_actor_model(
            actor_input_dim, action_dim, max_action, init_rng
        )
        actor_optimizer = optim.Adam(learning_rate=lr).create(actor_params)
        self.actor_optimizer = jax.device_put(actor_optimizer)

        init_rng = next(self.rng)

        critic_input_dim = [(1, state_dim), (1, action_dim)]

        critic_params = build_td3_critic_model(critic_input_dim, init_rng)
        self.critic_target_params = build_td3_critic_model(critic_input_dim, init_rng)
        critic_optimizer = optim.Adam(learning_rate=lr).create(critic_params)
        self.critic_optimizer = jax.device_put(critic_optimizer)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.expl_noise = expl_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.action_dim = action_dim

        self.total_it = 0

    @property
    def target_params(self):
        return (
            self.discount,
            self.policy_noise,
            self.noise_clip,
            self.max_action,
            self.action_dim,
            self.actor_target_params,
            self.critic_target_params,
        )

    def select_action(self, state: jnp.ndarray):
        return apply_td3_actor_model(
            self.actor_optimizer.target,
            self.action_dim,
            self.max_action,
            state.reshape(1, -1),
        ).flatten()

    def sample_action(self, state: jnp.ndarray):
        return self.select_action(
            state
        ) + self.max_action * self.expl_noise * random.normal(
            next(self.rng), self.action_dim
        )

    def train(self, replay_buffer, batch_size: int = 100):
        self.total_it += 1

        buffer_out = replay_buffer.sample(next(self.rng), batch_size)

        target_Q = jax.lax.stop_gradient(
            get_td_target(next(self.rng), *buffer_out, *self.target_params)
        )

        state, action, *_ = buffer_out

        self.critic_optimizer = critic_step(
            self.critic_optimizer, state, action, target_Q
        )

        if self.total_it % self.policy_freq == 0:

            self.actor_optimizer = actor_step(
                self.actor_optimizer,
                self.critic_optimizer.target,
                self.max_action,
                self.action_dim,
                state,
            )

            self.critic_target_params = copy_params(
                self.critic_target_params, self.critic_optimizer.target, self.tau
            )
            self.actor_target_params = copy_params(
                self.actor_target_params, self.actor_optimizer.target, self.tau
            )

    def save(self, filename: str):
        save_model(filename + "_critic", self.critic_optimizer)
        save_model(filename + "_actor", self.actor_optimizer)

    def load(self, filename: str):
        self.critic_optimizer = load_model(filename + "_critic", self.critic_optimizer)
        self.critic_optimizer = jax.device_put(self.critic_optimizer)
        self.critic_target_params = self.critic_optimizer.target.copy()

        self.actor_optimizer = load_model(filename + "_actor", self.actor_optimizer)
        self.actor_optimizer = jax.device_put(self.actor_optimizer)
        self.actor_target_params = self.actor_optimizer.target.copy()
