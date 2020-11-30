from flax.core.frozen_dict import FrozenDict
from flax import optim
from flax import linen as nn
import jax
from jax import random
import jax.numpy as jnp
from haiku import PRNGSequence

from jax_rl.models import TD3Actor
from jax_rl.models import TD3Critic
from jax_rl.models import build_td3_actor_model
from jax_rl.models import build_td3_critic_model
from jax_rl.utils import double_mse, apply_model, copy_params
from jax_rl.saving import save_model, load_model


@jax.jit
def get_td_target(
    rng: PRNGSequence,
    state: jnp.ndarray,
    action: jnp.ndarray,
    next_state: jnp.ndarray,
    reward: float,
    not_done: bool,
    discount: float,
    policy_noise: float,
    noise_clip: float,
    max_action: float,
    actor_target_params: FrozenDict,
    critic_target_params: FrozenDict,
) -> jnp.ndarray:
    noise = jnp.clip(
        random.normal(rng, action.shape) * policy_noise, -noise_clip, noise_clip
    )

    next_action = jnp.clip(
        apply_model(actor, actor_target_params, next_state) + noise,
        -max_action,
        max_action,
    )

    target_Q1, target_Q2 = apply_model(
        critic, critic_target_params, next_state, next_action
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
        current_Q1, current_Q2 = apply_model(critic, critic_params, state, action)
        critic_loss = double_mse(current_Q1, current_Q2, target_Q)
        return jnp.mean(critic_loss)

    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


# @jax.jit
def actor_step(
    optimizer: optim.Optimizer, critic_params: FrozenDict, state: jnp.ndarray,
) -> optim.Optimizer:

    def loss_fn(actor_params):
        actor_loss = -apply_model(
            critic,
            critic_params,
            state,
            apply_model(actor, actor_params, state),
            Q1=True,
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

        # TODO: has to be a cleaner way to do this
        global actor
        actor, actor_params = build_td3_actor_model(
            actor_input_dim, action_dim, max_action, init_rng
        )
        _, self.actor_target_params = build_td3_actor_model(
            actor_input_dim, action_dim, max_action, init_rng
        )
        actor_optimizer = optim.Adam(learning_rate=lr).create(actor_params)
        self.actor_optimizer = jax.device_put(actor_optimizer)

        init_rng = next(self.rng)

        critic_input_dim = [(1, state_dim), (1, action_dim)]

        # TODO: has to be a cleaner way to do this
        global critic
        critic, critic_params = build_td3_critic_model(critic_input_dim, init_rng)
        _, self.critic_target_params = build_td3_critic_model(
            critic_input_dim, init_rng
        )
        critic_optimizer = optim.Adam(learning_rate=lr).create(critic_params)
        self.critic_optimizer = jax.device_put(critic_optimizer)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.expl_noise = expl_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    @property
    def target_params(self):
        return (
            self.discount,
            self.policy_noise,
            self.noise_clip,
            self.max_action,
            self.actor_target_params,
            self.critic_target_params,
        )

    def select_action(self, state: jnp.ndarray):
        return apply_model(
            actor, self.actor_optimizer.target, state.reshape(1, -1)
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
                self.actor_optimizer, self.critic_optimizer, state
            )

            self.critic_target_params = copy_params(
                self.critic_optimizer.target, self.critic_target_params, self.tau
            )
            self.actor_target_params = copy_params(
                self.actor_optimizer.target, self.actor_target_params, self.tau
            )

    def save(self, filename: str):
        save_model(filename + "_critic", self.critic_optimizer)
        save_model(filename + "_actor", self.actor_optimizer)

    def load(self, filename: str):
        self.critic_optimizer = load_model(filename + "_critic", self.critic_optimizer)
        self.critic_optimizer = jax.device_put(self.critic_optimizer)
        self.critic_target_params = self.critic_optimizer.target

        self.actor_optimizer = load_model(filename + "_actor", self.actor_optimizer)
        self.actor_optimizer = jax.device_put(self.actor_optimizer)
        self.actor_target_params = self.actor_optimizer.target
