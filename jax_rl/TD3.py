from flax import optim
import jax
from jax import random
import jax.numpy as jnp
from haiku import PRNGSequence

from models import TD3Actor
from models import TD3Critic
from models import build_td3_actor_model
from models import build_td3_critic_model
from utils import double_mse, apply_model, copy_params
from saving import save_model, load_model


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
    actor_target_params: dict,
    critic_target_params: dict,
) -> jnp.ndarray:
    noise = jnp.clip(
        random.normal(rng, action.shape) * policy_noise, -noise_clip, noise_clip
    )

    next_action = jnp.clip(actor_target(next_state) + noise, -max_action, max_action)

    target_Q1, target_Q2 = critic_target(next_state, next_action)
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
    def loss_fn(critic):
        current_Q1, current_Q2 = critic(state, action)
        critic_loss = double_mse(current_Q1, current_Q2, target_Q)
        return jnp.mean(critic_loss)

    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


@jax.jit
def actor_step(optimizer, critic, state):
    critic = critic.target

    def loss_fn(actor):
        actor_loss = -critic(state, actor(state), Q1=True)
        return jnp.mean(actor_loss)

    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        lr=3e-4,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        expl_noise=0.1,
        noise_clip=0.5,
        policy_freq=2,
        seed=0,
    ):

        self.rng = PRNGSequence(seed)

        actor_input_dim = [((1, state_dim), jnp.float32)]

        init_rng = next(self.rng)

        actor = build_td3_actor_model(actor_input_dim, action_dim, max_action, init_rng)
        self.actor_target = build_td3_actor_model(
            actor_input_dim, action_dim, max_action, init_rng
        )
        actor_optimizer = optim.Adam(learning_rate=lr).create(actor)
        self.actor_optimizer = jax.device_put(actor_optimizer)

        init_rng = next(self.rng)

        critic_input_dim = [
            ((1, state_dim), jnp.float32),
            ((1, action_dim), jnp.float32),
        ]

        critic = build_td3_critic_model(critic_input_dim, init_rng)
        self.critic_target = build_td3_critic_model(critic_input_dim, init_rng)
        critic_optimizer = optim.Adam(learning_rate=lr).create(critic)
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
            self.actor_target,
            self.critic_target,
        )

    def select_action(self, state):
        return apply_model(self.actor_optimizer.target, state).flatten()

    def sample_action(self, state):
        return self.select_action(
            state
        ) + self.max_action * self.expl_noise * random.normal(
            next(self.rng), self.action_dim
        )

    def train(self, replay_buffer, batch_size=100):
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

            self.critic_target = copy_params(
                self.critic_optimizer.target, self.critic_target, self.tau
            )
            self.actor_target = copy_params(
                self.actor_optimizer.target, self.actor_target, self.tau
            )

    def save(self, filename):
        save_model(filename + "_critic", self.critic_optimizer)
        save_model(filename + "_actor", self.actor_optimizer)

    def load(self, filename):
        self.critic_optimizer = load_model(filename + "_critic", self.critic_optimizer)
        self.critic_optimizer = jax.device_put(self.critic_optimizer)
        self.critic_target = self.critic_target.replace(
            params=self.critic_optimizer.target.params
        )

        self.actor_optimizer = load_model(filename + "_actor", self.actor_optimizer)
        self.actor_optimizer = jax.device_put(self.actor_optimizer)
        self.actor_target = self.actor_target.replace(
            params=self.actor_optimizer.target.params
        )
