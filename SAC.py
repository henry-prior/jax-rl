from functools import partial
import jax
import jax.numpy as jnp
from flax import optim
from haiku import PRNGSequence

from utils import (double_mse, sample_from_multivariate_normal,
                   apply_model, copy_params)
from saving import save_model, load_model
from models import (build_gaussian_policy_model,
                    build_double_critic_model,
                    build_constant_model)


def actor_loss_fn(log_alpha, log_p, min_q):
    return (jnp.exp(log_alpha) * log_p - min_q).mean()


def alpha_loss_fn(log_alpha, target_entropy, log_p):
    return (log_alpha * (-log_p - target_entropy)).mean()


@jax.jit
def get_td_target(rng, state, action, next_state, reward, not_done,
                  discount, max_action, actor, critic_target, log_alpha):
    next_action, next_log_p = actor(next_state, sample=True, key=rng)

    target_Q1, target_Q2 = critic_target(next_state, next_action)
    target_Q = jnp.minimum(target_Q1, target_Q2) - jnp.exp(log_alpha()) * next_log_p
    target_Q = reward + not_done * discount * target_Q

    return target_Q


@jax.jit
def critic_step(optimizer, state, action, target_Q):
    def loss_fn(critic):
        current_Q1, current_Q2 = critic(state, action)
        critic_loss = double_mse(current_Q1, current_Q2, target_Q)
        return jnp.mean(critic_loss)
    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


@jax.jit
def actor_step(rng, optimizer, critic, state, log_alpha):
    critic, log_alpha = critic.target, log_alpha.target
    def loss_fn(actor):
        actor_action, log_p = actor(state, sample=True, key=rng)
        q1, q2 = critic(state, actor_action)
        min_q = jnp.minimum(q1, q2)
        partial_loss_fn = jax.vmap(partial(actor_loss_fn, jax.lax.stop_gradient(log_alpha())))
        actor_loss = partial_loss_fn(log_p, min_q)
        return jnp.mean(actor_loss), log_p
    grad, log_p = jax.grad(loss_fn, has_aux=True)(optimizer.target)
    return optimizer.apply_gradient(grad), log_p


@jax.jit
def alpha_step(optimizer, log_p, target_entropy):
    log_p = jax.lax.stop_gradient(log_p)
    def loss_fn(log_alpha):
        partial_loss_fn = jax.vmap(partial(alpha_loss_fn, log_alpha(), target_entropy))
        return jnp.mean(partial_loss_fn(log_p))
    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


class SAC():
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 discount=0.99,
                 tau=0.005,
                 policy_freq=2,
                 lr=3e-4,
                 entropy_tune=True,
                 seed=0):

        self.rng = PRNGSequence(seed)

        actor_input_dim = [((1, state_dim), jnp.float32)]

        actor = build_gaussian_policy_model(actor_input_dim, action_dim, max_action, next(self.rng))
        actor_optimizer = optim.Adam(learning_rate=lr).create(actor)
        self.actor_optimizer = jax.device_put(actor_optimizer)

        init_rng = next(self.rng)

        critic_input_dim = [((1, state_dim), jnp.float32),
                            ((1, action_dim), jnp.float32)]

        critic = build_double_critic_model(critic_input_dim, init_rng)
        self.critic_target = build_double_critic_model(critic_input_dim, init_rng)
        critic_optimizer = optim.Adam(learning_rate=lr).create(critic)
        self.critic_optimizer = jax.device_put(critic_optimizer)

        self.entropy_tune = entropy_tune

        log_alpha = build_constant_model(-3.5, next(self.rng))
        log_alpha_optimizer = optim.Adam(learning_rate=lr).create(log_alpha)
        self.log_alpha_optimizer = jax.device_put(log_alpha_optimizer)
        self.target_entropy = -action_dim

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq

        self.total_it = 0

    @property
    def target_params(self):
        return (self.discount, self.max_action, self.actor_optimizer.target,
                self.critic_target, self.log_alpha_optimizer.target)

    def select_action(self, state):
        mean, _ = apply_model(self.actor_optimizer.target, state)
        return mean.flatten()

    def sample_action(self, rng, state):
        mean, sig = apply_model(self.actor_optimizer.target, state)
        return sample_from_multivariate_normal(rng, mean, sig)

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        buffer_out = replay_buffer.sample(next(self.rng), batch_size)

        target_Q = jax.lax.stop_gradient(get_td_target(next(self.rng),
                                                       *buffer_out,
                                                       *self.target_params))

        state, action, _, _, _ = buffer_out

        self.critic_optimizer = critic_step(self.critic_optimizer,
                                            state, action, target_Q)

        if self.total_it % self.policy_freq == 0:

            self.actor_optimizer, log_p = actor_step(next(self.rng),
                                                     self.actor_optimizer,
                                                     self.critic_optimizer,
                                                     state,
                                                     self.log_alpha_optimizer)

            if self.entropy_tune:
                self.log_alpha_optimizer = alpha_step(self.log_alpha_optimizer,
                                                  log_p, self.target_entropy)

            self.critic_target = copy_params(self.critic_optimizer.target,
                                             self.critic_target, self.tau)

    def save(self, filename):
        save_model(filename + "_critic", self.critic_optimizer)
        save_model(filename + "_actor", self.actor_optimizer)

    def load(self, filename):
        self.critic_optimizer = load_model(filename + "_critic",
                                           self.critic_optimizer)
        self.critic_optimizer = jax.device_put(self.critic_optimizer)
        self.critic_target = self.critic_target.replace(params=self.critic_optimizer.target.params)

        self.actor_optimizer = load_model(filename + "_actor",
                                          self.actor_optimizer)
        self.actor_optimizer = jax.device_put(self.actor_optimizer)
