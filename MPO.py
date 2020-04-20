from functools import partial
import jax
from jax import random
from jax.scipy.special import logsumexp
import jax.numpy as jnp
from flax import optim
from flax import nn
from haiku import PRNGSequence

from saving import save_model, load_model
from utils import (apply_model,
                   double_mse,
                   gaussian_likelihood,
                   kl_mvg_diag)
from models import (build_gaussian_policy_model,
                    build_double_critic_model,
                    build_constant_model)


@jax.jit
def get_td_target(rng, state, action, next_state, reward, not_done,
                  discount, max_action, actor_target, critic_target):

    mu, var = actor_target(next_state, MPO=True)
    next_action = mu + var * random.normal(rng, mu.shape)
    next_action = max_action * jnp.tanh(next_action)

    target_Q1, target_Q2 = critic_target(next_state, next_action)
    target_Q = jnp.minimum(target_Q1, target_Q2)
    target_Q = reward + not_done * discount * target_Q

    return target_Q


@jax.jit
def temp_step(optimizer, Q, eps_q, action_sample_size):
    def loss_fn(temp):
        lse = logsumexp(Q / temp(), axis=1) - jnp.log(action_sample_size)
        temp_loss = temp() * (eps_q + lse.mean())
        return jnp.mean(temp_loss)
    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


@jax.jit
def lagrange_step(optimizer, reg):
    def loss_fn(lagrange):
        return jnp.mean(lagrange() * reg)
    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


@jax.jit
def actor_step(optimizer, weights, log_p, mu_lagrange, reg_mu, sig_lagrange, reg_sig):
    mu_lagrange, sig_lagrange = mu_lagrange.target, sig_lagrange.target
    def loss_fn(actor):
        actor_loss = -(jax.vmap(jnp.multiply)(weights, log_p)).sum(axis=1).mean()
        actor_loss -= mu_lagrange() * reg_mu
        actor_loss -= sig_lagrange() * reg_sig
        return actor_loss.mean()
    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


@jax.partial(jax.jit, static_argnums=(4, 6, 7, 9, 10))
def e_step(rng, actor_target, critic_target, max_action, action_dim,
           temp_optimizer, eps_q, temp_steps, state, batch_size, action_sample_size):
    mu, log_sig = actor_target(state, MPO=True)
    sig = jnp.exp(log_sig)
    sampled_actions = mu + random.normal(rng, (mu.shape[0], action_sample_size)) * sig
    sampled_actions = jnp.clip(sampled_actions, -max_action, max_action).transpose((0,1))
    sampled_actions = sampled_actions.reshape((batch_size * action_sample_size, action_dim))
    sampled_actions = max_action * jnp.tanh(sampled_actions)

    states_repeated = jnp.tile(state, (action_sample_size, 1))

    Q1 = critic_target(states_repeated, sampled_actions, Q1=True)
    Q1 = Q1.reshape((batch_size, action_sample_size))

    Q1 = jax.lax.stop_gradient(Q1)

    for _ in range(temp_steps):
        temp_optimizer = temp_step(temp_optimizer, Q1, eps_q, action_sample_size)
        temp_optimizer.target.params['value'] = jnp.abs(temp_optimizer.target.params['value'])

    Z = jnp.sum(jnp.exp(Q1 - jnp.max(Q1, axis=1)[0]) / temp_optimizer.target(), axis=1)[:, None]
    weights = jnp.exp((Q1 - jnp.max(Q1, axis=1)[0]) / temp_optimizer.target()) / Z
    weights = jax.lax.stop_gradient(weights)

    return temp_optimizer, weights, sampled_actions


@jax.jit
def m_step(rngs, actor_optimizer, actor_target, eps_mu, eps_sig,
           mu_lagrange_optimizer, sig_lagrange_optimizer, state, weights,
           sampled_actions):

    def loss_fn(mlo, slo, actor):
        mu, log_sig = actor(state, MPO=True)
        sig = jnp.exp(log_sig)
        target_mu, target_log_sig = actor_target(state, MPO=True)
        target_sig = jnp.exp(target_log_sig)

        actor_log_prob = gaussian_likelihood(sampled_actions, target_mu.squeeze(), sig.squeeze())
        actor_log_prob += gaussian_likelihood(sampled_actions, mu.squeeze(), target_sig.squeeze())
        actor_log_prob = actor_log_prob.transpose((0,1))

        mu, target_mu = nn.tanh(mu), nn.tanh(mu)

        reg_mu = eps_mu - kl_mvg_diag(target_mu, target_sig, mu, target_sig).mean()
        reg_sig = eps_sig - kl_mvg_diag(target_mu, target_sig, target_mu, sig).mean()

        mlo = lagrange_step(mlo, reg_mu)
        slo = lagrange_step(slo, reg_sig)

        actor_loss = -(jnp.dot(actor_log_prob, weights)).sum(axis=1).mean()
        actor_loss -= mu_lagrange_optimizer.target() * reg_mu
        actor_loss -= sig_lagrange_optimizer.target() * reg_sig
        return actor_loss.mean(), (mlo, slo)

    grad, optims = jax.grad(partial(loss_fn, mu_lagrange_optimizer, sig_lagrange_optimizer), has_aux=True)(actor_optimizer.target)
    mu_lagrange_optimizer, sig_lagrange_optimizer = optims

    actor_optimizer = actor_optimizer.apply_gradient(grad)

    return mu_lagrange_optimizer, sig_lagrange_optimizer, actor_optimizer


@jax.jit
def critic_step(optimizer, state, action, target_Q):
    def loss_fn(critic):
        current_Q1, current_Q2 = critic(state, action)
        critic_loss = double_mse(current_Q1, current_Q2, target_Q)
        return jnp.mean(critic_loss)
    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


class MPO():
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 discount=0.99,
                 lr=3e-4,
                 eps_q=0.1,
                 eps_mu=5e-4,
                 eps_sig=1e-5,
                 temp_steps=3,
                 target_freq=250,
                 seed=0):

        self.rng = PRNGSequence(seed)

        init_rng = next(self.rng)

        actor_input_dim = [((1, state_dim), jnp.float32)]

        actor = build_gaussian_policy_model(actor_input_dim, action_dim, max_action, init_rng)
        self.actor_target = build_gaussian_policy_model(actor_input_dim, action_dim, max_action, init_rng)
        actor_optimizer = optim.Adam(learning_rate=lr).create(actor)
        self.actor_optimizer = jax.device_put(actor_optimizer)

        init_rng = next(self.rng)

        critic_input_dim = [((1, state_dim), jnp.float32),
                            ((1, action_dim), jnp.float32)]

        critic = build_double_critic_model(critic_input_dim, init_rng)
        self.critic_target = build_double_critic_model(critic_input_dim, init_rng)
        critic_optimizer = optim.Adam(learning_rate=lr).create(critic)
        self.critic_optimizer = jax.device_put(critic_optimizer)

        temp = build_constant_model(1.0, next(self.rng))
        temp_optimizer = optim.Adam(learning_rate=lr).create(temp)
        self.temp_optimizer = jax.device_put(temp_optimizer)

        mu_lagrange = build_constant_model(1.0, next(self.rng))
        mu_lagrange_optimizer = optim.Adam(learning_rate=lr).create(mu_lagrange)
        self.mu_lagrange_optimizer = jax.device_put(mu_lagrange_optimizer)
        sig_lagrange = build_constant_model(100.0, next(self.rng))
        sig_lagrange_optimizer = optim.Adam(learning_rate=lr).create(sig_lagrange)
        self.sig_lagrange_optimizer = jax.device_put(sig_lagrange_optimizer)

        self.eps_q = eps_q
        self.eps_mu = eps_mu
        self.eps_sig = eps_sig
        self.temp_steps = temp_steps

        self.max_action = max_action
        self.discount = discount
        self.target_freq = target_freq

        self.action_dim = action_dim

        self.total_it = 0

    @property
    def td_params(self):
        return self.discount, self.max_action, self.actor_target, self.critic_target

    @property
    def e_params(self):
        return (self.actor_target, self.critic_target, self.max_action,
                self.action_dim, self.temp_optimizer, self.eps_q, self.temp_steps)

    @property
    def m_params(self):
        return (self.actor_optimizer, self.actor_target, self.eps_mu,
                self.eps_sig, self.mu_lagrange_optimizer, self.sig_lagrange_optimizer)

    def select_action(self, state):
        mu, _ = apply_model(self.actor_optimizer.target, state)
        return mu

    def sample_action(self, rng, state):
        mu, log_sig = apply_model(self.actor_optimizer.target, state)
        sig = jnp.abs(log_sig)
        return mu + random.normal(rng, mu.shape) * sig

    def train(self, replay_buffer, batch_size, action_sample_size):
        self.total_it += 1

        buffer_out = replay_buffer.sample(next(self.rng), batch_size)

        target_Q = jax.lax.stop_gradient(get_td_target(next(self.rng),
                                                       *buffer_out,
                                                       *self.td_params))

        state, action, _, _, _ = buffer_out

        self.critic_optimizer = critic_step(self.critic_optimizer,
                                            state, action, target_Q)

        self.temp_optimizer, weights, sampled_actions = e_step(next(self.rng),
                                                               *self.e_params,
                                                               state,
                                                               batch_size,
                                                               action_sample_size)

        weights, sampled_actions = list(map(jax.lax.stop_gradient, [weights, sampled_actions]))
        sampled_actions = sampled_actions.reshape((batch_size, action_sample_size, self.action_dim)).transpose((0,1))

        rngs = [next(self.rng) for _ in range(3)]

        self.mu_lagrange_optimizer, self.sig_lagrange_optimizer, self.actor_optimizer = \
            m_step(rngs, *self.m_params, state, weights, sampled_actions)

        if self.total_it % self.target_freq == 0:
            self.actor_target = self.actor_target.replace(params=self.actor_optimizer.target.params)
            self.critic_target = self.critic_target.replace(params=self.critic_optimizer.target.params)

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
        self.actor_target = self.actor_target.replace(params=self.actor_optimizer.target.params)
