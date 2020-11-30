from functools import partial
import jax
from jax import random
from jax.scipy.special import logsumexp
import jax.numpy as jnp
import numpy as onp
from flax.core.frozen_dict import FrozenDict
from flax import optim
from flax import linen as nn
from haiku import PRNGSequence

from jax_rl.saving import save_model
from jax_rl.saving import load_model
from jax_rl.utils import apply_model
from jax_rl.utils import double_mse
from jax_rl.utils import gaussian_likelihood
from jax_rl.utils import kl_mvg_diag
from jax_rl.models import build_gaussian_policy_model
from jax_rl.models import build_double_critic_model
from jax_rl.models import build_constant_model


@jax.jit
def get_td_target(
    rng: PRNGSequence,
    state: jnp.ndarray,
    action: jnp.ndarray,
    next_state: jnp.ndarray,
    reward: jnp.ndarray,
    not_done: jnp.ndarray,
    discount: float,
    max_action: float,
    actor_target_params: FrozenDict,
    critic_target_params: FrozenDict,
) -> jnp.ndarray:

    mu, var = apply_model(actor, actor_target_params, next_state, MPO=True)
    next_action = mu + var * random.normal(rng, mu.shape)
    next_action = max_action * jnp.tanh(next_action)

    target_Q1, target_Q2 = apply_model(
        critic, critic_target_params, next_state, next_action
    )
    target_Q = jnp.minimum(target_Q1, target_Q2)
    target_Q = reward + not_done * discount * target_Q

    return target_Q


@jax.jit
def temp_step(
    optimizer: optim.Optimizer, Q: jnp.ndarray, eps_q: float, action_sample_size: int
) -> optim.Optimizer:
    def loss_fn(temp_params):
        lse = logsumexp(Q / apply_model(temp, temp_params), axis=1) - jnp.log(
            action_sample_size
        )
        temp_loss = apply_model(temp, temp_params) * (eps_q + lse.mean())
        return jnp.mean(temp_loss)

    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


@jax.jit
def lagrange_step(optimizer: optim.Optimizer, reg: float) -> optim.Optimizer:
    def loss_fn(lagrange_params):
        return jnp.mean(apply_model(lagrange, lagrange_params) * reg)

    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


@jax.jit
def actor_step(
    optimizer: optim.Optimizer,
    weights: jnp.ndarray,
    log_p: jnp.ndarray,
    mu_lagrange_params: optim.Optimizer,
    reg_mu: float,
    sig_lagrange_params: FrozenDict,
    reg_sig: float,
) -> optim.Optimizer:
    def loss_fn(actor):
        actor_loss = -(jax.vmap(jnp.multiply)(weights, log_p)).sum(axis=1).mean()
        actor_loss -= apply_model(mu_lagrange, mu_lagrange_params) * reg_mu
        actor_loss -= apply_model(sig_lagrange, sig_lagrange_params) * reg_sig
        return actor_loss.mean()

    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


@jax.partial(jax.jit, static_argnums=(4, 6, 7, 9, 10))
def e_step(
    rng: PRNGSequence,
    actor_target_params: FrozenDict,
    critic_target_params: FrozenDict,
    max_action: float,
    action_dim: int,
    temp_optimizer: optim.Optimizer,
    eps_q: float,
    temp_steps: int,
    state: jnp.ndarray,
    batch_size: int,
    action_sample_size: int,
) -> optim.Optimizer:
    mu, log_sig = apply_model(actor, actor_target, state, MPO=True)
    sig = jnp.exp(log_sig)
    sampled_actions = mu + random.normal(rng, (mu.shape[0], action_sample_size)) * sig
    sampled_actions = jnp.clip(sampled_actions, -max_action, max_action).transpose(
        (0, 1)
    )
    sampled_actions = sampled_actions.reshape(
        (batch_size * action_sample_size, action_dim)
    )
    sampled_actions = max_action * jnp.tanh(sampled_actions)

    states_repeated = jnp.tile(state, (action_sample_size, 1))

    Q1 = apply_model(critic, critic_target, states_repeated, sampled_actions, Q1=True)
    Q1 = Q1.reshape((batch_size, action_sample_size))

    Q1 = jax.lax.stop_gradient(Q1)

    for _ in range(temp_steps):
        temp_optimizer = temp_step(temp_optimizer, Q1, eps_q, action_sample_size)
        temp_optimizer.target.params["value"] = jnp.maximum(
            0.0, temp_optimizer.target.params["value"]
        )

    Z = jnp.sum(
        jnp.exp(Q1 - jnp.max(Q1, axis=1)[0]) / apply_model(temp, temp_optimizer.target),
        axis=1,
    )[:, None]
    weights = (
        jnp.exp(
            (Q1 - jnp.max(Q1, axis=1)[0]) / apply_model(temp, temp_optimizer.target)
        )
        / Z
    )
    weights = jax.lax.stop_gradient(weights)

    return temp_optimizer, weights, sampled_actions


@jax.jit
def m_step(
    rngs: PRNGSequence,
    actor_optimizer: optim.Optimizer,
    actor_target_params: FrozenDict,
    eps_mu: float,
    eps_sig: float,
    mu_lagrange_optimizer: optim.Optimizer,
    sig_lagrange_optimizer: optim.Optimizer,
    state: jnp.ndarray,
    weights: jnp.ndarray,
    sampled_actions: jnp.ndarray,
) -> optim.Optimizer:
    def loss_fn(mlo, slo, actor_params):
        mu, log_sig = apply_model(actor, actor_params, state, MPO=True)
        sig = jnp.exp(log_sig)
        target_mu, target_log_sig = apply_model(actor, actor_target, state, MPO=True)
        target_sig = jnp.exp(target_log_sig)

        actor_log_prob = gaussian_likelihood(sampled_actions, target_mu, sig)
        actor_log_prob += gaussian_likelihood(sampled_actions, mu, target_sig)
        actor_log_prob = actor_log_prob.transpose((0, 1))

        mu, target_mu = nn.tanh(mu), nn.tanh(mu)

        reg_mu = eps_mu - kl_mvg_diag(target_mu, target_sig, mu, target_sig).mean()
        reg_sig = eps_sig - kl_mvg_diag(target_mu, target_sig, target_mu, sig).mean()

        mlo = lagrange_step(mlo, reg_mu)
        mlo.target.params["value"] = jnp.maximum(0.0, mlo.target.params["value"])
        slo = lagrange_step(slo, reg_sig)
        slo.target.params["value"] = jnp.maximum(0.0, slo.target.params["value"])

        actor_loss = -(actor_log_prob[:, None] * weights).sum(axis=1).mean()
        actor_loss -= apply_model(mu_lagrange, mu_lagrange_optimizer.target) * reg_mu
        actor_loss -= apply_model(sig_lagrange, sig_lagrange_optimizer.target) * reg_sig
        return actor_loss.mean(), (mlo, slo)

    grad, optims = jax.grad(
        partial(loss_fn, mu_lagrange_optimizer, sig_lagrange_optimizer), has_aux=True
    )(actor_optimizer.target)
    mu_lagrange_optimizer, sig_lagrange_optimizer = optims

    actor_optimizer = actor_optimizer.apply_gradient(grad)

    return mu_lagrange_optimizer, sig_lagrange_optimizer, actor_optimizer


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


class MPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        discount: float = 0.99,
        lr: float = 3e-4,
        eps_q: float = 0.1,
        eps_mu: float = 0.1,
        eps_sig: float = 1e-4,
        temp_steps: int = 10,
        target_freq: int = 250,
        seed: int = 0,
    ):
        self.rng = PRNGSequence(seed)

        init_rng = next(self.rng)

        actor_input_dim = (1, state_dim)

        global actor
        actor, actor_params = build_gaussian_policy_model(
            actor_input_dim, action_dim, max_action, init_rng
        )
        _, self.actor_target = build_gaussian_policy_model(
            actor_input_dim, action_dim, max_action, init_rng
        )
        actor_optimizer = optim.Adam(learning_rate=lr).create(actor_params)
        self.actor_optimizer = jax.device_put(actor_optimizer)

        init_rng = next(self.rng)

        critic_input_dim = [(1, state_dim), (1, action_dim)]

        global critic
        critic, critic_params = build_double_critic_model(critic_input_dim, init_rng)
        _, self.critic_target = build_double_critic_model(critic_input_dim, init_rng)
        critic_optimizer = optim.Adam(learning_rate=lr).create(critic_params)
        self.critic_optimizer = jax.device_put(critic_optimizer)

        global temp
        temp, temp_params = build_constant_model(1.0, next(self.rng))
        temp_optimizer = optim.Adam(learning_rate=lr).create(temp_params)
        self.temp_optimizer = jax.device_put(temp_optimizer)

        global mu_lagrange
        mu_lagrange, mu_lagrange_params = build_constant_model(1.0, next(self.rng))
        mu_lagrange_optimizer = optim.Adam(learning_rate=lr).create(mu_lagrange_params)
        self.mu_lagrange_optimizer = jax.device_put(mu_lagrange_optimizer)

        global sig_lagrange
        sig_lagrange, sig_lagrange_params = build_constant_model(100.0, next(self.rng))
        sig_lagrange_optimizer = optim.Adam(learning_rate=lr).create(
            sig_lagrange_params
        )
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
        return (
            self.actor_target_params,
            self.critic_target_params,
            self.max_action,
            self.action_dim,
            self.temp_optimizer,
            self.eps_q,
            self.temp_steps,
        )

    @property
    def m_params(self):
        return (
            self.actor_optimizer,
            self.actor_target_params,
            self.eps_mu,
            self.eps_sig,
            self.mu_lagrange_optimizer,
            self.sig_lagrange_optimizer,
        )

    def select_action(self, state):
        mu, _ = apply_model(actor, self.actor_optimizer.target, state.reshape(1, -1))
        return mu

    def sample_action(self, rng, state):
        mu, log_sig = apply_model(
            actor, self.actor_optimizer.target, state.reshape(1, -1)
        )
        sig = jnp.abs(log_sig)
        return mu + random.normal(rng, mu.shape) * sig

    def train(self, replay_buffer, batch_size, action_sample_size):
        self.total_it += 1

        buffer_out = replay_buffer.sample(next(self.rng), batch_size)

        target_Q = jax.lax.stop_gradient(
            get_td_target(next(self.rng), *buffer_out, *self.td_params)
        )

        state, action, *_ = buffer_out

        self.critic_optimizer = critic_step(
            self.critic_optimizer, state, action, target_Q
        )

        self.temp_optimizer, weights, sampled_actions = e_step(
            next(self.rng), *self.e_params, state, batch_size, action_sample_size
        )

        weights, sampled_actions = list(
            map(jax.lax.stop_gradient, [weights, sampled_actions])
        )
        sampled_actions = (
            sampled_actions.reshape((batch_size, action_sample_size, self.action_dim))
            .transpose((0, 1))
            .squeeze()
        )

        rngs = [next(self.rng) for _ in range(3)]

        (
            self.mu_lagrange_optimizer,
            self.sig_lagrange_optimizer,
            self.actor_optimizer,
        ) = m_step(rngs, *self.m_params, state, weights, sampled_actions)

        if self.total_it % self.target_freq == 0:
            self.actor_target_params = self.actor_target.replace(
                params=self.actor_optimizer.target.params
            )
            self.critic_target_params = self.critic_target.replace(
                params=self.critic_optimizer.target.params
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
