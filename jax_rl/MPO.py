from functools import partial
from typing import Any
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import optim
from flax.core.frozen_dict import FrozenDict
from haiku import PRNGSequence
from jax import random
from jax.experimental.optimizers import clip_grads
from jax.scipy.special import logsumexp
from jax_rl.buffers import ReplayBuffer
from jax_rl.models import apply_constant_model
from jax_rl.models import apply_double_critic_model
from jax_rl.models import apply_gaussian_policy_model
from jax_rl.models import build_constant_model
from jax_rl.models import build_double_critic_model
from jax_rl.models import build_gaussian_policy_model
from jax_rl.saving import load_model
from jax_rl.saving import save_model
from jax_rl.utils import double_mse
from jax_rl.utils import gaussian_likelihood
from jax_rl.utils import kl_mvg_diag
from scipy.optimize import minimize


def set_frozen_dict(frozen_dict: FrozenDict, key: str, value: Any) -> FrozenDict:
    unfrozen_dict = frozen_dict.unfreeze()
    unfrozen_dict[key] = value
    return FrozenDict(**unfrozen_dict)


@partial(jax.jit, static_argnums=(6, 7, 8))
def get_td_target(
    rng: PRNGSequence,
    state: jnp.ndarray,
    action: jnp.ndarray,
    next_state: jnp.ndarray,
    reward: jnp.ndarray,
    not_done: jnp.ndarray,
    discount: float,
    max_action: float,
    action_dim: int,
    actor_target_params: FrozenDict,
    critic_target_params: FrozenDict,
) -> jnp.ndarray:
    mu, log_sig = apply_gaussian_policy_model(
        actor_target_params, action_dim, max_action, next_state, None, False, True
    )
    next_action = mu + jnp.exp(log_sig) * random.normal(rng, mu.shape)
    next_action = max_action * nn.tanh(next_action)

    target_Q1, target_Q2 = apply_double_critic_model(
        critic_target_params, next_state, next_action, False
    )
    target_Q = jnp.minimum(target_Q1, target_Q2)
    target_Q = reward + not_done * discount * target_Q

    return target_Q


@jax.jit
def mu_lagrange_step(optimizer: optim.Optimizer, reg: float) -> optim.Optimizer:
    def loss_fn(mu_lagrange_params):
        return jnp.sum(apply_constant_model(mu_lagrange_params, 1.0, True) * reg)

    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


@jax.jit
def sig_lagrange_step(optimizer: optim.Optimizer, reg: float) -> optim.Optimizer:
    def loss_fn(sig_lagrange_params):
        return jnp.sum(apply_constant_model(sig_lagrange_params, 100.0, True) * reg)

    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


@partial(jax.jit, static_argnums=1)
def dual(Q1: jnp.ndarray, eps_eta: float, temp: jnp.ndarray) -> float:
    """
    Dual function of the non-parametric variational distribution using samples.
    g(η) = η*ε + η \\mean \\log (\\mean \\exp(Q(a, s)/η))
    """
    out = temp * (
        eps_eta + jnp.mean(logsumexp(Q1 / temp, axis=1)) - jnp.log(Q1.shape[1])
    )
    return out.sum()


@partial(jax.jit, static_argnums=(3, 4, 6, 7))
def sample_actions_and_evaluate(
    rng: PRNGSequence,
    actor_target_params: FrozenDict,
    critic_target_params: FrozenDict,
    max_action: float,
    action_dim: int,
    state: jnp.ndarray,
    batch_size: int,
    action_sample_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    To build our nonparametric policy, q(s, a), we sample `action_sample_size`
    actions from each policy in the batch and evaluate their Q-values.
    """
    # get the policy distribution for each state and sample `action_sample_size`
    # actions from each
    mu, log_sig = apply_gaussian_policy_model(
        actor_target_params, action_dim, max_action, state, None, False, True
    )
    mu = jnp.expand_dims(mu, axis=1)
    sig = jnp.expand_dims(jnp.exp(log_sig), axis=1)
    sampled_actions = (
        mu + random.normal(rng, (batch_size, action_sample_size, action_dim)) * sig
    )
    sampled_actions = sampled_actions.reshape(
        (batch_size * action_sample_size, action_dim)
    )

    sampled_actions = jax.lax.stop_gradient(sampled_actions)

    states_repeated = jnp.repeat(state, action_sample_size, axis=0)

    # evaluate each of the sampled actions at their corresponding state
    # we keep the `sampled_actions` array unnquashed because we need to calcuate
    # the log probabilities using it, but we pass the squashed actions to the critic
    Q1 = apply_double_critic_model(
        critic_target_params,
        states_repeated,
        max_action * nn.tanh(sampled_actions),
        True,
    )
    Q1 = Q1.reshape((batch_size, action_sample_size))

    Q1 = jax.lax.stop_gradient(Q1)

    return Q1, sampled_actions


def e_step(
    rng: PRNGSequence,
    actor_target_params: FrozenDict,
    critic_target_params: FrozenDict,
    max_action: float,
    action_dim: int,
    temp: float,
    eps_eta: float,
    state: jnp.ndarray,
    batch_size: int,
    action_sample_size: int,
) -> Tuple[optim.Optimizer, jnp.ndarray, jnp.ndarray]:
    """
    The 'E-step' from the MPO paper. We calculate our weights, which correspond
    to the relative likelihood of obtaining the maximum reward for each of the
    sampled actions. We also take steps on our temperature parameter, which
    induces diversity in the weights.
    """
    Q1, sampled_actions = sample_actions_and_evaluate(
        rng,
        actor_target_params,
        critic_target_params,
        max_action,
        action_dim,
        state,
        batch_size,
        action_sample_size,
    )

    jac = jax.grad(dual, argnums=2)
    jac = partial(jac, Q1, eps_eta)

    # use nonconvex optimizer to minimize the dual of the temperature parameter
    # we have direct access to the jacobian function with jax so we can take
    # advantage of it here
    this_dual = partial(dual, Q1, eps_eta)
    bounds = [(1e-6, None)]
    res = minimize(this_dual, temp, jac=jac, method="SLSQP", bounds=bounds)
    temp = jax.lax.stop_gradient(res.x)

    # calculate the sample-based q distribution. we can think of these weights
    # as the relative likelihood of each of the sampled actions giving us the
    # maximum score when taken at the corresponding state.
    weights = jax.nn.softmax(Q1 / temp, axis=1)
    weights = jax.lax.stop_gradient(weights)
    weights = jnp.expand_dims(weights, axis=-1)

    return temp, weights, sampled_actions


@partial(jax.jit, static_argnums=(3, 4, 7, 8))
def m_step(
    rngs: PRNGSequence,
    actor_optimizer: optim.Optimizer,
    actor_target_params: FrozenDict,
    eps_mu: float,
    eps_sig: float,
    mu_lagrange_optimizer: optim.Optimizer,
    sig_lagrange_optimizer: optim.Optimizer,
    max_action: float,
    action_dim: int,
    state: jnp.ndarray,
    weights: jnp.ndarray,
    sampled_actions: jnp.ndarray,
) -> Tuple[optim.Optimizer, optim.Optimizer, optim.Optimizer]:
    """
    The 'M-step' from the MPO paper. We optimize our policy network to maximize
    the lower bound on the probablility of obtaining the maximum reward given
    that we act according to our policy (i.e. weighted according to our sampled actions).
    """

    def loss_fn(mlo, slo, actor_params):
        # get the distribution of the actor network (current policy)
        mu, log_sig = apply_gaussian_policy_model(
            actor_params, action_dim, max_action, state, None, False, True
        )
        sig = jnp.exp(log_sig)
        # get the distribution of the target network (old policy)
        target_mu, target_log_sig = apply_gaussian_policy_model(
            actor_target_params, action_dim, max_action, state, None, False, True
        )
        target_mu = jax.lax.stop_gradient(target_mu)
        target_log_sig = jax.lax.stop_gradient(target_log_sig)
        target_sig = jnp.exp(target_log_sig)

        # get the log likelihooods of the sampled actions according to the
        # decoupled distributions. described in section 4.2.1 of
        # Relative Entropy Regularized Policy Iteration
        # this ensures that the nonparametric policy won't collapse to give
        # a probability of 1 to the best action, which is a risk when we use
        # the on-policy distribution to calculate the likelihood.
        actor_log_prob = gaussian_likelihood(sampled_actions, target_mu, log_sig)
        actor_log_prob += gaussian_likelihood(sampled_actions, mu, target_log_sig)
        actor_log_prob = actor_log_prob.transpose((0, 1))

        mu_kl = kl_mvg_diag(target_mu, target_sig, mu, target_sig).mean()
        sig_kl = kl_mvg_diag(target_mu, target_sig, target_mu, sig).mean()

        mlo = mu_lagrange_step(mlo, eps_mu - jax.lax.stop_gradient(mu_kl))
        slo = sig_lagrange_step(slo, eps_sig - jax.lax.stop_gradient(sig_kl))

        # maximize the log likelihood, regularized by the divergence between
        # the target policy and the current policy. the goal here is to fit
        # the parametric policy to have the minimum divergence with the nonparametric
        # distribution based on the sampled actions.
        actor_loss = -(actor_log_prob * weights).sum(axis=1).mean()
        actor_loss -= jax.lax.stop_gradient(
            apply_constant_model(mlo.target, 1.0, True)
        ) * (eps_mu - mu_kl)
        actor_loss -= jax.lax.stop_gradient(
            apply_constant_model(slo.target, 100.0, True)
        ) * (eps_sig - sig_kl)
        return actor_loss.mean(), (mlo, slo)

    grad, (mu_lagrange_optimizer, sig_lagrange_optimizer) = jax.grad(
        partial(loss_fn, mu_lagrange_optimizer, sig_lagrange_optimizer), has_aux=True
    )(actor_optimizer.target)
    grad = clip_grads(grad, 40.0)

    actor_optimizer = actor_optimizer.apply_gradient(grad)

    return mu_lagrange_optimizer, sig_lagrange_optimizer, actor_optimizer


@jax.jit
def critic_step(
    optimizer: optim.Optimizer,
    state: jnp.ndarray,
    action: jnp.ndarray,
    target_Q: jnp.ndarray,
) -> optim.Optimizer:
    """
    The critic is optimized the same way as typical actor critic methods,
    minimizing the TD error.
    """

    def loss_fn(critic_params):
        current_Q1, current_Q2 = apply_double_critic_model(
            critic_params, state, action, False
        )
        critic_loss = double_mse(current_Q1, current_Q2, target_Q)
        return critic_loss.mean()

    grad = jax.grad(loss_fn)(optimizer.target)
    grad = clip_grads(grad, 40.0)
    return optimizer.apply_gradient(grad)


class MPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        discount: float = 0.99,
        lr: float = 3e-4,
        eps_eta: float = 0.1,
        eps_mu: float = 5e-4,
        eps_sig: float = 1e-5,
        target_freq: int = 250,
        seed: int = 0,
    ):
        self.rng = PRNGSequence(seed)

        init_rng = next(self.rng)

        actor_input_dim = (1, state_dim)

        actor_params = build_gaussian_policy_model(
            actor_input_dim, action_dim, max_action, init_rng
        )
        self.actor_target_params = build_gaussian_policy_model(
            actor_input_dim, action_dim, max_action, init_rng
        )
        actor_optimizer = optim.Adam(learning_rate=lr).create(actor_params)
        self.actor_optimizer = jax.device_put(actor_optimizer)

        init_rng = next(self.rng)

        critic_input_dim = [(1, state_dim), (1, action_dim)]

        critic_params = build_double_critic_model(critic_input_dim, init_rng)
        self.critic_target_params = build_double_critic_model(
            critic_input_dim, init_rng
        )
        critic_optimizer = optim.Adam(learning_rate=lr).create(critic_params)
        self.critic_optimizer = jax.device_put(critic_optimizer)

        mu_lagrange_params = build_constant_model(
            1.0, absolute=True, init_rng=next(self.rng)
        )
        mu_lagrange_optimizer = optim.Adam(learning_rate=lr).create(mu_lagrange_params)
        self.mu_lagrange_optimizer = jax.device_put(mu_lagrange_optimizer)

        sig_lagrange_params = build_constant_model(
            100.0, absolute=True, init_rng=next(self.rng)
        )
        sig_lagrange_optimizer = optim.Adam(learning_rate=lr).create(
            sig_lagrange_params
        )
        self.sig_lagrange_optimizer = jax.device_put(sig_lagrange_optimizer)

        self.temp = 1.0
        self.eps_eta = eps_eta
        self.eps_mu = eps_mu
        self.eps_sig = eps_sig

        self.max_action = max_action
        self.discount = discount
        self.target_freq = target_freq

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.total_it = 0

    @property
    def td_params(self):
        return (
            self.discount,
            self.max_action,
            self.action_dim,
            self.actor_target_params,
            self.critic_target_params,
        )

    @property
    def e_params(self):
        return (
            self.actor_target_params,
            self.critic_target_params,
            self.max_action,
            self.action_dim,
            self.temp,
            self.eps_eta,
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
            self.max_action,
            self.action_dim,
        )

    def select_action(self, state: jnp.ndarray) -> jnp.ndarray:
        mu, _ = apply_gaussian_policy_model(
            self.actor_optimizer.target,
            self.action_dim,
            self.max_action,
            state.reshape(1, -1),
            None,
            False,
            True,
        )
        return mu

    def sample_action(self, rng: PRNGSequence, state: jnp.ndarray) -> jnp.ndarray:
        mu, log_sig = apply_gaussian_policy_model(
            self.actor_optimizer.target,
            self.action_dim,
            self.max_action,
            state.reshape(1, -1),
            None,
            False,
            True,
        )
        sig = jnp.exp(log_sig)
        return mu + random.normal(rng, mu.shape) * sig

    def train(
        self, replay_buffer: ReplayBuffer, batch_size: int, action_sample_size: int
    ):
        self.total_it += 1

        buffer_out = replay_buffer.sample(next(self.rng), batch_size)

        target_Q = jax.lax.stop_gradient(
            get_td_target(next(self.rng), *buffer_out, *self.td_params)
        )

        state, action, *_ = buffer_out

        self.critic_optimizer = critic_step(
            self.critic_optimizer, state, action, target_Q
        )

        self.temp, weights, sampled_actions = e_step(
            next(self.rng), *self.e_params, state, batch_size, action_sample_size
        )

        sampled_actions = sampled_actions.reshape(
            (batch_size, action_sample_size, self.action_dim)
        )

        rngs = [next(self.rng) for _ in range(3)]

        (
            self.mu_lagrange_optimizer,
            self.sig_lagrange_optimizer,
            self.actor_optimizer,
        ) = m_step(rngs, *self.m_params, state, weights, sampled_actions)

        if self.total_it % self.target_freq == 0:
            self.actor_target_params = self.actor_optimizer.target.copy()
            self.critic_target_params = self.critic_optimizer.target.copy()

    def save(self, filename):
        save_model(filename + "_critic", self.critic_optimizer)
        save_model(filename + "_actor", self.actor_optimizer)

    def load(self, filename):
        self.critic_optimizer = load_model(filename + "_critic", self.critic_optimizer)
        self.critic_optimizer = jax.device_put(self.critic_optimizer)
        self.critic_target_params = self.critic_optimizer.target.copy()

        self.actor_optimizer = load_model(filename + "_actor", self.actor_optimizer)
        self.actor_optimizer = jax.device_put(self.actor_optimizer)
        self.actor_target_params = self.actor_optimizer.target.copy()
