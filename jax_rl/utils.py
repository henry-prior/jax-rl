import jax
import jax.numpy as jnp
import numpy as onp
from dm_control import suite
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from haiku import PRNGSequence
from jax import random


def flat_obs(o):
    return onp.concatenate([o[k].flatten() for k in o])


def eval_policy(policy, domain_name, task_name, seed, eval_episodes=10):
    eval_env = suite.load(domain_name, task_name, {"random": seed + 100})

    avg_reward = 0.0
    for _ in range(eval_episodes):
        timestep = eval_env.reset()
        while not timestep.last():
            action = policy.select_action(flat_obs(timestep.observation))
            timestep = eval_env.step(action)
            avg_reward += timestep.reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


@jax.jit
def copy_params(
    orig_params: FrozenDict, target_params: FrozenDict, tau: float
) -> nn.Module:
    update_params = jax.tree_multimap(
        lambda m1, mt: tau * m1 + (1 - tau) * mt, orig_params, target_params,
    )

    return update_params


@jax.vmap
def double_mse(q1: jnp.ndarray, q2: jnp.ndarray, qt: jnp.ndarray) -> float:
    return jnp.square(qt - q1).mean() + jnp.square(qt - q2).mean()


@jax.vmap
def mse(pred: jnp.ndarray, true: jnp.ndarray) -> float:
    return jnp.square(true - pred).mean()


@jax.jit
def sample_from_multivariate_normal(
    rng: PRNGSequence, mean: jnp.ndarray, cov: jnp.ndarray, shape: tuple = None
) -> jnp.ndarray:
    return random.multivariate_normal(rng, mean, cov, shape=shape)


@jax.jit
def gaussian_likelihood(
    sample: jnp.ndarray, mu: jnp.ndarray, log_sig: jnp.ndarray
) -> jnp.ndarray:
    pre_sum = -0.5 * (
        ((sample - mu) / (jnp.exp(log_sig) + 1e-6)) ** 2
        + 2 * log_sig
        + jnp.log(2 * onp.pi)
    )
    return jnp.sum(pre_sum, axis=1)


@jax.vmap
def kl_mvg_diag(
    pm: jnp.ndarray, pv: jnp.ndarray, qm: jnp.ndarray, qv: jnp.ndarray
) -> jnp.ndarray:
    """
    Kullback-Leibler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    """
    if len(qm.shape) == 2:
        axis = 1
    else:
        axis = 0
    # Determinants of diagonal covariances pv, qv
    dpv = pv.prod()
    dqv = qv.prod(axis)
    # Inverse of diagonal covariance qv
    iqv = 1.0 / qv
    # Difference between means pm, qm
    diff = qm - pm
    return 0.5 * (
        jnp.log(dqv / dpv)  # log |\Sigma_q| / |\Sigma_p|
        + (iqv * pv).sum(axis)  # + tr(\Sigma_q^{-1} * \Sigma_p)
        + (diff * iqv * diff).sum(axis)  # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
        - len(pm)
    )
