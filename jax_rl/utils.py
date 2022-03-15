import jax
import jax.numpy as jnp
import numpy as onp
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from haiku import PRNGSequence
from jax import random


@jax.jit
def copy_params(
    orig_params: FrozenDict, target_params: FrozenDict, tau: float
) -> nn.Module:
    """
    Applies polyak averaging between two sets of parameters.
    """
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
@jax.vmap
def gaussian_likelihood(
    sample: jnp.ndarray, mu: jnp.ndarray, log_sig: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculates the log likelihood of a sample from a Gaussian distribution.
    i.e. the log of the pdf evaluated at `sample`

    Args:
        sample (jnp.ndarray): an array of samples from the distribution
        mu (jnp.ndarray): the mean of the distribution
        log_sig (jnp.ndarray): the log of the standard deviation of the distribution

    Returns:
        the log likelihood of the sample
    """
    return -0.5 * (
        ((sample - mu) / (jnp.exp(log_sig) + 1e-6)) ** 2
        + 2 * log_sig
        + jnp.log(2 * onp.pi)
    )


@jax.vmap
def kl_mvg_diag(
    pm: jnp.ndarray, pv: jnp.ndarray, qm: jnp.ndarray, qv: jnp.ndarray
) -> jnp.ndarray:
    """
    Kullback-Leibler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.

    Args:
        pm: mean of starting distribution
        pv: standard deviation of starting distribution
        qm: mean of target distribution
        qv: standard deviation of target distribution

    Returns:
        KL divergence from p to q
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
