from haiku import PRNGSequence
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import random
import numpy as onp


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(2e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = onp.zeros((max_size, state_dim))
        self.action = onp.zeros((max_size, action_dim))
        self.next_state = onp.zeros((max_size, state_dim))
        self.reward = onp.zeros((max_size, 1))
        self.not_done = onp.zeros((max_size, 1))

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


@jax.jit
def copy_params(model: nn.Module, model_target: nn.Module, tau: float) -> nn.Module:
    update_params = jax.tree_multimap(
        lambda m1, mt: tau * m1 + (1 - tau) * mt, model.params, model_target.params
    )

    return model_target.replace(params=update_params)


@jax.vmap
def double_mse(q1: jnp.ndarray, q2: jnp.ndarray, qt: jnp.ndarray) -> float:
    return jnp.square(qt - q1).mean() + jnp.square(qt - q2).mean()


@jax.vmap
def mse(pred: jnp.ndarray, true: jnp.ndarray) -> float:
    return jnp.square(true - pred).mean()


@jax.jit
def apply_model(model: nn.Module, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
    return model(x.reshape(1, -1), *args, **kwargs)


@jax.jit
def sample_from_multivariate_normal(
    rng: PRNGSequence, mean: jnp.ndarray, cov: jnp.ndarray, shape: tuple = None
) -> jnp.ndarray:
    return random.multivariate_normal(rng, mean, cov, shape=shape)


@jax.jit
def gaussian_likelihood(sample: jnp.ndarray, mu: float, log_sig: float):
    pre_sum = -0.5 * (
        ((sample - mu) / (jnp.exp(log_sig) + 1e-6)) ** 2
        + 2 * log_sig
        + jnp.log(2 * onp.pi)
    )
    return jnp.sum(pre_sum, axis=1)


@jax.vmap
def kl_mvg_diag(pm, pv, qm, qv):
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
