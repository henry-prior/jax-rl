from flax import nn
from jax import random
import jax.numpy as jnp


class TD3_Actor(nn.Module):
    def apply(self, x, action_dim, max_action):
        x = nn.Dense(x, features=256)
        x = nn.relu(x)
        x = nn.Dense(x, features=256)
        x = nn.relu(x)
        x = nn.Dense(x, features=action_dim)
        return max_action * nn.tanh(x)


class TD3_Critic(nn.Module):
    def apply(self, state, action, Q1=False):
        state_action = jnp.concatenate([state, action], axis=1)

        q1 = nn.Dense(state_action, features=256)
        q1 = nn.relu(q1)
        q1 = nn.Dense(q1, features=256)
        q1 = nn.relu(q1)
        q1 = nn.Dense(q1, features=1)

        if Q1: return q1

        q2 = nn.Dense(state_action, features=256)
        q2 = nn.relu(q2)
        q2 = nn.Dense(q2, features=256)
        q2 = nn.relu(q2)
        q2 = nn.Dense(q2, features=1)

        return q1, q2


def build_td3_actor_model(input_shapes, action_dim, max_action, init_rng):
    actor = TD3_Actor.partial(action_dim=action_dim, max_action=max_action)
    _, init_params = actor.init_by_shape(init_rng, input_shapes)

    return nn.Model(actor, init_params)


def build_td3_critic_model(input_shapes, init_rng):
    critic = TD3_Critic.partial()
    _, init_params = critic.init_by_shape(init_rng, input_shapes)

    return nn.Model(critic, init_params)
