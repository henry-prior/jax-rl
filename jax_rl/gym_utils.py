import gym
import numpy as np


def optional_squeeze(array: np.ndarray, dim: int = -1) -> np.ndarray:
    if array.shape[dim] == 1:
        return array.squeeze(dim)
    return array


def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(state)
            action = optional_squeeze(action)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += float(reward)

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward
