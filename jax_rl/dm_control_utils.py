import numpy as onp
from dm_control import suite


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
