import os
import argparse
import numpy as np
from dm_control import suite

import utils
import TD3, SAC, MPO


def flat_obs(o):
    return np.concatenate([o[k].flatten() for k in o])


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, SAC, or MPO)
    parser.add_argument(
        "--domain_name", default="cartpole"
    )  # DeepMind control suite environment name
    parser.add_argument(
        "--task_name", default="swingup"
    )  # Task name within environment
    parser.add_argument("--train_steps", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)  # Sets DM control and JAX seeds
    parser.add_argument(
        "--start_timesteps", default=1e4, type=int
    )  # Time steps initial random policy is used
    parser.add_argument(
        "--buffer_size", default=2e6, type=int
    )  # Max size of replay buffer
    parser.add_argument(
        "--eval_freq", default=5e3, type=int
    )  # How often (time steps) we evaluate
    parser.add_argument(
        "--max_timesteps", default=1e6, type=int
    )  # Max time steps to run environment
    parser.add_argument(
        "--expl_noise", default=0.1
    )  # Std of Gaussian exploration noise
    parser.add_argument(
        "--batch_size", default=256, type=int
    )  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument(
        "--policy_noise", default=0.2
    )  # Noise added to target policy during critic update
    parser.add_argument(
        "--noise_clip", default=0.5
    )  # Range to clip target policy noise
    parser.add_argument(
        "--policy_freq", default=2, type=int
    )  # Frequency of delayed policy updates
    parser.add_argument(
        "--actor_updates", default=1, type=int
    )  # Number of gradient steps for policy network per update
    parser.add_argument(
        "--save_model", action="store_true"
    )  # Save model and optimizer parameters
    parser.add_argument(
        "--load_model", default=""
    )  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--num_action_samples", default=20, type=int)
    parser.add_argument("--save_freq", default=5e3, type=int)
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.domain_name}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.domain_name}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    if not os.path.exists("./graphs"):
        os.makedirs("./graphs")

    env = suite.load(args.domain_name, args.task_name, {"random": args.seed})

    # Set seeds
    np.random.seed(args.seed)

    temp_timestep = env.reset()
    state_dim = flat_obs(temp_timestep.observation).shape[0]
    action_dim = env.action_spec().shape[0]
    max_action = float(env.action_spec().maximum[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        kwargs["expl_noise"] = args.expl_noise
        kwargs["tau"] = args.tau
        policy = TD3.TD3(**kwargs)
    elif args.policy == "SAC":
        kwargs["policy_freq"] = args.policy_freq
        kwargs["tau"] = args.tau
        policy = SAC.SAC(**kwargs)
    elif args.policy == "MPO":
        policy = MPO.MPO(**kwargs)
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(
        state_dim, action_dim, max_size=int(args.buffer_size)
    )

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.domain_name, args.task_name, args.seed)]

    timestep = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        state = flat_obs(timestep.observation)

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = np.random.uniform(
                env.action_spec().minimum,
                env.action_spec().maximum,
                size=env.action_spec().shape,
            )
        else:
            action = (policy.select_action(state)).clip(-max_action, max_action)

        # Perform action
        timestep = env.step(action)
        done_bool = float(timestep.last())

        # Store data in replay buffer
        replay_buffer.add(
            state, action, flat_obs(timestep.observation), timestep.reward, done_bool
        )

        episode_reward += timestep.reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            for _ in range(args.train_steps):
                if args.policy == "MPO":
                    policy.train(
                        replay_buffer, args.batch_size, args.num_action_samples
                    )
                else:
                    policy.train(replay_buffer, args.batch_size)

        if timestep.last():
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
            )
            # Reset environment
            timestep = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(
                eval_policy(policy, args.domain_name, args.task_name, args.seed)
            )
            np.save(f"./results/{file_name}", evaluations)
        if (t + 1) % args.save_freq == 0:
            if args.save_model:
                policy.save(f"./models/{file_name}_" + str(t + 1))
