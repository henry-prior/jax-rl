import argparse
import os

import numpy as np
from dm_control import suite

from buffers import ReplayBuffer
from MPO import MPO
from SAC import SAC
from TD3 import TD3
from train_loops import base_train_loop
from train_loops import mpo_train_loop
from utils import flat_obs


TRAIN_LOOPS = dict(TD3=base_train_loop, SAC=base_train_loop, MPO=mpo_train_loop)


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
    parser.add_argument("--num_action_samples", default=64, type=int)
    parser.add_argument("--save_freq", default=5e3, type=int)
    parser.add_argument("--episode_length", default=None, type=int)
    args = parser.parse_args()

    args.file_name = f"{args.policy}_{args.domain_name}_{args.seed}"
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
        policy = TD3(**kwargs)
    elif args.policy == "SAC":
        kwargs["policy_freq"] = args.policy_freq
        kwargs["tau"] = args.tau
        policy = SAC(**kwargs)
    elif args.policy == "MPO":
        args.episode_length = 1000
        args.target_freq = 250
        policy = MPO(**kwargs)
    if args.load_model != "":
        policy_file = (
            args.file_name if args.load_model == "default" else args.load_model
        )
        policy.load(f"./models/{policy_file}")

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        max_size=int(args.buffer_size),
        episode_length=int(args.episode_length),
    )

    train_loop = TRAIN_LOOPS[args.policy]

    train_loop(args, policy, replay_buffer, env)
