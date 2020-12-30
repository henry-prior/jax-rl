from utils import eval_policy
from utils import flat_obs
from buffers import ReplayBuffer

import numpy as np


def base_train_loop(args: dict, policy, replay_buffer: ReplayBuffer, env):
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
            action = policy.select_action(state).clip(-max_action, max_action)

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
                policy.train(replay_buffer, args.batch_size)

        if timestep.last():
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} "
                f"Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
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
            np.save(f"./results/{args.file_name}", evaluations)
        if (t + 1) % args.save_freq == 0:
            if args.save_model:
                policy.save(f"./models/{args.file_name}_{str(t+1)}") #+ str(t + 1))

def mpo_train_loop(args, policy, replay_buffer, env):
    evaluations = [eval_policy(policy, args.domain_name, args.task_name, args.seed)]

    timestep = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1
        replay_buffer, mean_reward = policy._sample_trajectory(replay_buffer, env, 1000, 1000)

        for _ in range(args.target_freq):
            policy.train(replay_buffer, args.batch_size)

        print(f"Episode = {t+1} ; "
            f"Mean reward = {np.mean(mean_reward)} ; "
            #f"Mean Q loss = {mean_q_loss / self.runs} ; "
            #f"Policy loss = {mean_policy / self.runs} ; "
            #f"η = {self.η} ; η_kl = {self.η_kl} ; "
            #f"time = {(time.time() - start_time):.2f}"
            )
