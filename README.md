# jax-rl

Core Deep Reinforcement Learning algorithms using JAX for improved performance relative to PyTorch and TensorFlow. Control tasks rely on the [DeepMind Control Suite](https://github.com/deepmind/dm_control); see repo for further setup if you don't have MuJoCo configured.

#### Current implementations

- [x] TD3
- [x] SAC
- [x] MPO
- [ ] PPO
- [ ] A2C/A3C
- [ ] ACKTR

# Run

To test each algorithm on cartpole swingup:

```bash
python main_dm_control.py --max_timestep 100000
```

```bash
python main_dm_control.py --policy SAC --max_timesteps 100000
```

