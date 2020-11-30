# jax-rl

Core Deep Reinforcement Learning algorithms using JAX for improved performance relative to PyTorch and TensorFlow. Control tasks rely on the [DeepMind Control Suite](https://github.com/deepmind/dm_control); see repo for further setup if you don't have MuJoCo configured.

#### Current implementations

- [x] TD3
- [x] SAC
- [x] MPO
- [ ] A2C/A3C

# Migrating to Linen

`flax` has a new interface for models which requires a decent amount of changes. The folowing have been updated:
 - [x] TD3
 - [x] SAC
 - [x] MPO


## Environment and Testing

This repo makes use of the `poetry` package and dependency management tool. To build a local environment with all necessary packages run:

```bash
make install
```

To test local changes run:

```bash
make test
```

# Run

To run each algorithm on cartpole swingup from the base directory:

```bash
python jax_rl/main_dm_control.py --max_timestep 100000
```

```bash
python jax_rl/main_dm_control.py --policy SAC --max_timesteps 100000
```
