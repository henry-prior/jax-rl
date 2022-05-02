# jax-rl

Core Deep Reinforcement Learning algorithms using JAX for improved performance relative to PyTorch and TensorFlow. Control tasks rely on the [DeepMind Control Suite](https://github.com/deepmind/dm_control) or [OpenAI Gym](https://github.com/openai/gym). DeepMind has recently open-sourced the MuJoCo physics engine, which is a dependency of this repo. If you haven't already set up MuJoCo, see the [download site](https://mujoco.org/download) and copy the unzipped folder to a `.mujoco` folder in your base directory.

## Current implementations

- [x] TD3
- [x] SAC
- [x] MPO

## Environment and Testing

This repo makes use of the `poetry` package and dependency management tool. To build a local environment with all necessary packages run:

```bash
make init
```

To test local changes run:

```bash
make test
```

### Run

To run each algorithm with DeepMind Control Suite as the environment backend on cartpole swingup from the base directory:

```bash
python jax_rl/main_dm_control.py --policy TD3 --max_timestep 100000
python jax_rl/main_dm_control.py --policy SAC --max_timesteps 100000
python jax_rl/main_dm_control.py --policy MPO --max_timesteps 100000
```

To use the OpenAI Gym environment backend use the `jax_rl/main_gym.py` file instead.


## Results

### Speed

As one would hope, the time per training step is significantly faster between JAX and other leading deep learning frameworks. The following comparison is the time in seconds per 1000 training steps with the same hyperparameters. For those interested in hardware, these were all run on the same machine, mid-2019 MacBook Pro 15-inch - 2.3 GHz Intel Core i9 - 16GB RAM.

|     |      JAX     |    PyTorch    |
|-----|:------------:|:-------------:|
| TD3 | 2.35 ± 0.17  | 6.93 ± 0.16   |
| SAC | 5.57 ± 0.07  | 32.37 ± 1.32  |
| MPO | 39.19 ± 1.09 | 107.51 ± 3.56 |

### Performance

![](docs/_static/cartpole_graph.png?raw=true)
Evaluation of deterministic policy (acting according to the mean of the policy distributions for SAC and MPO) every 5000 training steps for each algorithm. Important parameters are constant for all, including batch size of 256 per training step, 10000 samples to the replay buffer with uniform random sampling before training, and 250000 total steps in the environment.

## Notes on MPO Implementation

Because we have direct access to the jacobian function with JAX, I've opted to use `scipy.optimize.minimize` instead of taking a single gradient step on the temperature parameter per iteration. In my testing this gives much greater stability with only a marginal increase in time per iteration.

One important aspect to note if you are benchmarking these two approaches is that a standard profiler will be misleading. Most of the time will show up in the call to `scipy.optimize.minimize`, but this is due to how JAX calls work internally. JAX does not wait for an operation to complete when an operation is called, but rather returns a pointer to a `DeviceArray` whose value will be updated when the dispatched call is complete. If this object is passed into another JAX method, the same process will be repeated and control will be returned to Python. Any time Python attempts to access the value of a `DeviceArray` it will need to wait for the computation to complete. Because `scipy.optimize.minimize` passed the values of the parameter and the gradient to FORTRAN, this step will require the whole program to wait for all previous JAX calls to complete. To get a more accurate comparison, compare the total time per training step. To read more about how asynchronous dispatch works in JAX, see [this reference](https://jax.readthedocs.io/en/latest/async_dispatch.html).

I've run a quick comparison of the two following the same procedure as the `Speed` section above.

| Sequential Least Squares | Gradient Descent |
|:------------------------:|:----------------:|
|       39.19 ± 1.09       |   38.26 ± 2.74   |
