#! /usr/bin/env python
"""FPO / latent-SAC training on LIBERO and ALOHA Transfer Cube with a frozen pi0.

Entry point: ``examples/launch_train_fpo.py`` (see ``examples/scripts/``).
"""
import os

# XLA: Triton GEMM gives ~30% more steps/s on some GPUs 
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_triton_gemm_any=True'

import copy
import pathlib
import tempfile
from functools import partial

import gym_aloha  # noqa: F401  (registers gym_aloha)
import gymnasium as gym
import jax
import numpy as np
import tensorflow as tf
from gym.spaces import Box, Dict
from jax.experimental.compilation_cache import compilation_cache
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as openpi_config

from examples.train_utils_fpo import trajwise_alternating_training_loop
from jaxrl2.agents.pixel_fpo.pixel_fpo_learner import PixelFPOLearner
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.data.replay_buffer_fpo import FPOReplayBuffer
from jaxrl2.utils.general_utils import add_batch_dim
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name

compilation_cache.initialize_cache(os.environ.get('JAX_COMPILATION_CACHE_DIR',
                                                  os.path.join(os.path.expanduser('~'), '.cache', 'jax_fpo')))

PI0_LATENT_DIM = 32  # width of pi0's action-noise chunk; one 32-d latent per query step


def _get_libero_env(task, resolution, seed):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)  # the seed also affects object placement
    return env, task.language


def shard_batch(batch, sharding):
    """Shard the leading (batch) dimension across all local devices."""
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))), batch)


class DummyEnv(gym.ObservationWrapper):
    """Observation / latent-action spaces seen by the learner."""

    def __init__(self, variant):
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        spaces = {'pixels': Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)}
        if variant.add_states:
            state_dim = 8 if variant.env == 'libero' else 14
            spaces['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        self.observation_space = Dict(spaces)
        self.action_space = Box(low=-1, high=1, shape=(1, PI0_LATENT_DIM), dtype=np.float32)


def make_envs(variant):
    if variant.env == 'libero':
        task_suite = benchmark.get_benchmark_dict()[variant.libero_suite]()
        task = task_suite.get_task(variant.libero_task_id)
        env, task_description = _get_libero_env(task, 256, variant.seed)
        print(f'LIBERO {variant.libero_suite} task {variant.libero_task_id}: "{task_description}"')
        variant.task_description = task_description
        variant.env_max_reward = 1
        variant.max_timesteps = variant.max_timesteps if variant.max_timesteps > 0 else 400
        return env, env
    if variant.env == 'aloha_cube':
        from gymnasium.envs.registration import register
        register(id="gym_aloha/AlohaTransferCube-v0", entry_point="gym_aloha.env:AlohaEnv", max_episode_steps=400,
                 nondeterministic=True, kwargs={"obs_type": "pixels", "task": "transfer_cube"})
        env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
        variant.env_max_reward = 4
        variant.max_timesteps = variant.max_timesteps if variant.max_timesteps > 0 else 400
        return env, copy.deepcopy(env)
    raise NotImplementedError(variant.env)


def load_pi0(variant):
    name = {'libero': 'pi0_libero', 'aloha_cube': 'pi0_aloha_sim'}[variant.env]
    checkpoint_dir = download.maybe_download(f"s3://openpi-assets/checkpoints/{name}")
    return policy_config.create_trained_policy(openpi_config.get_config(name), checkpoint_dir)


def main(variant):
    devices = jax.local_devices()
    assert variant.batch_size % len(devices) == 0, \
        f"batch_size ({variant.batch_size}) must be divisible by the number of devices ({len(devices)})"
    shard_fn = partial(shard_batch, sharding=jax.sharding.PositionalSharding(devices))
    tf.config.set_visible_devices([], "GPU")  # keep TensorFlow off the GPUs

    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps

    expname = create_exp_name(variant.prefix, seed=variant.seed) + (f"_{variant.suffix}" if variant.suffix else "")
    variant.outputdir = os.path.join(os.environ.get('EXP', './logs'), expname)
    os.makedirs(os.path.join(variant.outputdir, 'latents') if variant.save_latents else variant.outputdir,
                exist_ok=True)
    print('writing to output dir', variant.outputdir)

    env, eval_env = make_envs(variant)
    wandb_logger = WandBLogger(variant.use_wandb and variant.prefix != '', variant, variant.wandb_project,
                               experiment_id=expname, output_dir=tempfile.mkdtemp(),
                               group_name=variant.prefix + '_' + variant.launch_group_id)

    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    agent_dp = load_pi0(variant)

    if variant.algorithm == 'pixel_fpo':
        agent = PixelFPOLearner(variant.seed, sample_obs, sample_action, **kwargs)
        num_fpo_samples = kwargs['num_fpo_samples']
    elif variant.algorithm == 'pixel_sac':
        # Table II "without CFM ratio proxy": SAC-style update of the same latent actor
        agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)
        num_fpo_samples = None
    else:
        raise ValueError(variant.algorithm)

    replay_buffer = FPOReplayBuffer(dummy_env.observation_space, dummy_env.action_space,
                                    int(variant.max_steps // variant.multi_grad_step),
                                    num_fpo_samples=num_fpo_samples, window=variant.buffer_window)
    replay_buffer.seed(variant.seed)
    np.random.seed(variant.seed)
    replay_buffer_iterator = map(shard_fn, replay_buffer.get_iterator(variant.batch_size))

    trajwise_alternating_training_loop(variant, agent, env, eval_env, replay_buffer, replay_buffer_iterator,
                                       wandb_logger, agent_dp=agent_dp)
