"""CPU check of the FPO data path without JAX, gym, Flax, openpi or a simulator.

Runs the real ``FPOReplayBuffer`` (sliding window, cached CFM fields, GAE) and the
real rollout/update loop of ``examples/train_utils_fpo.py`` against small stand-ins:
NumPy versions of the few JAX / Flax / gym functions they call, a fake LIBERO-style
environment, a fake pi0 decoder and a fake agent. It checks shapes, the window,
GAE values, terminal handling and the iteration structure of Algorithm 1
(one theta_old sync per rollout, ``len(episode) * multi_grad_step`` updates).

    python scripts/check_loop_cpu.py

The stand-ins are installed in this process only; run it as a separate command.
"""
import pathlib
import sys
import types

import numpy as np
import PIL.Image  # noqa: F401  (examples/train_utils_sim.py uses PIL.Image via `import PIL`)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def install_standins():
    class Key:
        def __init__(self, s):
            self.s = int(s)

    jax = types.ModuleType('jax')
    rnd = types.ModuleType('jax.random')
    rnd.PRNGKey = Key
    rnd.split = lambda k, n=2: [Key(k.s * 7 + i + 1) for i in range(n)]
    rnd.normal = lambda k, shape: np.random.default_rng(k.s).normal(size=shape)
    jax.random, jax.numpy = rnd, np
    jax.device_put = jax.device_get = lambda x: x
    flax = types.ModuleType('flax')
    core = types.ModuleType('flax.core')
    fd = types.ModuleType('flax.core.frozen_dict')
    fd.freeze, fd.FrozenDict = (lambda d: d), dict
    core.FrozenDict, core.frozen_dict, flax.core = dict, fd, core

    gym = types.ModuleType('gym')
    spaces = types.ModuleType('gym.spaces')
    utils = types.ModuleType('gym.utils')
    seeding = types.ModuleType('gym.utils.seeding')

    class Space:
        pass

    class Box(Space):
        def __init__(self, low, high, shape, dtype):
            self.shape, self.dtype = shape, dtype

    class Dict(Space):
        def __init__(self, spaces_):
            self.spaces = spaces_

    spaces.Box, spaces.Dict, spaces.Space, gym.spaces, gym.Space = Box, Dict, Space, spaces, Space
    seeding.np_random = lambda seed: (np.random.default_rng(seed), seed)
    utils.seeding, gym.utils = seeding, utils

    image_tools = types.ModuleType('openpi_client.image_tools')
    image_tools.resize_with_pad = lambda img, h, w: img
    image_tools.convert_to_uint8 = lambda x: x
    openpi_client = types.ModuleType('openpi_client')
    openpi_client.image_tools = image_tools

    class _Bar:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def update(self, n=1):
            pass

    tqdm = types.ModuleType('tqdm')
    tqdm.tqdm = _Bar
    # jaxrl2/agents/__init__.py imports the SAC learner (JAX); expose the package path only
    agents = types.ModuleType('jaxrl2.agents')
    agents.__path__ = [str(ROOT / 'jaxrl2' / 'agents')]
    sys.modules.update({
        'jax': jax, 'jax.random': rnd, 'jax.numpy': np, 'flax': flax, 'flax.core': core,
        'flax.core.frozen_dict': fd, 'gym': gym, 'gym.spaces': spaces, 'gym.utils': utils,
        'gym.utils.seeding': seeding, 'openpi_client': openpi_client, 'openpi_client.image_tools': image_tools,
        'wandb': types.ModuleType('wandb'), 'tqdm': tqdm, 'jaxrl2.agents': agents})
    return Key, Box, Dict


Key, Box, Dict = install_standins()
from jaxrl2.agents.pixel_fpo.fpo_math import gae_advantages  # noqa: E402
from jaxrl2.data.replay_buffer_fpo import FPOReplayBuffer  # noqa: E402
from jaxrl2.utils.general_utils import AttrDict  # noqa: E402
import examples.train_utils_fpo as loop  # noqa: E402


def check_buffer():
    obs_space = Dict({'pixels': Box(0, 255, (4, 4, 3, 1), np.uint8)})
    buf = FPOReplayBuffer(obs_space, Box(-1, 1, (1, 32), np.float32), 4, num_fpo_samples=2, window=6)
    for ep, T in enumerate([3, 4, 2]):
        for t in range(T):
            last = t == T - 1
            buf.insert({'observations': {'pixels': np.full((4, 4, 3, 1), t, np.uint8)},
                        'next_observations': {'pixels': np.full((4, 4, 3, 1), t + 1, np.uint8)},
                        'actions': np.full((1, 32), 0.1 * t, np.float32),
                        'eps': np.full((2, 32), ep, np.float32), 't': np.full((2, 1), 0.5, np.float32),
                        'initial_cfm_loss': np.full(2, ep + t, np.float32),
                        'rewards': 0.0 if (last and ep == 1) else -1.0,
                        'masks': 0.0 if (last and ep == 1) else 1.0})
        buf.increment_traj_counter()
    assert len(buf) == 9 and buf.capacity == 16 and buf.window_start() == 3
    b = buf.sample(512)
    assert b['eps'].shape == (512, 2, 32) and b['t'].shape == (512, 2, 1)
    assert set(np.unique(b['eps'][:, 0, 0]).tolist()) <= {1.0, 2.0}, 'sampled outside the window'
    sizes = []

    def value_fn(o, a):            # V = -10 * latent[0] = -t
        sizes.append(len(a))
        return -np.asarray(a).reshape(len(a), -1)[:, 0] * 10.0

    buf.compute_gae(value_fn, lambda o: np.full(len(o['pixels']), -5.0), 0.99, 0.95, chunk_size=4)
    assert set(sizes) == {4}, 'value chunks must have a fixed size'
    np.testing.assert_allclose(buf.data['advantages'][3:7],
                               gae_advantages([-1, -1, -1, 0], [0, -1, -2, -3], 0.0, [1, 1, 1, 0], 0.99, 0.95),
                               rtol=1e-5, atol=1e-5)    # terminal episode: no bootstrap
    np.testing.assert_allclose(buf.data['advantages'][7:9],
                               gae_advantages([-1, -1], [0, -1], -5.0, [1, 1], 0.99, 0.95),
                               rtol=1e-5, atol=1e-5)    # time-limit episode: fresh-latent bootstrap
    assert np.all(buf.data['advantages'][:3] == 0), 'trajectory outside the window was touched'
    print('replay window + GAE: OK')


class FakeLiberoEnv:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def _obs(self):
        return {'agentview_image': self.rng.integers(0, 255, (32, 32, 3), dtype=np.uint8),
                'robot0_eye_in_hand_image': np.zeros((32, 32, 3), np.uint8), 'robot0_eef_pos': np.zeros(3),
                'robot0_eef_quat': np.array([0, 0, 0, 1.0]), 'robot0_gripper_qpos': np.zeros(2)}

    def reset(self):
        self.t, self.solves = 0, self.rng.random() < 0.4
        return self._obs()

    def step(self, action):
        assert action.shape == (7,)
        self.t += 1
        done = self.solves and self.t >= 15
        return self._obs(), float(done), done, {}


class FakePi0:
    def infer(self, obs, noise):
        assert noise.shape == (1, 50, 32)
        return {'actions': np.zeros((50, 7))}


class FakeAgent:
    action_chunk_shape, latent_clip, advantage_type, discount, gae_lambda = (1, 32), 2.0, 'gae', 0.99, 0.95

    def __init__(self):
        self._rng, self.syncs, self.updates, self.N = Key(0), 0, 0, 3

    def sync_old_actor(self):
        self.syncs += 1

    def sample_actions_with_info(self, obs, latent=None):
        assert obs['pixels'].shape == (1, 16, 16, 3, 1) and obs['state'].shape == (1, 8, 1)
        latent = np.random.normal(size=(1, 32)).clip(-2, 2) if latent is None else np.reshape(latent, (1, 32))
        return latent, {'eps': np.zeros((self.N, 32)), 't': np.full((self.N, 1), 0.5),
                        'initial_cfm_loss': np.full(self.N, 0.3)}

    def sample_actions(self, obs):
        return np.zeros((1, 32))

    def values(self, obs, latents):
        return np.zeros(len(latents))

    def fresh_values(self, obs):
        return np.zeros(len(obs['pixels']))

    def update(self, batch):
        assert batch['eps'].shape == (8, 3, 32) and batch['initial_cfm_loss'].shape == (8, 3)
        self.updates += 1
        return {'actor_loss': np.float32(0.0)}


class Log:
    def __init__(self):
        self.rows = []

    def log(self, d, step=None):
        self.rows.append((step, d))


def check_loop():
    variant = AttrDict(env='libero', algorithm='pixel_fpo', resize_image=16, add_states=1, query_freq=10,
                       max_timesteps=40, env_max_reward=1, start_online_updates=12, multi_grad_step=2,
                       max_steps=60, log_interval=10, eval_interval=30, eval_episodes=2, checkpoint_interval=-1,
                       save_latents=0, eval_at_start=1, discount=0.99, seed=0, clip_base_actions=1,
                       task_description='task')
    space = Dict({'pixels': Box(0, 255, (16, 16, 3, 1), np.uint8), 'state': Box(-1, 1, (8, 1), np.float32)})
    buf = FPOReplayBuffer(space, Box(-1, 1, (1, 32), np.float32), 8, num_fpo_samples=3, window=20)
    agent, log = FakeAgent(), Log()
    loop.trajwise_alternating_training_loop(variant, agent, FakeLiberoEnv(0), FakeLiberoEnv(1), buf,
                                            buf.get_iterator(8), log, agent_dp=FakePi0())
    assert agent.updates == variant.max_steps
    assert agent.syncs == buf._traj_counter, 'theta_old must be synced once per rollout'
    for s, e in buf.traj_bounds.values():
        succ = buf.data['masks'][e - 1] == 0
        assert buf.data['rewards'][e - 1] == (0.0 if succ else -1.0)
        assert np.all(buf.data['rewards'][s:e - 1] == -1.0) and np.all(buf.data['masks'][s:e - 1] == 1.0)
    evals = [step for step, d in log.rows if 'evaluation/success_rate' in d]
    assert evals == [0, 30, 60], evals
    print(f'rollout/update loop: OK ({buf._traj_counter} episodes, {len(buf)} transitions, '
          f'{agent.updates} updates, evaluations at {evals})')


if __name__ == '__main__':
    check_buffer()
    check_loop()
    print('LOOP CHECK OK')
