"""JAX-side checks: the JAX implementations agree with the NumPy reference in
``fpo_math`` and one learner update step runs on CPU. Skipped when JAX / Flax /
gym are not installed (e.g. in a NumPy-only environment).

    JAX_PLATFORMS=cpu python -m unittest tests.test_fpo_jax -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import jax
    import jax.numpy as jnp
    import flax  # noqa: F401
    import gym  # noqa: F401
    HAVE_JAX = True
except Exception:  # pragma: no cover
    HAVE_JAX = False


@unittest.skipUnless(HAVE_JAX, "requires jax, flax and gym")
class TestJaxAgreesWithNumpy(unittest.TestCase):

    def setUp(self):
        from jaxrl2.agents.pixel_fpo import fpo_math, noise_flow_policy
        self.fm, self.nfp = fpo_math, noise_flow_policy
        self.obs = {'pixels': np.zeros((5, 8, 8, 3, 1), np.uint8)}
        self.W = np.random.default_rng(0).normal(size=(4, 5)) * 0.3

    def apply_fn(self, variables, observations, x, t, training=False):
        return jnp.concatenate([x, t], -1) @ jnp.asarray(self.W).T

    def test_euler_integration(self):
        key = jax.random.PRNGKey(0)
        x_jax = self.nfp.euler_integration(self.apply_fn, None, self.obs, key, 4, 6)
        x0 = np.clip(np.asarray(jax.random.normal(key, (5, 4), dtype=jnp.float32)), -3, 3)
        ref = self.fm.euler_integrate(lambda x, tau: np.concatenate([x, np.full((5, 1), tau)], -1) @ self.W.T, x0, 6)
        np.testing.assert_allclose(np.asarray(x_jax), ref, rtol=1e-5, atol=1e-5)

    def test_cfm_loss(self):
        rng = np.random.default_rng(1)
        x1 = rng.normal(size=(5, 4)).astype(np.float32)
        eps = rng.normal(size=(5, 3, 4)).astype(np.float32)
        t = rng.uniform(0.01, 0.99, size=(5, 3, 1)).astype(np.float32)
        got = self.nfp.compute_cfm_loss_per_sample(self.apply_fn, None, self.obs, x1, eps, t)
        xt = self.fm.cfm_interpolant(x1, eps, t)
        v = np.concatenate([xt, t], -1) @ self.W.T
        np.testing.assert_allclose(np.asarray(got), self.fm.per_sample_cfm_loss(v, x1, eps), rtol=1e-5, atol=1e-6)


@unittest.skipUnless(HAVE_JAX, "requires jax, flax and gym")
class TestLearnerStep(unittest.TestCase):

    def test_one_iteration(self):
        from flax.core import frozen_dict
        from jaxrl2.agents.pixel_fpo.pixel_fpo_learner import PixelFPOLearner

        B, N = 8, 3
        obs = {'pixels': np.random.randint(0, 255, (1, 16, 16, 3, 1), np.uint8),
               'state': np.zeros((1, 8, 1), np.float32)}
        act = np.zeros((1, 1, 32), np.float32)
        agent = PixelFPOLearner(0, obs, act, encoder_type='small', latent_dim=16, hidden_dims=(32,),
                                flow_hidden_dims=(32, 32), num_qs=2, num_fpo_samples=N, num_euler_steps=2,
                                advantage_type='q', color_jitter=False, actor_lr=1e-3)
        agent.sync_old_actor()
        rows = []
        for _ in range(B):
            o = {'pixels': np.random.randint(0, 255, (1, 16, 16, 3, 1), np.uint8),
                 'state': np.random.normal(size=(1, 8, 1)).astype(np.float32)}
            latent, info = agent.sample_actions_with_info(o)
            self.assertEqual(info['eps'].shape, (N, 32))
            self.assertTrue(np.all(np.abs(latent) <= 2.0))
            rows.append((o, latent, info))
        batch = frozen_dict.freeze({
            'observations': {k: np.concatenate([r[0][k] for r in rows]) for k in ('pixels', 'state')},
            'next_observations': {k: np.concatenate([r[0][k] for r in rows]) for k in ('pixels', 'state')},
            'actions': np.stack([r[1].reshape(1, 32) for r in rows]),
            'rewards': -np.ones(B, np.float32), 'masks': np.ones(B, np.float32),
            'eps': np.stack([r[2]['eps'] for r in rows]), 't': np.stack([r[2]['t'] for r in rows]),
            'initial_cfm_loss': np.stack([r[2]['initial_cfm_loss'] for r in rows]),
            'advantages': np.zeros(B, np.float32)})
        before = jax.tree_util.tree_leaves(agent._actor.params)
        info = agent.update(batch)
        # first step after the sync: Delta = 0, rho = 1, update still moves the actor
        self.assertAlmostEqual(float(info['rho_mean']), 1.0, places=3)
        self.assertEqual(float(info['skipped_nan']), 0.0)
        self.assertTrue(np.isfinite(float(info['critic_loss'])))
        after = jax.tree_util.tree_leaves(agent._actor.params)
        self.assertTrue(any(not np.allclose(a, b) for a, b in zip(before, after)))
        info = agent.update(batch)
        self.assertTrue(np.isfinite(float(info['actor_loss'])))


@unittest.skipUnless(HAVE_JAX, "requires jax, flax and gym")
class TestReplayWindowAndGAE(unittest.TestCase):

    def test_window_and_gae(self):
        from gym.spaces import Box, Dict
        from jaxrl2.agents.pixel_fpo.fpo_math import gae_advantages
        from jaxrl2.data.replay_buffer_fpo import FPOReplayBuffer

        obs_space = Dict({'pixels': Box(0, 255, (4, 4, 3, 1), np.uint8)})
        buf = FPOReplayBuffer(obs_space, Box(-1, 1, (1, 32), np.float32), 4, num_fpo_samples=2, window=6)
        for ep, T in enumerate([3, 4, 2]):
            for t in range(T):
                last = t == T - 1
                buf.insert({'observations': {'pixels': np.full((4, 4, 3, 1), t, np.uint8)},
                            'next_observations': {'pixels': np.full((4, 4, 3, 1), t + 1, np.uint8)},
                            'actions': np.full((1, 32), 0.1 * t, np.float32),
                            'rewards': 0.0 if (last and ep == 1) else -1.0,
                            'masks': 0.0 if (last and ep == 1) else 1.0})
            buf.increment_traj_counter()
        self.assertEqual(len(buf), 9)
        self.assertEqual(buf.window_start(), 3)
        idx = np.asarray(buf.sample(256)['rewards']).shape[0]
        self.assertEqual(idx, 256)

        value_fn = lambda o, a: -np.asarray(a).reshape(len(a), -1)[:, 0] * 10.0   # V = -t
        fresh_fn = lambda o: np.full(len(o['pixels']), -5.0)
        buf.compute_gae(value_fn, fresh_fn, 0.99, 0.95)
        np.testing.assert_allclose(buf.data['advantages'][3:7],
                                   gae_advantages([-1, -1, -1, 0], [0, -1, -2, -3], 0.0, [1, 1, 1, 0], 0.99, 0.95),
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(buf.data['advantages'][7:9],
                                   gae_advantages([-1, -1], [0, -1], -5.0, [1, 1], 0.99, 0.95), rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
