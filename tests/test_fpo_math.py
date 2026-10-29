"""CPU unit tests for the FPO math (paper Eqs. 2-7). NumPy only.

    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _fpo_math_loader import load_fpo_math  # noqa: E402

fm = load_fpo_math()
RNG = np.random.default_rng(0)


def huber(x):
    a = np.abs(x)
    return np.where(a <= 1.0, 0.5 * x ** 2, a - 0.5)


class TestCFMLoss(unittest.TestCase):

    def test_interpolant_and_target(self):
        x1 = RNG.normal(size=(3, 5))
        eps = RNG.normal(size=(3, 4, 5))
        t = RNG.uniform(size=(3, 4, 1))
        xt = fm.cfm_interpolant(x1, eps, t)
        np.testing.assert_allclose(xt[1, 2], (1 - t[1, 2]) * eps[1, 2] + t[1, 2] * x1[1])
        np.testing.assert_allclose(fm.cfm_target_velocity(x1, eps)[2, 3], x1[2] - eps[2, 3])

    def test_per_sample_loss_shapes_and_values(self):
        B, N, D = 4, 3, 6
        x1, eps = RNG.normal(size=(B, D)), RNG.normal(size=(B, N, D))
        v = RNG.normal(size=(B, N, D)) * 2.0
        target = x1[:, None, :] - eps
        hub = fm.per_sample_cfm_loss(v, x1, eps, 'huber')
        mse = fm.per_sample_cfm_loss(v, x1, eps, 'mse')
        self.assertEqual(hub.shape, (B, N))
        np.testing.assert_allclose(hub, huber(v - target).mean(-1))
        np.testing.assert_allclose(mse, ((v - target) ** 2).mean(-1))
        # exact velocity -> zero loss
        np.testing.assert_allclose(fm.per_sample_cfm_loss(target, x1, eps), 0.0, atol=1e-12)
        with self.assertRaises(ValueError):
            fm.per_sample_cfm_loss(v, x1, eps, 'l1')


class TestRatioProxy(unittest.TestCase):

    def test_loss_differential_sign_and_clip(self):
        init = np.array([[1.0, 2.0, 10.0], [0.5, 0.5, 0.5]])
        curr = np.array([[0.5, 1.0, 0.0], [1.5, 0.5, 0.5]])
        d = fm.cfm_loss_differential(init, curr, cfm_clip=3.0)
        # (0.5 + 1.0 + clip(10, 3)) / 3 and (-1 + 0 + 0) / 3
        np.testing.assert_allclose(d, [(0.5 + 1.0 + 3.0) / 3, -1.0 / 3])

    def test_standardised_ratio(self):
        delta = RNG.normal(size=256) * 3.0 + 1.0
        rho, z = fm.ratio_proxy(delta, beta=1.0)
        self.assertAlmostEqual(z.mean(), 0.0, places=6)
        self.assertAlmostEqual(z.std(), 1.0, places=5)
        np.testing.assert_allclose(rho, np.exp(np.clip(z, -5, 5)))
        # order preserving: larger loss reduction -> larger ratio (Sec. III-B-a)
        order = np.argsort(delta)
        self.assertTrue(np.all(np.diff(rho[order]) >= 0))

    def test_beta_sharpness_and_log_clip(self):
        delta = np.linspace(-1, 1, 101)
        rho1, z = fm.ratio_proxy(delta, beta=1.0)
        rho2, _ = fm.ratio_proxy(delta, beta=2.0)
        np.testing.assert_allclose(np.log(rho2), np.clip(2 * z, -5, 5), atol=1e-12)
        self.assertGreater(np.log(rho2).std(), np.log(rho1).std())
        rho_big, _ = fm.ratio_proxy(delta, beta=100.0, log_ratio_clip=5.0)
        self.assertLessEqual(rho_big.max(), np.exp(5.0) + 1e-9)

    def test_equal_deltas_are_finite(self):
        # first gradient step after theta_old <- theta: every Delta is 0
        rho, z = fm.ratio_proxy(np.zeros(32))
        np.testing.assert_allclose(z, 0.0)
        np.testing.assert_allclose(rho, 1.0)


class TestClippedSurrogate(unittest.TestCase):

    @staticmethod
    def _grad_rho(rho, adv, use_clip=True, h=1e-6):
        f = lambda r: fm.clipped_surrogate_loss(np.array([r]), np.array([adv]), 0.2, use_clip)
        return (f(rho + h) - f(rho - h)) / (2 * h)

    def test_value(self):
        rho = np.array([0.5, 1.0, 1.5, 0.7])
        adv = np.array([1.0, -1.0, 1.0, -2.0])
        expected = -np.mean(np.minimum(rho * adv, np.clip(rho, 0.8, 1.2) * adv))
        self.assertAlmostEqual(fm.clipped_surrogate_loss(rho, adv), expected)
        self.assertAlmostEqual(fm.clipped_surrogate_loss(rho, adv, use_clip=False), -np.mean(rho * adv))

    def test_trust_region(self):
        # positive advantage: no incentive to push rho above 1 + eps
        self.assertEqual(self._grad_rho(1.5, 1.0), 0.0)
        self.assertLess(self._grad_rho(1.1, 1.0), 0.0)
        # negative advantage: no incentive to push rho below 1 - eps
        self.assertEqual(self._grad_rho(0.5, -1.0), 0.0)
        self.assertGreater(self._grad_rho(0.9, -1.0), 0.0)
        # pessimistic bound keeps the unclipped term when it is lower
        self.assertGreater(self._grad_rho(0.5, 1.0), -2.0)
        # without clipping the gradient never vanishes
        self.assertLess(self._grad_rho(1.5, 1.0, use_clip=False), 0.0)

    def test_clipped_fraction(self):
        rho = np.array([0.7, 0.9, 1.0, 1.3])
        self.assertAlmostEqual(float(fm.clipped_fraction(rho, 0.2)), 0.5)

    def test_advantage_standardisation(self):
        adv = RNG.normal(size=1000) * 5 + 3
        a = fm.standardize_advantages(adv)
        self.assertAlmostEqual(a.mean(), 0.0, places=6)
        self.assertAlmostEqual(a.std(), 1.0, places=4)
        self.assertLessEqual(np.abs(fm.standardize_advantages(np.r_[np.zeros(999), 1e6])).max(), 10.0)


class TestActorGradient(unittest.TestCase):
    """End-to-end gradient of Eqs. 2-4 for a linear velocity field
    v_theta(x, t | s) = W [x; s; t] (finite differences, batch statistics frozen as
    with jax.lax.stop_gradient)."""

    B, N, D, S = 16, 4, 3, 2

    def setUp(self):
        rng = np.random.default_rng(1)
        self.s = rng.normal(size=(self.B, self.S))
        self.x1 = rng.normal(size=(self.B, self.D))
        self.eps = rng.normal(size=(self.B, self.N, self.D))
        self.t = rng.uniform(0.01, 0.99, size=(self.B, self.N, 1))
        self.theta_old = rng.normal(size=(self.D, self.D + self.S + 1)) * 0.1
        self.adv = fm.standardize_advantages(rng.normal(size=self.B))
        self.init = self.cfm(self.theta_old)

    def cfm(self, theta):
        xt = fm.cfm_interpolant(self.x1, self.eps, self.t)
        feats = np.concatenate([xt, np.repeat(self.s[:, None], self.N, 1), self.t], -1)
        return fm.per_sample_cfm_loss(feats @ theta.T, self.x1, self.eps)

    def objective(self, theta, frozen_stats):
        delta = fm.cfm_loss_differential(self.init, self.cfm(theta))
        mu, sigma = frozen_stats
        z = (delta - mu) / sigma
        rho = np.exp(np.clip(z, -5, 5))
        return fm.clipped_surrogate_loss(rho, self.adv)

    def grad(self, theta, h=1e-6):
        delta = fm.cfm_loss_differential(self.init, self.cfm(theta))
        stats = (delta.mean(), np.sqrt(delta.var() + 1e-12))
        g = np.zeros_like(theta)
        for idx in np.ndindex(theta.shape):
            e = np.zeros_like(theta)
            e[idx] = h
            g[idx] = (self.objective(theta + e, stats) - self.objective(theta - e, stats)) / (2 * h)
        return g

    def test_first_step_after_sync_has_signal(self):
        """At theta == theta_old every Delta is 0, yet the gradient is finite and
        non-zero: the step lowers the CFM loss of positive-advantage samples."""
        g = self.grad(self.theta_old)
        self.assertTrue(np.all(np.isfinite(g)))
        self.assertGreater(np.abs(g).max(), 0.0)
        step = self.theta_old - 1e-3 * g / np.abs(g).max()
        change = self.cfm(step).mean(-1) - self.init.mean(-1)
        pos, neg = self.adv > 0, self.adv < 0
        self.assertLess(np.sum(change[pos] * self.adv[pos]) + np.sum(change[neg] * self.adv[neg]), 0.0)


class TestCriticAndGAE(unittest.TestCase):

    def test_reduce_and_td_target(self):
        qs = np.array([[1.0, -2.0, 3.0], [0.5, -1.0, 4.0]])
        np.testing.assert_allclose(fm.reduce_ensemble(qs, 'min'), [0.5, -2.0, 3.0])
        np.testing.assert_allclose(fm.reduce_ensemble(qs, 'mean'), [0.75, -1.5, 3.5])
        y = fm.td_target(np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, 0.0]), qs, 0.99)
        np.testing.assert_allclose(y, [-1 + 0.99 * 0.5, -1 + 0.99 * -2.0, 0.0])  # terminal: no bootstrap
        self.assertAlmostEqual(float(fm.critic_loss(qs, np.zeros(3))), float(np.mean(qs ** 2)))

    def test_gae_matches_reference(self):
        r = np.array([-1.0, -1.0, -1.0, 0.0])
        v = np.array([-2.0, -1.5, -1.0, -0.5])
        m = np.array([1.0, 1.0, 1.0, 0.0])
        g, lam = 0.99, 0.95
        adv = fm.gae_advantages(r, v, 123.0, m, g, lam)  # bootstrap ignored: last step is terminal
        ref = np.zeros(4)
        nxt = np.append(v[1:], 0.0)
        run = 0.0
        for i in reversed(range(4)):
            delta = r[i] + g * m[i] * nxt[i] - v[i]
            run = delta + g * lam * m[i] * run
            ref[i] = run
        np.testing.assert_allclose(adv, ref, rtol=1e-6)

    def test_gae_limits(self):
        r = -np.ones(5)
        v = RNG.normal(size=5)
        m = np.ones(5)
        boot = 0.3
        nxt = np.append(v[1:], boot)
        np.testing.assert_allclose(fm.gae_advantages(r, v, boot, m, 0.9, 0.0), r + 0.9 * nxt - v, rtol=1e-6)
        mc = np.array([sum(0.9 ** k * r[i + k] for k in range(5 - i)) + 0.9 ** (5 - i) * boot for i in range(5)])
        np.testing.assert_allclose(fm.gae_advantages(r, v, boot, m, 0.9, 1.0), mc - v, rtol=1e-5)


class TestEuler(unittest.TestCase):

    def test_time_grid(self):
        self.assertEqual(fm.euler_time_grid(1), [1.0])
        np.testing.assert_allclose(fm.euler_time_grid(4), [0.25, 0.5, 0.75, 1.0])

    def test_constant_field(self):
        x0 = RNG.normal(size=(5, 3))
        v = np.array([0.3, -0.2, 0.1])
        for K in (1, 3, 8):
            np.testing.assert_allclose(fm.euler_integrate(lambda x, tau: v + 0 * x, x0, K), x0 + v, atol=1e-12)

    def test_time_dependent_field_and_clips(self):
        taus = []
        fm.euler_integrate(lambda x, tau: taus.append(tau) or 0 * x, np.zeros((1, 2)), 3)
        np.testing.assert_allclose(taus, [1 / 3, 2 / 3, 1.0])
        out = fm.euler_integrate(lambda x, tau: 100 + 0 * x, np.zeros((1, 2)), 1)
        np.testing.assert_allclose(out, 5.0)  # velocity clipped to 10, state to 5


if __name__ == '__main__':
    unittest.main()
