"""CPU smoke run of FPO (Algorithm 1) on a toy latent-steering task. NumPy only.

It exercises the same ``fpo_math`` functions as the JAX learner (ratio proxy,
clipped surrogate, Q-ensemble TD targets, GAE, multi-step Euler exploration)
inside the rollout/update loop of Algorithm 1, with tiny linear models in place
of the ResNet actor/critic and a fixed random "decoder" in place of pi0:

* state s in R^2 (planar position), goal: reach ||s|| < 0.2 within 8 steps;
* actor: linear velocity field v_theta(x, tau | s) on a 4-d latent x;
* frozen decoder: a = -0.08 s + 0.25 tanh(W x) (a weak prior plus latent-driven motion);
* reward -1 per step, 0 and terminal on success.

This is a functional check of the algorithm on a toy task (the loop runs, the
invariants below hold, all parameters stay finite). Invariants checked:

* the loss cached at rollout equals the loss re-evaluated under theta_old, so
  Delta = 0 and rho = 1 on the first update after theta_old <- theta;
* all ablation switches (``--no_clip``, ``--euler_steps 1``, ``--num_qs 1``) run.

    python scripts/smoke_cpu.py                 # FPO
    python scripts/smoke_cpu.py --no_clip       # ablations run through the same switches
    python scripts/smoke_cpu.py --euler_steps 1
    python scripts/smoke_cpu.py --num_qs 1
"""
import argparse
import importlib.util
import pathlib
import time

import numpy as np

_path = pathlib.Path(__file__).resolve().parents[1] / 'jaxrl2' / 'agents' / 'pixel_fpo' / 'fpo_math.py'
_spec = importlib.util.spec_from_file_location('fpo_math', _path)
fm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fm)

D, S, T_MAX = 4, 2, 8


class Adam:
    def __init__(self, shape, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.m, self.v, self.t, self.lr, self.b1, self.b2, self.eps = np.zeros(shape), np.zeros(shape), 0, lr, b1, b2, eps

    def step(self, p, g, max_norm=None):
        if max_norm is not None:
            n = np.linalg.norm(g)
            g = g * min(1.0, max_norm / (n + 1e-8))
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh, vh = self.m / (1 - self.b1 ** self.t), self.v / (1 - self.b2 ** self.t)
        return p - self.lr * mh / (np.sqrt(vh) + self.eps)


# ----------------------------------------------------------------------------- actor
def velocity(theta, x, tau, s):
    """v_theta(x, tau | s) = A x + B s + c tau + d, theta = [A | B | c | d] (D x (D + S + 2))."""
    tau = np.broadcast_to(np.asarray(tau, dtype=float).reshape(-1, 1) if np.ndim(tau) else tau, (x.shape[0], 1))
    feats = np.concatenate([x, s, tau, np.ones((x.shape[0], 1))], -1)
    return feats @ theta.T


def sample_latent(theta, s, rng, K, latent_clip=2.0):
    x0 = np.clip(rng.normal(size=(len(s), D)), -3, 3)
    x = fm.euler_integrate(lambda x, tau: velocity(theta, x, tau, s), x0, K)            # Eq. 7
    return np.clip(x, -latent_clip, latent_clip)


def cfm_loss(theta, s, x1, eps, t):
    B, N, _ = eps.shape
    xt = fm.cfm_interpolant(x1, eps, t).reshape(B * N, D)
    v = velocity(theta, xt, t.reshape(B * N), np.repeat(s, N, 0)).reshape(B, N, D)
    return fm.per_sample_cfm_loss(v, x1, eps, 'huber')


# ----------------------------------------------------------------------------- critic ensemble
def features(s, x):
    z = np.concatenate([s, x], -1)
    iu = np.triu_indices(z.shape[1])
    quad = (z[:, :, None] * z[:, None, :])[:, iu[0], iu[1]]
    return np.concatenate([np.ones((len(z), 1)), z, quad], -1)


def q_values(w, s, x):
    return features(s, x) @ w.T            # [B, M] -> transpose for fpo_math ([M, B])


# ----------------------------------------------------------------------------- environment
class Env:
    def __init__(self, rng):
        self.rng = rng
        self.W = np.random.default_rng(123).normal(size=(S, D)) / np.sqrt(D)

    def reset(self):
        ang, r = self.rng.uniform(0, 2 * np.pi), self.rng.uniform(0.8, 1.0)
        self.s, self.t = np.array([r * np.cos(ang), r * np.sin(ang)]), 0
        return self.s.copy()

    def step(self, x):
        a = -0.08 * self.s + 0.25 * np.tanh(self.W @ x)        # frozen "base policy" decoder
        self.s, self.t = self.s + a, self.t + 1
        success = np.linalg.norm(self.s) < 0.2
        return self.s.copy(), success, success or self.t >= T_MAX


def run_episode(env, latent_fn):
    s, obs, lat, done, success = env.reset(), [], [], False, False
    while not done:
        x = latent_fn(s)
        obs.append(s)
        lat.append(x)
        s, success, done = env.step(x)
    n = len(lat)
    rewards = np.r_[-np.ones(n - 1), 0.0] if success else -np.ones(n)
    masks = np.r_[np.ones(n - 1), 0.0] if success else np.ones(n)
    return dict(obs=np.array(obs), next_obs=np.vstack([obs[1:], s[None]]) if n > 1 else s[None], latents=np.array(lat),
                rewards=rewards, masks=masks, success=success)


def evaluate(env, theta, rng, K, episodes):
    return np.mean([run_episode(env, lambda s: sample_latent(theta, s[None], rng, K)[0])['success']
                    for _ in range(episodes)])


# ----------------------------------------------------------------------------- FPO (Algorithm 1)
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iterations', type=int, default=40)
    p.add_argument('--episodes_per_iter', type=int, default=8)
    p.add_argument('--updates_per_iter', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--window', type=int, default=600)
    p.add_argument('--num_fpo_samples', type=int, default=4)
    p.add_argument('--euler_steps', type=int, default=4)
    p.add_argument('--num_qs', type=int, default=5)
    p.add_argument('--clip_epsilon', type=float, default=0.2)
    p.add_argument('--no_clip', action='store_true')
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gae_lambda', type=float, default=0.95)
    p.add_argument('--eval_episodes', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    env = Env(rng)
    theta = np.zeros((D, D + S + 2))                      # v = 0: latents are N(0, I), i.e. the unmodified base
    actor_opt = Adam(theta.shape, 3e-3)
    n_feat = features(np.zeros((1, S)), np.zeros((1, D))).shape[1]
    w = rng.normal(size=(a.num_qs, n_feat)) * 0.1
    w_target, critic_opt = w.copy(), Adam(w.shape, 3e-3)

    buf = dict(obs=[], next_obs=[], latents=[], rewards=[], masks=[], eps=[], t=[], init=[], adv=[])
    bounds = []
    diag = {'clip_frac': [], 'max_abs_delta_after_sync': 0.0}
    t0 = time.time()
    sr0 = evaluate(env, theta, rng, a.euler_steps, a.eval_episodes)
    print(f'iteration 0: success rate {sr0:.3f} (unmodified base, N(0, I) latents)')

    for it in range(1, a.iterations + 1):
        theta_old = theta.copy()                                          # line 3
        for _ in range(a.episodes_per_iter):                              # lines 5-12
            ep = run_episode(env, lambda s: sample_latent(theta_old, s[None], rng, a.euler_steps)[0])
            n = len(ep['latents'])
            eps = np.clip(rng.normal(size=(n, a.num_fpo_samples, D)), -2.5, 2.5)
            t = rng.uniform(0.01, 0.99, size=(n, a.num_fpo_samples, 1))
            init = cfm_loss(theta_old, ep['obs'], ep['latents'], eps, t)  # line 10: cache l_init
            start = len(buf['rewards'])
            for k in ('obs', 'next_obs', 'latents', 'rewards', 'masks'):
                buf[k].extend(list(ep[k]))
            buf['eps'].extend(list(eps)); buf['t'].extend(list(t)); buf['init'].extend(list(init))
            buf['adv'].extend([0.0] * n)
            bounds.append((start, start + n))
        size = len(buf['rewards'])
        lo = max(0, size - a.window)
        arr = {k: np.asarray(v) for k, v in buf.items()}

        # GAE on the window with V(s) = min_i Q_i(s, x_stored); fresh latent for open ends
        for (s0, e0) in bounds:
            if e0 <= lo:
                continue
            V = fm.reduce_ensemble(q_values(w, arr['obs'][s0:e0], arr['latents'][s0:e0]).T, 'min')
            boot = 0.0
            if arr['masks'][e0 - 1] > 0:
                xn = sample_latent(theta, arr['next_obs'][e0 - 1:e0], rng, a.euler_steps)
                boot = float(fm.reduce_ensemble(q_values(w, arr['next_obs'][e0 - 1:e0], xn).T, 'min')[0])
            arr['adv'][s0:e0] = fm.gae_advantages(arr['rewards'][s0:e0], V, boot, arr['masks'][s0:e0], a.gamma,
                                                  a.gae_lambda)

        first_update = it == 1
        for _ in range(a.updates_per_iter):                               # lines 14-28
            idx = rng.integers(lo, size, a.batch_size)
            s_b, x_b, r_b, m_b = arr['obs'][idx], arr['latents'][idx], arr['rewards'][idx], arr['masks'][idx]
            # critic (Eqs. 5-6)
            xn = sample_latent(theta, arr['next_obs'][idx], rng, a.euler_steps)
            y = fm.td_target(r_b, m_b, q_values(w_target, arr['next_obs'][idx], xn).T, a.gamma, 'min')
            phi = features(s_b, x_b)
            err = phi @ w.T - y[:, None]                                  # [B, M]
            w = critic_opt.step(w, 2 * err.T @ phi / a.batch_size)
            w_target = 0.95 * w_target + 0.05 * w                          # Polyak
            # actor (Eqs. 2-4): finite-difference gradient, batch statistics frozen (stop-gradient)
            adv = fm.standardize_advantages(arr['adv'][idx])
            eps_b, t_b, init_b = arr['eps'][idx], arr['t'][idx], arr['init'][idx]

            def delta_of(th):
                return fm.cfm_loss_differential(init_b, cfm_loss(th, s_b, x_b, eps_b, t_b))

            d0 = delta_of(theta)
            stats = (d0.mean(), np.sqrt(d0.var() + 1e-3 ** 2))
            rho0 = np.exp(np.clip(a.beta * (d0 - stats[0]) / stats[1], -5, 5))
            diag['clip_frac'].append(float(fm.clipped_fraction(rho0, a.clip_epsilon)))
            if first_update:      # theta == theta_old and the batch holds only fresh rollouts
                diag['max_abs_delta_after_sync'] = max(diag['max_abs_delta_after_sync'], float(np.abs(d0).max()))
                first_update = False

            def objective(th):
                z = (delta_of(th) - stats[0]) / stats[1]
                rho = np.exp(np.clip(a.beta * z, -5, 5))
                return fm.clipped_surrogate_loss(rho, adv, a.clip_epsilon, not a.no_clip)

            g, h = np.zeros_like(theta), 1e-5
            for ij in np.ndindex(theta.shape):
                e = np.zeros_like(theta)
                e[ij] = h
                g[ij] = (objective(theta + e) - objective(theta - e)) / (2 * h)
            theta = actor_opt.step(theta, g, max_norm=0.5)

        if it % 10 == 0 or it == a.iterations:
            q = q_values(w, s_b, x_b)
            print(f'iteration {it}: buffer {size} (window {size - lo})  critic |Q| {np.abs(q).mean():.2f}  '
                  f'clipped fraction {np.mean(diag["clip_frac"][-a.updates_per_iter:]):.2f}')

    sr1 = evaluate(env, theta, rng, a.euler_steps, a.eval_episodes)
    ok = np.isfinite(theta).all() and np.isfinite(w).all()
    print(f'max |Delta| on the first update after a sync: {diag["max_abs_delta_after_sync"]:.2e} (expected 0)')
    print(f'final: success rate {sr0:.3f} -> {sr1:.3f}  ({time.time() - t0:.1f} s, finite={ok})')
    if not ok or diag['max_abs_delta_after_sync'] > 1e-9:
        raise SystemExit('smoke check failed')
    print('SMOKE OK')


if __name__ == '__main__':
    main()
