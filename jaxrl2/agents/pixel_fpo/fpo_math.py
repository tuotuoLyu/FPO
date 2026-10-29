"""Framework-agnostic FPO math (paper Sec. III, Eqs. 2-7 and Algorithm 1).

Every function takes an array namespace ``xp``: ``jax.numpy`` inside the JAX
learner and ``numpy`` in the CPU unit tests and the CPU smoke run, so the same
code is exercised in both places. Functions that sit on the actor's gradient
path also take a ``stop_gradient`` callable (``jax.lax.stop_gradient`` during
training, the identity in NumPy).

Shapes (B = batch, N = Monte-Carlo (eps, tau) draws per sample, D = latent dim,
M = critic-ensemble size):

* latent ``x1``: [B, D]          * noise ``eps``: [B, N, D]
* flow time ``t``: [B, N, 1]     * per-sample CFM loss: [B, N]
* ensemble Q values: [M, B]
"""

from typing import Callable, Sequence

import numpy as np


def _identity(x):
    return x


# ---------------------------------------------------------------------------
# Conditional flow matching (per-sample loss used by the ratio proxy)
# ---------------------------------------------------------------------------

def cfm_interpolant(x1, eps, t):
    """Linear CFM path x_t = (1 - t) * eps + t * x1 (Lipman et al.)."""
    return (1.0 - t) * eps + t * x1[:, None, :]


def cfm_target_velocity(x1, eps):
    """Target velocity of the linear path: x1 - eps."""
    return x1[:, None, :] - eps


def per_sample_cfm_loss(v_pred, x1, eps, loss_type: str = "huber", xp=np):
    """Per-sample CFM loss l_cfm(x | s; theta) for each (eps, t) draw -> [B, N].

    ``v_pred`` is v_theta(x_t, t | s) evaluated on ``cfm_interpolant(x1, eps, t)``.
    Default: elementwise Huber penalty (delta = 1, i.e. 0.5 * squared error for
    |error| <= 1) averaged over latent dimensions; ``loss_type="mse"`` gives the
    plain squared error.
    """
    diff = v_pred - cfm_target_velocity(x1, eps)
    if loss_type == "huber":
        abs_diff = xp.abs(diff)
        err = xp.where(abs_diff <= 1.0, 0.5 * diff ** 2, abs_diff - 0.5)
    elif loss_type == "mse":
        err = diff ** 2
    else:
        raise ValueError(f"Unknown cfm loss type: {loss_type}")
    return xp.mean(err, axis=-1)


# ---------------------------------------------------------------------------
# Likelihood-free ratio proxy (Eqs. 2-3) and clipped surrogate (Eq. 4)
# ---------------------------------------------------------------------------

def cfm_loss_differential(init_loss, curr_loss, cfm_clip: float = 3.0, xp=np):
    """Eq. 2: Delta l_t = l(x_t | s_t; theta_old) - l(x_t | s_t; theta).

    ``init_loss`` is the loss cached at rollout time under theta_old and
    ``curr_loss`` the loss re-evaluated under the current actor on the same
    (s_t, x_t, eps, t). Each Monte-Carlo draw is clipped to [-cfm_clip, cfm_clip]
    (numerical guard) and the draws are averaged -> [B].
    """
    diff = xp.clip(init_loss - curr_loss, -cfm_clip, cfm_clip)
    return xp.mean(diff, axis=-1)


def batch_standardize(x, eps: float = 1e-6, xp=np, stop_gradient: Callable = _identity):
    """(x - mu) / sigma with batch statistics treated as constants.

    sigma = sqrt(var + eps^2) stays finite when all entries are equal, which is
    the case on the first gradient step after theta_old <- theta.
    """
    mu = stop_gradient(xp.mean(x))
    sigma = stop_gradient(xp.sqrt(xp.mean((x - mu) ** 2) + eps ** 2))
    return (x - mu) / sigma


def ratio_proxy(delta, beta: float = 1.0, log_ratio_clip: float = 5.0, std_eps: float = 1e-6,
                xp=np, stop_gradient: Callable = _identity):
    """Eq. 3: z_t = (Delta_t - mu_Delta) / sigma_Delta,  rho_t = exp(beta * z_t).

    Returns ``(rho, z)``. ``beta`` is the sharpness (``cfm_diff_temperature`` in
    the configs); beta * z is clipped to +-log_ratio_clip before exp (numerical
    guard; inactive for |z| < 5 / beta).
    """
    z = batch_standardize(delta, std_eps, xp=xp, stop_gradient=stop_gradient)
    log_rho = xp.clip(beta * z, -log_ratio_clip, log_ratio_clip)
    return xp.exp(log_rho), z


def standardize_advantages(adv, eps: float = 1e-8, clip: float = 10.0, xp=np):
    """Minibatch advantage standardisation (Sec. III-B-b), then clip to +-10."""
    adv = (adv - xp.mean(adv)) / (xp.std(adv) + eps)
    return xp.clip(adv, -clip, clip)


def clipped_surrogate_loss(rho, adv, clip_epsilon: float = 0.2, use_clip: bool = True, xp=np):
    """Eq. 4: L_actor = -E[min(rho * A, clip(rho, 1 - eps, 1 + eps) * A)].

    ``use_clip=False`` is the "without PPO clipping" ablation (Table II):
    L = -E[rho * A].
    """
    surr1 = rho * adv
    if not use_clip:
        return -xp.mean(surr1)
    surr2 = xp.clip(rho, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
    return -xp.mean(xp.minimum(surr1, surr2))


def clipped_fraction(rho, clip_epsilon: float = 0.2, xp=np):
    return xp.mean((xp.abs(rho - 1.0) > clip_epsilon).astype(xp.float32))


# ---------------------------------------------------------------------------
# Critic ensemble (Eqs. 5-6) and advantages (GAE)
# ---------------------------------------------------------------------------

def reduce_ensemble(qs, reduction: str = "min", xp=np):
    """Conservative ensemble reduction over the member axis: [M, B] -> [B]."""
    if reduction == "min":
        return xp.min(qs, axis=0)
    if reduction == "mean":
        return xp.mean(qs, axis=0)
    raise ValueError(f"Unknown reduction: {reduction}")


def td_target(rewards, masks, next_qs, discount: float, reduction: str = "min", xp=np):
    """Eq. 5: y_t = r_t + gamma * m_t * min_i Qbar_i(s_{t+1}, x'_{t+1}); m_t = 0 at terminals."""
    return rewards + discount * masks * reduce_ensemble(next_qs, reduction, xp)


def critic_loss(qs, target, xp=np):
    """Eq. 6: squared TD error averaged over ensemble members and batch."""
    return xp.mean((qs - target[None, :]) ** 2)


def gae_advantages(rewards, values, bootstrap_value: float, masks, discount: float, lam: float):
    """Generalised advantage estimation (Schulman et al.) along one trajectory.

    ``values[t] = V(s_t)`` (conservative ensemble value of the stored latent),
    ``bootstrap_value = V(s_T)`` for the observation after the last transition,
    ``masks[t] = 0`` if transition t ends in a terminal state (bootstrap masked).
    NumPy only (computed once per iteration on the replay window).
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    masks = np.asarray(masks, dtype=np.float64)
    T = len(rewards)
    next_values = np.append(values[1:], bootstrap_value)
    deltas = rewards + discount * masks * next_values - values
    adv = np.zeros(T, dtype=np.float64)
    running = 0.0
    for i in reversed(range(T)):
        running = deltas[i] + discount * lam * masks[i] * running
        adv[i] = running
    return adv.astype(np.float32)


# ---------------------------------------------------------------------------
# Multi-step latent exploration (Eq. 7)
# ---------------------------------------------------------------------------

def euler_time_grid(num_steps: int) -> Sequence[float]:
    """Flow-time discretisation tau^(k) = (k + 1) / K, k = 0..K-1."""
    return [(k + 1) / float(num_steps) for k in range(num_steps)]


def euler_integrate(velocity_fn: Callable, x0, num_steps: int, velocity_clip: float = 10.0,
                    state_clip: float = 5.0, xp=np):
    """Eq. 7: x^(k+1) = x^(k) + eta * v_theta(x^(k), tau^(k) | s) with eta = 1 / K.

    ``velocity_fn(x, tau)`` returns the actor velocity; velocities are clipped
    to +-velocity_clip and states to +-state_clip (numerical guards).
    K = 1 is the "single-step integration" ablation.
    """
    x = x0
    eta = 1.0 / float(num_steps)
    for tau in euler_time_grid(num_steps):
        v = xp.clip(velocity_fn(x, tau), -velocity_clip, velocity_clip)
        x = xp.clip(x + eta * v, -state_clip, state_clip)
    return x
