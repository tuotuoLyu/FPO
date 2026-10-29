"""Flow-matching actor in the latent (noise) space of a frozen pi0.

The actor pi_theta(x | s) is a conditional flow v_theta(x_t, t | s) whose
samples x (a 32-d latent per query step) are fed to pi0 as the initial noise of
its own flow; pi0 itself stays frozen (paper Sec. III-A, Fig. 1b).
"""

import functools
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.agents.pixel_fpo import fpo_math
from jaxrl2.networks.encoders.networks import PixelMultiplexer


class IdentityNetwork(nn.Module):
    """PixelMultiplexer expects a head network; the actor only needs the encoding."""

    @nn.compact
    def __call__(self, x, training: bool = False):
        return x


class VelocityNetwork(nn.Module):
    """CFM velocity head v_theta(x_t, t | s) -> R^D."""
    action_dim: int
    hidden_dims: Tuple[int, ...] = (512, 512, 512)

    @nn.compact
    def __call__(self, obs_encoding: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        h = jnp.concatenate([obs_encoding, x_t, t], axis=-1)
        for i, dim in enumerate(self.hidden_dims):
            h = nn.Dense(dim, kernel_init=nn.initializers.variance_scaling(2.0, "fan_in", "truncated_normal"))(h)
            if i < len(self.hidden_dims) - 1:
                h = nn.LayerNorm()(h)
            h = nn.relu(h)
        # small output init: the untrained actor starts close to the identity map eps -> x
        return nn.Dense(self.action_dim, kernel_init=nn.initializers.normal(stddev=0.01),
                        bias_init=nn.initializers.zeros)(h)


class NoiseFlowPolicy(nn.Module):
    """Pixel encoder (PixelMultiplexer) + velocity head."""
    encoder_def: nn.Module
    action_dim: int
    latent_dim: int = 256
    hidden_dims: Tuple[int, ...] = (512, 512, 512)
    use_bottleneck: bool = True

    def setup(self):
        self.backbone = PixelMultiplexer(encoder=self.encoder_def, network=IdentityNetwork(),
                                         latent_dim=self.latent_dim, use_bottleneck=self.use_bottleneck)
        self.velocity_head = VelocityNetwork(self.action_dim, self.hidden_dims)

    def __call__(self, observations: Dict[str, Any], x_t: jnp.ndarray, t: jnp.ndarray,
                 training: bool = False) -> jnp.ndarray:
        enc = self.backbone(observations, training=training)
        if isinstance(enc, (dict, FrozenDict)):
            enc = enc['pixels']  # PixelMultiplexer writes the image encoding back under 'pixels'
        return self.velocity_head(enc, x_t, t)


class FlowPolicyState(TrainState):
    batch_stats: Optional[Any] = None
    action_dim: int = 0


def _variables(params, batch_stats):
    return {"params": params} if batch_stats is None else {"params": params, "batch_stats": batch_stats}


# ---------------------------------------------------------------------------
# Multi-step latent exploration (Eq. 7)
# ---------------------------------------------------------------------------

def euler_integration(apply_fn: Callable, params, observations: Dict[str, Any], key: jax.random.PRNGKey,
                      action_dim: int, num_steps: int, batch_stats: Optional[Any] = None,
                      init_clip: float = 3.0, velocity_clip: float = 10.0, state_clip: float = 5.0) -> jnp.ndarray:
    """x^(0) ~ N(0, I) (clipped to +-init_clip), then K Euler steps of size 1/K at
    tau^(k) = (k + 1) / K; JAX twin of ``fpo_math.euler_integrate``."""
    B = observations["pixels"].shape[0]
    x0 = jnp.clip(jax.random.normal(key, (B, action_dim), dtype=jnp.float32), -init_clip, init_clip)
    variables = _variables(params, batch_stats)
    eta = 1.0 / float(num_steps)

    def one_step(x, k):
        tau = ((k + 1) / float(num_steps)) * jnp.ones((B, 1), dtype=jnp.float32)
        v = jnp.clip(apply_fn(variables, observations, x, tau, training=False), -velocity_clip, velocity_clip)
        return jnp.clip(x + eta * v, -state_clip, state_clip), None

    x, _ = jax.lax.scan(one_step, x0, jnp.arange(num_steps, dtype=jnp.int32))
    return x


def sample_latent(apply_fn: Callable, params, observations: Dict[str, Any], key: jax.random.PRNGKey,
                  action_dim: int, num_steps: int, batch_stats: Optional[Any] = None,
                  latent_clip: float = 2.0) -> jnp.ndarray:
    """Latent x_t ~ pi_theta(. | s_t) exactly as executed by pi0 (clipped to +-latent_clip).

    Used for rollouts, evaluation, critic targets x'_{t+1} (Eq. 5) and GAE bootstraps,
    so every consumer sees the same latent distribution."""
    x = euler_integration(apply_fn, params, observations, key, action_dim, num_steps, batch_stats)
    return jnp.clip(x, -latent_clip, latent_clip)


# ---------------------------------------------------------------------------
# Per-sample CFM loss (Eq. 2)
# ---------------------------------------------------------------------------

def sample_cfm_noise(key: jax.random.PRNGKey, batch_size: int, num_samples: int, action_dim: int,
                     eps_clip: float = 2.5, t_min: float = 0.01, t_max: float = 0.99):
    """N Monte-Carlo draws (eps, t) per sample; drawn once at rollout and stored with the transition."""
    key_eps, key_t = jax.random.split(key)
    eps = jnp.clip(jax.random.normal(key_eps, (batch_size, num_samples, action_dim), dtype=jnp.float32),
                   -eps_clip, eps_clip)
    t = jax.random.uniform(key_t, (batch_size, num_samples, 1), minval=t_min, maxval=t_max, dtype=jnp.float32)
    return eps, t


def compute_cfm_loss_per_sample(apply_fn: Callable, params, observations: Dict[str, Any], x1: jnp.ndarray,
                                eps: jnp.ndarray, t: jnp.ndarray, batch_stats: Optional[Any] = None,
                                loss_type: str = "huber") -> jnp.ndarray:
    """l_cfm(x1 | s; theta) for each of the N (eps, t) draws -> [B, N]."""
    B, N, A = eps.shape
    x_t = fpo_math.cfm_interpolant(x1, eps, t)                                   # [B, N, A]
    obs_tiled = jax.tree_util.tree_map(lambda v: jnp.repeat(v, N, axis=0), observations)
    v_pred = apply_fn(_variables(params, batch_stats), obs_tiled, x_t.reshape(B * N, A),
                      t.reshape(B * N, 1), training=False).reshape(B, N, A)
    return fpo_math.per_sample_cfm_loss(v_pred, x1, eps, loss_type=loss_type, xp=jnp)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def create_flow_policy(sample_observations: Dict[str, Any], action_chunk_shape: Tuple[int, ...],
                       encoder_def: nn.Module, seed: int = 42, actor_lr=5e-5, latent_dim: int = 256,
                       hidden_dims: Tuple[int, ...] = (512, 512, 512), use_bottleneck: bool = True,
                       max_grad_norm: float = 1.0) -> FlowPolicyState:
    key = jax.random.PRNGKey(seed)
    action_dim = 1
    for d in action_chunk_shape:
        action_dim *= int(d)
    policy_def = NoiseFlowPolicy(encoder_def=encoder_def, action_dim=action_dim, latent_dim=latent_dim,
                                 hidden_dims=tuple(hidden_dims), use_bottleneck=use_bottleneck)
    dummy_obs = jax.tree_util.tree_map(lambda x: x[:1], sample_observations)
    variables = policy_def.init(key, dummy_obs, jnp.zeros((1, action_dim)), jnp.zeros((1, 1)), training=False)
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(actor_lr))
    state = FlowPolicyState.create(apply_fn=policy_def.apply, params=variables['params'], tx=tx,
                                   batch_stats=variables.get('batch_stats', None))
    return state.replace(action_dim=action_dim)


# ---------------------------------------------------------------------------
# Jitted entry points used by the learner
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("action_dim", "num_steps", "latent_clip"))
def jit_sample_latent(state: FlowPolicyState, params, observations, key, action_dim: int, num_steps: int,
                      latent_clip: float):
    """``state`` supplies apply_fn / batch_stats; ``params`` may be theta or theta_old."""
    return sample_latent(state.apply_fn, params, observations, key, action_dim, num_steps, state.batch_stats,
                         latent_clip)


@functools.partial(jax.jit, static_argnames=("num_samples", "loss_type"))
def jit_rollout_cfm_cache(state: FlowPolicyState, params, observations, latent, key, num_samples: int,
                          loss_type: str):
    """l_init = l_cfm(x_t | s_t; theta_old) on the exact executed latent (Alg. 1, line 10)."""
    B, A = latent.shape
    eps, t = sample_cfm_noise(key, B, num_samples, A)
    init_loss = compute_cfm_loss_per_sample(state.apply_fn, params, observations, latent, eps, t,
                                            state.batch_stats, loss_type)
    return eps, t, init_loss
