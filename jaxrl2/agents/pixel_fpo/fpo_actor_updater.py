"""FPO actor update (paper Sec. III-B-a/b, Eqs. 2-4; Algorithm 1 lines 23-27)."""

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.agents.pixel_fpo import fpo_math
from jaxrl2.agents.pixel_fpo.noise_flow_policy import FlowPolicyState, compute_cfm_loss_per_sample


def critic_q_values(critic: TrainState, observations, latents) -> jnp.ndarray:
    variables = {"params": critic.params}
    if getattr(critic, "batch_stats", None) is not None:
        variables["batch_stats"] = critic.batch_stats
    return critic.apply_fn(variables, observations, latents)  # [M, B]


def q_advantages(critic: TrainState, observations, latents, reduction: str = "min") -> jnp.ndarray:
    """``advantage_type='q'``: A = min_i Q_i(s, x) (standardised later); an
    alternative to GAE."""
    return fpo_math.reduce_ensemble(critic_q_values(critic, observations, latents), reduction, xp=jnp)


def update_actor_fpo(
        actor: FlowPolicyState,
        critic: TrainState,
        batch: Dict[str, Any],
        actor_observations: Dict[str, Any],
        *,
        advantage_type: str,
        clip_epsilon: float,
        use_ppo_clip: bool,
        cfm_diff_temperature: float,
        cfm_clip: float,
        log_ratio_clip: float,
        ratio_std_floor: float,
        q_reduction: str,
        max_grad_norm: float,
        cfm_loss_type: str,
) -> Tuple[FlowPolicyState, Dict[str, Any]]:
    """One clipped-surrogate step on a minibatch from the replay window.

    ``batch`` carries, per transition, the executed latent x_t (``actions``), the
    Monte-Carlo draws (``eps``, ``t``) and the CFM loss cached under theta_old at
    rollout time (``initial_cfm_loss``). The loss is re-evaluated under the current
    actor on the same (s_t, x_t, eps, t) using the un-augmented observations
    ``actor_observations`` (the ones the cached loss was computed on).
    """
    B = batch["actions"].shape[0]
    x1 = batch["actions"].reshape(B, -1)
    eps, t, init_loss = batch["eps"], batch["t"], batch["initial_cfm_loss"]

    if advantage_type == "gae":
        adv = batch["advantages"]
    elif advantage_type == "q":
        adv = q_advantages(critic, batch["observations"], batch["actions"], q_reduction)
    else:
        raise ValueError(f"Unknown advantage type: {advantage_type}")
    adv = fpo_math.standardize_advantages(jax.lax.stop_gradient(adv), xp=jnp)

    def _bad(x):
        return jnp.logical_or(jnp.any(jnp.isnan(x)), jnp.any(jnp.isinf(x)))

    has_nan_input = _bad(adv) | _bad(x1) | _bad(eps) | _bad(t) | _bad(init_loss)

    def loss_fn(params):
        curr_loss = compute_cfm_loss_per_sample(actor.apply_fn, params, actor_observations, x1, eps, t,
                                                batch_stats=actor.batch_stats, loss_type=cfm_loss_type)
        delta = fpo_math.cfm_loss_differential(init_loss, curr_loss, cfm_clip, xp=jnp)            # Eq. 2
        rho, z = fpo_math.ratio_proxy(delta, beta=cfm_diff_temperature, log_ratio_clip=log_ratio_clip,
                                      std_eps=ratio_std_floor, xp=jnp,
                                      stop_gradient=jax.lax.stop_gradient)                         # Eq. 3
        loss = fpo_math.clipped_surrogate_loss(rho, adv, clip_epsilon, use_ppo_clip, xp=jnp)      # Eq. 4
        info = {
            "actor_loss": loss,
            "rho_mean": jnp.mean(rho), "rho_std": jnp.std(rho),
            "rho_min": jnp.min(rho), "rho_max": jnp.max(rho),
            "clipped_fraction": fpo_math.clipped_fraction(rho, clip_epsilon, xp=jnp),
            "cfm_loss_current": jnp.mean(curr_loss),
            "cfm_loss_init": jnp.mean(init_loss),
            "cfm_diff_mean": jnp.mean(delta), "cfm_diff_std": jnp.std(delta),
            "advantages_mean": jnp.mean(adv), "advantages_std": jnp.std(adv),
        }
        return loss, info

    def skip_update():
        _, info = loss_fn(actor.params)
        info = jax.tree_util.tree_map(jnp.zeros_like, info)
        info["grad_norm"] = jnp.zeros(())
        info["skipped_nan"] = jnp.ones(())
        return actor, info

    def safe_update():
        grads, info = jax.grad(loss_fn, has_aux=True)(actor.params)
        grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
        scale = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-8))
        grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g * scale), grads)
        info["grad_norm"] = grad_norm
        info["skipped_nan"] = jnp.zeros(())
        return actor.apply_gradients(grads=grads), info

    return jax.lax.cond(has_nan_input, skip_update, safe_update)
