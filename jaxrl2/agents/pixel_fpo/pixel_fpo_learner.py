"""Flow Policy Optimization (FPO) learner for latent-space fine-tuning of pi0.

Drop-in replacement for ``PixelSACLearner``: a flow-matching actor in pi0's
noise space, trained with the CFM-loss ratio proxy and a clipped surrogate, plus a
Q-ensemble critic (paper Sec. III, Algorithm 1).
"""

import copy
import functools
import pathlib
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import checkpoints, train_state

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.pixel_fpo import fpo_math
from jaxrl2.agents.pixel_fpo.fpo_actor_updater import critic_q_values, update_actor_fpo
from jaxrl2.agents.pixel_fpo.noise_flow_policy import (FlowPolicyState, create_flow_policy,
                                                       jit_rollout_cfm_cache, jit_sample_latent,
                                                       sample_latent)
from jaxrl2.data.augmentations import batched_random_crop, color_transform
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
from jaxrl2.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.networks.values import StateActionEnsemble
from jaxrl2.types import PRNGKey
from jaxrl2.utils.target_update import soft_target_update


class TrainState(train_state.TrainState):
    batch_stats: Any = None


# ---------------------------------------------------------------------------
# Critic update (Eqs. 5-6)
# ---------------------------------------------------------------------------

def update_critic(key: PRNGKey, actor: FlowPolicyState, critic: TrainState, target_critic_params: Any,
                  batch: DatasetDict, discount: float, critic_reduction: str, action_dim: int,
                  num_euler_steps: int, latent_clip: float) -> Tuple[TrainState, Dict[str, Any]]:
    """TD target y = r + gamma * m * min_i Qbar_i(s', x'), x' ~ pi_theta(. | s') (no entropy term)."""
    next_latents = sample_latent(actor.apply_fn, actor.params, batch["next_observations"], key, action_dim,
                                 num_euler_steps, actor.batch_stats, latent_clip)
    next_qs = critic_q_values(critic.replace(params=target_critic_params), batch["next_observations"],
                              next_latents)
    target_q = fpo_math.td_target(batch["rewards"], batch["masks"], next_qs, discount, critic_reduction, xp=jnp)
    target_q = jax.lax.stop_gradient(target_q)

    def loss_fn(critic_params):
        qs = critic_q_values(critic.replace(params=critic_params), batch["observations"], batch["actions"])
        loss = fpo_math.critic_loss(qs, target_q, xp=jnp)
        return loss, {"critic_loss": loss, "q_mean": qs.mean(), "q_min": qs.min(), "q_max": qs.max(),
                      "q_std": qs.std(), "target_q_mean": target_q.mean()}

    grads, info = jax.grad(loss_fn, has_aux=True)(critic.params)
    return critic.apply_gradients(grads=grads), info


def _augment(rng, pixels, color_jitter: bool, num_cameras: int):
    """Image augmentation (random crop + optional colour jitter) for the critic."""
    if pixels.squeeze().ndim == 2:
        return rng, pixels
    rng, key = jax.random.split(rng)
    pixels = batched_random_crop(key, pixels)
    if color_jitter:
        rng, key = jax.random.split(rng)
        if num_cameras > 1:
            for i in range(num_cameras):
                s, e = i * 3, (i + 1) * 3
                pixels = pixels.at[:, :, :, s:e].set(
                    (color_transform(key, pixels[:, :, :, s:e].astype(jnp.float32) / 255.0) * 255).astype(jnp.uint8))
        else:
            pixels = (color_transform(key, pixels.astype(jnp.float32) / 255.0) * 255).astype(jnp.uint8)
    return rng, pixels


@functools.partial(jax.jit, static_argnames=(
    "critic_reduction", "color_jitter", "aug_next", "num_cameras", "clip_epsilon", "use_ppo_clip",
    "num_euler_steps", "advantage_type", "action_dim", "q_reduction", "cfm_diff_temperature", "cfm_clip",
    "log_ratio_clip", "ratio_std_floor", "latent_clip", "max_grad_norm", "cfm_loss_type"))
def _update_jit_fpo(rng, actor: FlowPolicyState, critic: TrainState, target_critic_params, batch: DatasetDict,
                    discount: float, tau: float, *, critic_reduction: str, color_jitter: bool, aug_next: bool,
                    num_cameras: int, clip_epsilon: float, use_ppo_clip: bool, num_euler_steps: int,
                    advantage_type: str, action_dim: int, q_reduction: str, cfm_diff_temperature: float,
                    cfm_clip: float, log_ratio_clip: float, ratio_std_floor: float, latent_clip: float,
                    max_grad_norm: float, cfm_loss_type: str):
    """One gradient step of the update phase (Algorithm 1, lines 15-27)."""
    raw_observations = batch["observations"]          # the actor's ratio uses the cached, un-augmented inputs

    rng, pixels = _augment(rng, batch["observations"]["pixels"], color_jitter, num_cameras)
    batch = batch.copy(add_or_replace={"observations": batch["observations"].copy(add_or_replace={"pixels": pixels})})
    if aug_next:
        rng, next_pixels = _augment(rng, batch["next_observations"]["pixels"], color_jitter, num_cameras)
        batch = batch.copy(add_or_replace={
            "next_observations": batch["next_observations"].copy(add_or_replace={"pixels": next_pixels})})

    # critic (Eqs. 5-6) + Polyak averaging once per gradient step
    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key, actor, critic, target_critic_params, batch, discount,
                                            critic_reduction, action_dim, num_euler_steps, latent_clip)
    new_target_critic_params = soft_target_update(new_critic.params, target_critic_params, tau)

    # actor (Eqs. 2-4); theta_old is NOT touched here, it is synced once per iteration
    new_actor, actor_info = update_actor_fpo(
        actor, new_critic, batch, raw_observations, advantage_type=advantage_type, clip_epsilon=clip_epsilon,
        use_ppo_clip=use_ppo_clip, cfm_diff_temperature=cfm_diff_temperature, cfm_clip=cfm_clip,
        log_ratio_clip=log_ratio_clip, ratio_std_floor=ratio_std_floor, q_reduction=q_reduction,
        max_grad_norm=max_grad_norm,
        cfm_loss_type=cfm_loss_type)

    return rng, new_actor, new_critic, new_target_critic_params, {**critic_info, **actor_info}


@functools.partial(jax.jit, static_argnames=("reduction",))
def _value_jit(critic: TrainState, observations, latents, reduction: str):
    """Conservative ensemble value V(s) = min_i Q_i(s, x) (Sec. III-B-c)."""
    return fpo_math.reduce_ensemble(critic_q_values(critic, observations, latents), reduction, xp=jnp)


def _make_encoder(encoder_type, cnn_features, cnn_strides, cnn_padding, encoder_norm, use_spatial_softmax,
                  softmax_temperature):
    if encoder_type == 'small':
        return Encoder(cnn_features, cnn_strides, cnn_padding)
    if encoder_type == 'impala':
        return ImpalaEncoder()
    if encoder_type == 'impala_small':
        return SmallerImpalaEncoder()
    if encoder_type == 'resnet_18_v1':
        return ResNet18(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax,
                        softmax_temperature=softmax_temperature)
    if encoder_type == 'resnet_34_v1':
        return ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax,
                        softmax_temperature=softmax_temperature)
    if encoder_type == 'resnet_18_v2':
        return ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
    if encoder_type == 'resnet_34_v2':
        return ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
    raise ValueError(f"Encoder type {encoder_type} not found!")


class PixelFPOLearner(Agent):

    def __init__(self,
                 seed: int,
                 observations: Union[jnp.ndarray, DatasetDict],
                 actions: jnp.ndarray,
                 # optimisation
                 actor_lr: float = 1e-5,
                 critic_lr: float = 1e-4,
                 decay_steps: Optional[int] = None,
                 # networks
                 hidden_dims: Sequence[int] = (256, 256, 256),
                 flow_hidden_dims: Sequence[int] = (512, 512, 512),
                 latent_dim: int = 256,
                 encoder_type: str = "resnet_34_v1",
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 encoder_norm: str = "group",
                 use_spatial_softmax: bool = True,
                 softmax_temperature: float = 1.0,
                 use_bottleneck: bool = True,
                 color_jitter: bool = True,
                 aug_next: bool = True,
                 num_cameras: int = 1,
                 # critic ensemble (Eqs. 5-6)
                 discount: float = 0.99,
                 tau: float = 0.005,
                 num_qs: int = 10,
                 critic_reduction: str = "min",
                 # FPO (Eqs. 2-4, 7)
                 num_fpo_samples: int = 4,
                 num_euler_steps: int = 6,
                 clip_epsilon: float = 0.2,
                 use_ppo_clip: bool = True,
                 cfm_diff_temperature: float = 1.0,
                 cfm_clip: float = 3.0,
                 log_ratio_clip: float = 5.0,
                 ratio_std_floor: float = 1e-3,
                 cfm_loss_type: str = "huber",
                 advantage_type: str = "gae",
                 gae_lambda: float = 0.95,
                 q_reduction: str = "min",
                 latent_clip: float = 2.0,
                 max_grad_norm: float = 0.5,
                 **kwargs):  # SAC-only keys of the shared launcher (temperature, action magnitude, ...) are ignored
        self.action_dim = int(np.prod(actions.shape[-2:]))
        self.action_chunk_shape = actions.shape[-2:]
        self.discount = discount
        self.tau = tau
        self.gae_lambda = gae_lambda
        self.num_fpo_samples = int(num_fpo_samples)
        self.num_euler_steps = int(num_euler_steps)
        self.latent_clip = float(latent_clip)
        self.cfm_loss_type = cfm_loss_type
        self.advantage_type = advantage_type
        self.critic_reduction = critic_reduction
        self.q_reduction = q_reduction
        self._static = dict(
            critic_reduction=critic_reduction, color_jitter=bool(color_jitter), aug_next=bool(aug_next),
            num_cameras=int(num_cameras), clip_epsilon=float(clip_epsilon), use_ppo_clip=bool(use_ppo_clip),
            num_euler_steps=self.num_euler_steps, advantage_type=advantage_type, action_dim=self.action_dim,
            q_reduction=q_reduction, cfm_diff_temperature=float(cfm_diff_temperature), cfm_clip=float(cfm_clip),
            log_ratio_clip=float(log_ratio_clip), ratio_std_floor=float(ratio_std_floor), latent_clip=self.latent_clip,
            max_grad_norm=float(max_grad_norm), cfm_loss_type=cfm_loss_type)

        rng = jax.random.PRNGKey(seed)
        rng, critic_key = jax.random.split(rng)
        encoder_def = _make_encoder(encoder_type, cnn_features, cnn_strides, cnn_padding, encoder_norm,
                                    use_spatial_softmax, softmax_temperature)
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        # actor: conditional flow in pi0's noise space
        self._actor = create_flow_policy(observations, self.action_chunk_shape, encoder_def, seed=seed,
                                         actor_lr=actor_lr, latent_dim=latent_dim, hidden_dims=tuple(flow_hidden_dims),
                                         use_bottleneck=use_bottleneck)
        self._old_actor_params = copy.deepcopy(self._actor.params)   # theta_old (rollout copy)

        # critic ensemble {Q_phi_i}_{i=1..M} over (s, x); separate encoder
        critic_def = PixelMultiplexer(encoder=encoder_def, network=StateActionEnsemble(hidden_dims, num_qs=num_qs),
                                      latent_dim=latent_dim, use_bottleneck=use_bottleneck)
        variables = critic_def.init(critic_key, observations, actions)
        self._critic = TrainState.create(apply_fn=critic_def.apply, params=variables["params"],
                                         tx=optax.adam(learning_rate=critic_lr),
                                         batch_stats=variables.get("batch_stats", None))
        self._target_critic_params = copy.deepcopy(variables["params"])
        self._rng = rng
        print(f"[FPO] action_dim={self.action_dim} N={num_fpo_samples} K={num_euler_steps} eps={clip_epsilon} "
              f"clip={use_ppo_clip} beta={cfm_diff_temperature} M={num_qs} adv={advantage_type}")

    # ---------------- Algorithm 1, line 3 ----------------
    def sync_old_actor(self):
        """theta_old <- theta (once per iteration, before the rollout phase)."""
        self._old_actor_params = self._actor.params

    # ---------------- sampling ----------------
    def _next_key(self, seed=None):
        if seed is not None:
            return seed
        self._rng, key = jax.random.split(self._rng)
        return key

    def sample_actions(self, observations, *, seed: Optional[PRNGKey] = None) -> np.ndarray:
        """x ~ pi_theta(. | s) with the current actor (evaluation)."""
        x = jit_sample_latent(self._actor, self._actor.params, observations, self._next_key(seed),
                              self.action_dim, self.num_euler_steps, self.latent_clip)
        return np.asarray(x)

    def sample_actions_with_info(self, observations, latent: Optional[np.ndarray] = None
                                 ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Rollout step (Algorithm 1, lines 7 and 10).

        Samples x_t ~ pi_theta_old(. | s_t) (or takes ``latent``, e.g. the Gaussian
        warm-up latents) and caches l_init = l_cfm(x_t | s_t; theta_old) on that exact
        latent together with its N Monte-Carlo draws (eps, t).
        """
        if latent is None:
            latent = jit_sample_latent(self._actor, self._old_actor_params, observations, self._next_key(),
                                       self.action_dim, self.num_euler_steps, self.latent_clip)
        latent = jnp.asarray(latent, dtype=jnp.float32).reshape(-1, self.action_dim)
        eps, t, init_loss = jit_rollout_cfm_cache(self._actor, self._old_actor_params, observations, latent,
                                                  self._next_key(), self.num_fpo_samples, self.cfm_loss_type)
        info = {"eps": np.asarray(eps)[0], "t": np.asarray(t)[0], "initial_cfm_loss": np.asarray(init_loss)[0]}
        return np.asarray(latent), info

    # ---------------- advantages (Sec. III-B-c) ----------------
    def values(self, observations, latents) -> np.ndarray:
        """V(s) = min_i Q_i(s, x) on stored latents."""
        return np.asarray(_value_jit(self._critic, observations, latents, self.q_reduction))

    def fresh_values(self, observations) -> np.ndarray:
        """V(s) = min_i Q_i(s, x'), x' ~ pi_theta(. | s) (when no stored latent exists)."""
        latents = jit_sample_latent(self._actor, self._actor.params, observations, self._next_key(),
                                    self.action_dim, self.num_euler_steps, self.latent_clip)
        return self.values(observations, latents)

    # ---------------- update phase ----------------
    def update(self, batch: FrozenDict) -> Dict[str, Any]:
        self._rng, key = jax.random.split(self._rng)
        _, self._actor, self._critic, self._target_critic_params, info = _update_jit_fpo(
            key, self._actor, self._critic, self._target_critic_params, batch, self.discount, self.tau,
            **self._static)
        return info

    # ---------------- checkpoints ----------------
    @property
    def _save_dict(self):
        return {"critic": self._critic, "target_critic_params": self._target_critic_params,
                "actor": self._actor, "old_actor_params": self._old_actor_params}

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).exists(), f"Checkpoint {dir} not found."
        out = checkpoints.restore_checkpoint(dir, self._save_dict)
        self._actor = out["actor"]
        self._critic = out["critic"]
        self._target_critic_params = out["target_critic_params"]
        self._old_actor_params = out["old_actor_params"]

    def save_checkpoint(self, dir, step, keep_every_n_steps=None):
        checkpoints.save_checkpoint(dir, self._save_dict, step=step, keep_every_n_steps=keep_every_n_steps)
