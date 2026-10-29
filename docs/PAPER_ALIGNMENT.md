# Paper alignment

How each component of the ICRA 2026 paper (equations, Algorithm 1, experimental
settings, Tables I-II, Figs. 3-5) maps to this code base, with the defaults of the two
paper settings. The scripts in `examples/scripts/` pass every value explicitly. Values that
the paper does not specify are marked as such and explained under
[Implementation choices](#implementation-choices).

## Component table

| Paper (section / equation) | Code | LIBERO | ALOHA Transfer Cube |
|---|---|---|---|
| Frozen base policy pi0 decodes (s_t, x_t) into actions (Sec. III-A, Fig. 1b) | `examples/train_fpo.py:load_pi0`, `examples/train_utils_fpo.py:_decode` | `pi0_libero` | `pi0_aloha_sim` |
| Latent action chunk x_t in R^D fed to pi0 as its initial noise (Sec. III-A) | `train_utils_fpo.py:latent_to_pi0_noise` | D = 32: one 32-d latent per query, used as the first row of pi0's 50 x 32 noise chunk, last row repeated | same |
| Action chunking: one latent per pi0 call (Sec. III-A-a) | `--query_freq` | 32 env steps per call | 50 env steps per call |
| Objective J = E[sum gamma^t r_t], sparse reward (Eq. 1) | `train_utils_fpo.py:collect_traj` | r = -1 per query step, 0 and terminal on success | same |
| Discount gamma (Eq. 1, 5) | `--discount` (per query step) | 0.99 | 0.99 |
| Actor pi_theta(x \| s): conditional flow in latent space | `jaxrl2/agents/pixel_fpo/noise_flow_policy.py:NoiseFlowPolicy` | ResNet-34 (group norm, spatial softmax) + 256-d bottleneck + 3 x 512 MLP velocity head; 128 px image | same; 64 px image |
| theta_old <- theta once per iteration (Alg. 1, line 3; Sec. III-A-b) | `pixel_fpo_learner.py:PixelFPOLearner.sync_old_actor`, called in `train_utils_fpo.py:trajwise_alternating_training_loop` before every rollout | one rollout episode per iteration | same |
| Multi-step latent exploration, Euler steps (Eq. 7) | `noise_flow_policy.py:euler_integration` (JAX), `fpo_math.py:euler_integrate` (reference) | x^(0) ~ N(0, I) clipped to +-3, eta = 1/K, tau^(k) = (k+1)/K, **K = 6** (not specified in the paper; see choice 7) | **K = 8** |
| Executed latent | `noise_flow_policy.py:sample_latent` | clipped to +-2 (`--latent_clip`) | same |
| Cache l_cfm(x_t \| s_t; theta_old) on the executed latent (Alg. 1, line 10) | `PixelFPOLearner.sample_actions_with_info`, `noise_flow_policy.py:jit_rollout_cfm_cache` | N = 4 Monte-Carlo draws (eps ~ N(0, I) clipped to +-2.5, t ~ U(0.01, 0.99)), stored with the transition | N = 8 |
| Per-sample CFM loss l_cfm (Sec. III-B-a) | `fpo_math.py:per_sample_cfm_loss` | Huber (delta = 1) of v_theta(x_t, t \| s) - (x - eps), mean over latent dims (`--cfm_loss_type huber`; `mse` available) | same |
| Loss reduction Delta l = l(theta_old) - l(theta) on the same (s_t, x_t) (Eq. 2) | `fpo_math.py:cfm_loss_differential`, used in `fpo_actor_updater.py:update_actor_fpo` | each draw clipped to +-3 (`--cfm_clip`), mean over the N draws | same |
| Batch standardisation z = (Delta - mu)/sigma, rho = exp(beta z) (Eq. 3) | `fpo_math.py:ratio_proxy`, `batch_standardize` | statistics over the minibatch, no gradient through them; sigma = sqrt(var + 1e-3^2) (`--ratio_std_floor`); **beta = 1.0** (`--cfm_diff_temperature`; not specified in the paper); beta z clipped to +-5 | same |
| Clipped surrogate (Eq. 4) | `fpo_math.py:clipped_surrogate_loss` | **epsilon = 0.2** (`--clip_epsilon`; not specified in the paper) | same |
| Advantages standardised per minibatch, gradient stopped (Sec. III-B-b) | `fpo_math.py:standardize_advantages` | standardised, clipped to +-10 | same |
| Critic ensemble {Q_phi_i}, conservative TD target with x' ~ pi_theta(. \| s') (Eq. 5) | `pixel_fpo_learner.py:update_critic`, `fpo_math.py:td_target` | **M = 10** members (`--num_qs`), min over target members, terminal masked; MLP 1 x 256 on separate ResNet-34 encoder (image + proprioception) | M = 10, MLP 3 x 256 |
| Squared TD loss (Eq. 6) | `fpo_math.py:critic_loss` | mean over members and batch | same |
| Polyak averaging once per gradient step | `jaxrl2/utils/target_update.py:soft_target_update` | tau = 0.005 | same |
| GAE with V(s) = min_i Q_i, stored latents, fresh latent otherwise (Sec. III-B-c) | `jaxrl2.data.replay_buffer_fpo.FPOReplayBuffer.compute_gae`, `fpo_math.py:gae_advantages` | lambda = 0.95, recomputed on the replay window once per iteration; bootstrap of a time-limit end uses a fresh latent | same |
| Small sliding-window trajectory buffer (Sec. III-A-a, III-B-d) | `FPOReplayBuffer(window=...)` | most recent 4000 transitions (`--buffer_window`; size not specified in the paper) | 4000 |
| Several SGD steps per interaction batch (Sec. III-A-b) | `--multi_grad_step` | 40 gradient steps per collected transition | 20 |
| Minibatch size | `--batch_size` | 210 | 192 |
| Optimisers (not specified in the paper) | `noise_flow_policy.py:create_flow_policy`, `pixel_fpo_learner.py` | Adam; actor 1e-5, critic 1e-4; actor gradient norm clipped to 0.5 | actor 3e-5, critic 3e-4 |
| Warm-up | `--start_online_updates` | 1000 transitions with N(0, I) latents before the first update | same |
| Training budget (Fig. 3, Fig. 4) | `--max_steps` (gradient steps) | 500K | 3M (Fig. 4 shows the first 1.6M) |
| Evaluation: success rate, average return, episode length (Fig. 3) | `train_utils_fpo.py:perform_control_eval` | every 10K steps, 10 episodes, 400 env steps; step-0 evaluation of unmodified pi0 | success = reward 4 |
| Seeds | `--seed` | 42 | 0 |
| LIBERO suites, per-suite success rate (Table I) | `examples/scripts/run_fpo_libero_suite.sh`, `--libero_suite`, `--libero_task_id` | one run per task, suite SR = mean over the 10 tasks (protocol detail not specified in the paper) | - |
| Ablation task, LIBERO-90 "pick up the butter and put it in the tray" (Table II) | `examples/scripts/run_fpo_ablations.sh` | `--libero_suite libero_90 --libero_task_id 56` | - |
| Table II: without CFM ratio proxy (SAC-style latent update) | `--algorithm pixel_sac` (`PixelSACLearner`, same budget and launcher) | | |
| Table II: without PPO clipping | `--use_ppo_clip 0` (L = -E[rho A]) | | |
| Table II: single-step integration (K = 1) | `--num_euler_steps 1` | | |
| Table II: single critic (without ensemble) | `--num_qs 1` | | |
| Latent-space evolution (Fig. 5) | `--save_latents 1`, `analysis/latent_evolution.py` | | |
| Latent statistics of failed vs successful rollouts (Fig. 7) | `--save_latents 1` saves the evaluation latents of failed and successful rollouts (`failed_/successful_eval_latents_step_*.npy`) | | |

## Implementation choices

Where the paper does not specify a value or detail, this release uses the following.

1. **Gradient path of the ratio (Sec. III-B-b, "stop gradients through rho_t").** Gradients
   are stopped through the batch statistics (mu, sigma), the cached loss and the
   advantages; they flow through l_cfm(x_t | s_t; theta), the only term of Eq. 4 that
   depends on theta.
2. **Floor on sigma in Eq. 3.** Not specified in the paper; default sigma = sqrt(var + 1e-3^2).
   The cached loss (computed at rollout time) and the re-evaluated loss differ by
   floating-point noise right after theta_old <- theta; the floor keeps the
   standardisation from amplifying that noise.
3. **Ratio sharpness and clipping (Eqs. 3-4).** Not specified in the paper; defaults
   beta = 1.0 (unit temperature) and epsilon = 0.2 (the common PPO value). beta * z is
   clipped to +-5 and each Monte-Carlo difference to +-3 as numerical guards.
4. **Monte-Carlo estimate of l_cfm.** Not specified in the paper; defaults N = 4 (LIBERO) /
   8 (ALOHA) draws per sample, eps ~ N(0, I) clipped to +-2.5, t ~ U(0.01, 0.99) to stay
   away from the end points of the path. The draws are stored with the transition so that
   the same (eps, t) are used for l(theta_old) and l(theta).
5. **Form of the per-sample CFM loss.** The paper cites the standard CFM objective; the
   default is an elementwise Huber penalty (delta = 1), equal to half the squared error for
   errors below 1 and less sensitive to outliers; `--cfm_loss_type mse` gives the squared
   error. Under batch standardisation the factor 1/2 has no effect.
6. **Observation used for the ratio.** The re-evaluated loss uses the stored, un-augmented
   observation, i.e. the same input as the cached loss; image augmentation (random crop,
   colour jitter) is applied to the critic only.
7. **Euler discretisation (Eq. 7).** Not specified in the paper; default eta = 1/K with
   tau^(k) = (k+1)/K, so the K steps cover the whole flow-time interval; x^(0) ~ N(0, I)
   clipped to +-3, velocities clipped to +-10, states to +-5, executed latent to +-2.
   K = 6 for LIBERO and K = 8 for ALOHA. The same sampler produces rollout latents,
   evaluation latents, critic targets x' and GAE bootstraps.
8. **GAE.** Not specified in the paper; default lambda = 0.95, gamma = 0.99 per pi0 call.
   Minibatches are drawn i.i.d. from the window, so advantages are computed over the
   window's trajectories once per iteration (before the update phase) and stored with
   the transitions. `--gae_lambda 0` gives the one-step TD residual;
   `--advantage_type q` uses the conservative Q(s, x_t) instead.
9. **Window size.** Not specified in the paper; default 4000 transitions (several hundred
   recent episodes, about 20 times the minibatch size); `--buffer_window 0` keeps every
   transition.
10. **Encoders.** The frozen component is pi0 (vision-language backbone and action expert).
    The actor and the critic each use their own image encoder (ResNet-34) on a downsampled
    third-person image; the critic also
    receives proprioception.
11. **Optimisation and budget.** Not specified in the paper; defaults: Adam, actor learning
    rate 1e-5 / 3e-5 and critic 1e-4 / 3e-4 (LIBERO / ALOHA), actor gradient norm clipped
    to 0.5, minibatch 210 / 192, 40 / 20 gradient steps per collected transition,
    1000 warm-up transitions with N(0, I) latents, 500K / 3M gradient steps.
12. **LIBERO episodes and Table I protocol.** Not specified in the paper; defaults: 400
    environment steps per episode for every suite, one run per task, suite success rate =
    mean final evaluation success over the suite's 10 tasks (`run_fpo_libero_suite.sh`).
13. **Table II task and variants.** "pick up the butter and put it in the tray" is
    `libero_90` task index 56. Each variant changes one switch of the full configuration:
    `--algorithm pixel_sac` (SAC-style update of the same latent actor,
    `PixelSACLearner`), `--use_ppo_clip 0`, `--num_euler_steps 1`, `--num_qs 1`.
14. **Baseline reference.** `--eval_at_start 1` evaluates the unmodified pi0 (its own
    N(0, I) noise) at step 0, the reference line of Figs. 3-4.
15. **Fig. 5 analysis.** `analysis/latent_evolution.py` groups exploration latents by the
    training step at which they were collected (0-10K, 80-100K, 130K+ steps), embeds them
    jointly with t-SNE and reports spread statistics.

## Tests

`tests/test_fpo_math.py` checks Eqs. 2-7 and GAE in NumPy, `tests/test_fpo_jax.py`
checks the JAX implementation against them and runs a learner update (requires JAX),
`scripts/smoke_cpu.py` runs Algorithm 1 on a toy task and `scripts/check_loop_cpu.py`
runs the replay window and the training loop with stand-in components.
