"""Rollout / update loop for FPO on LIBERO and ALOHA (paper Algorithm 1).

One iteration = theta_old <- theta, one rollout episode with the frozen pi0
decoder, insertion into the sliding-window buffer, GAE on the window, then
``len(episode) * multi_grad_step`` gradient steps. The observation/pi0 adapters
come from ``examples/train_utils_sim.py``.
"""

import os

import jax
import numpy as np
from tqdm import tqdm

from examples.train_utils_sim import obs_to_img, obs_to_pi_zero_input, obs_to_qpos

PI0_HORIZON = 50  # pi0 denoises a 50 x 32 noise chunk; the latent fills the first rows, the last row is repeated


def latent_to_pi0_noise(latent, chunk_shape):
    latent = np.reshape(np.asarray(latent), chunk_shape)
    pad = np.repeat(latent[-1:, :], PI0_HORIZON - latent.shape[0], axis=0)
    return np.concatenate([latent, pad], axis=0)[None]


def _obs_dict(obs, variant):
    d = {'pixels': obs_to_img(obs, variant)[np.newaxis, ..., np.newaxis]}
    if variant.add_states:
        d['state'] = obs_to_qpos(obs, variant)[np.newaxis, ..., np.newaxis]
    return d


def _env_reset(env, variant):
    return env.reset() if 'libero' in variant.env else env.reset()[0]


def _env_step(env, action, variant):
    if 'libero' in variant.env:
        obs, reward, done, _ = env.step(action)
    else:
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return obs, reward, done


def _decode(agent_dp, obs, latent, chunk_shape, variant, noise=None):
    """Frozen pi0 decodes (s_t, x_t) into an action chunk (Algorithm 1, line 8).
    ``noise`` overrides the latent with a full pi0 noise chunk (baseline evaluation)."""
    if noise is None:
        noise = latent_to_pi0_noise(latent, chunk_shape)
    actions = np.asarray(agent_dp.infer(obs_to_pi_zero_input(obs, variant), noise=noise)["actions"])
    if np.any(np.isnan(actions)):
        actions = np.zeros_like(actions)
    elif variant.clip_base_actions:
        actions = np.clip(actions, -1.0, 1.0)
    return actions


def is_fpo(variant):
    return variant.algorithm == 'pixel_fpo'


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def trajwise_alternating_training_loop(variant, agent, env, eval_env, replay_buffer, replay_buffer_iterator,
                                       wandb_logger, agent_dp=None):
    i = 0                       # gradient steps
    total_env_steps = 0
    exploration_latents = []
    wandb_logger.log({'num_online_samples': 0, 'num_online_trajs': 0, 'env_steps': 0}, step=i)
    if variant.eval_at_start:
        perform_control_eval(agent, eval_env, 0, variant, wandb_logger, agent_dp)   # pi0 with N(0, I) latents

    with tqdm(total=variant.max_steps, desc="gradient steps") as pbar:
        while i < variant.max_steps:
            # ---- rollout phase (Algorithm 1, lines 3-12) ----
            if is_fpo(variant):
                agent.sync_old_actor()
            traj = collect_traj(variant, agent, env, i, agent_dp)
            add_online_data_to_buffer(variant, traj, replay_buffer)
            total_env_steps += traj['env_steps']
            exploration_latents.append(np.asarray(traj['actions']).reshape(len(traj['actions']), -1))
            if len(replay_buffer) < variant.start_online_updates:
                continue

            # ---- advantages on the window (Sec. III-B-c) ----
            if is_fpo(variant) and agent.advantage_type == 'gae':
                replay_buffer.compute_gae(agent.values, agent.fresh_values, agent.discount, agent.gae_lambda)

            # ---- update phase (Algorithm 1, lines 14-28) ----
            for _ in range(len(traj['rewards']) * variant.multi_grad_step):
                if i >= variant.max_steps:
                    break
                update_info = agent.update(next(replay_buffer_iterator))
                pbar.update()
                i += 1

                if i % variant.log_interval == 0:
                    for k, v in jax.device_get(update_info).items():
                        if np.ndim(v) == 0:
                            wandb_logger.log({f'training/{k}': v}, step=i)
                    wandb_logger.log({'replay_buffer_size': len(replay_buffer),
                                      'episode_return (exploration)': traj['episode_return'],
                                      'is_success (exploration)': int(traj['is_success'])}, step=i)

                if i % variant.eval_interval == 0:
                    wandb_logger.log({'num_online_samples': len(replay_buffer),
                                      'num_online_trajs': replay_buffer._traj_counter,
                                      'env_steps': total_env_steps}, step=i)
                    perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                    if variant.save_latents:
                        np.save(os.path.join(variant.outputdir, 'latents', f'latent_actions_step_{i}.npy'),
                                np.concatenate(exploration_latents, axis=0))

                if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                    agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)


def add_online_data_to_buffer(variant, traj, replay_buffer):
    actions = np.array(traj['actions'])     # (T, *chunk_shape)
    T = len(actions)
    for t in range(T):
        obs = {k: v[0] for k, v in traj['observations'][t].items()}
        next_obs = {k: v[0] for k, v in traj['observations'][t + 1].items()}
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)
        item = dict(observations=obs, next_observations=next_obs, actions=actions[t],
                    next_actions=actions[t + 1] if t < T - 1 else actions[t],
                    rewards=traj['rewards'][t], masks=traj['masks'][t],
                    discount=variant.discount ** variant.query_freq)
        if 'fpo_info' in traj:
            for k in ('eps', 't', 'initial_cfm_loss'):
                item[k] = traj['fpo_info'][k][t]
        replay_buffer.insert(item)
    replay_buffer.increment_traj_counter()


def collect_traj(variant, agent, env, training_step, agent_dp):
    """One episode; a latent is chosen every ``query_freq`` env steps and decoded by pi0."""
    chunk_shape = agent.action_chunk_shape
    clip = getattr(agent, 'latent_clip', 2.0)
    agent._rng, rng = jax.random.split(agent._rng)
    obs = _env_reset(env, variant)
    # Gaussian warm-up latents until gradient updates start
    warmup = training_step < variant.start_online_updates

    rewards, latents, obs_list, fpo_info = [], [], [], []
    reward = 0.0
    for t in range(variant.max_timesteps):
        if t % variant.query_freq == 0:
            obs_dict = _obs_dict(obs, variant)
            latent = None
            if warmup:
                rng, key = jax.random.split(rng)
                latent = np.clip(np.asarray(jax.random.normal(key, (1, *chunk_shape))), -clip, clip)
            if is_fpo(variant):
                latent, info = agent.sample_actions_with_info(obs_dict, latent=latent)
                fpo_info.append(info)
            elif latent is None:
                latent = agent.sample_actions(obs_dict)
            latent = np.reshape(np.clip(latent, -clip, clip), chunk_shape)
            actions = _decode(agent_dp, obs, latent, chunk_shape, variant)
            latents.append(latent)
            obs_list.append(obs_dict)

        obs, reward, done = _env_step(env, actions[t % variant.query_freq], variant)
        rewards.append(reward)
        if done:
            break

    final = {'pixels': obs_to_img(obs, variant)[np.newaxis, ..., np.newaxis],
             'state': obs_to_qpos(obs, variant)[np.newaxis, ..., np.newaxis]}
    obs_list.append(final)

    is_success = reward == variant.env_max_reward
    n = len(latents)
    # sparse reward per query step: -1 until success, 0 (terminal, bootstrap masked) on success
    step_rewards = np.concatenate([-np.ones(n - 1), [0.0]]) if is_success else -np.ones(n)
    masks = np.concatenate([np.ones(n - 1), [0.0]]) if is_success else np.ones(n)

    traj = {'observations': obs_list, 'actions': latents, 'rewards': step_rewards, 'masks': masks,
            'is_success': is_success, 'episode_return': float(np.sum(rewards)), 'env_steps': t + 1}
    if fpo_info:
        traj['fpo_info'] = {k: np.stack([d[k] for d in fpo_info]) for k in fpo_info[0]}
    return traj


def perform_control_eval(agent, env, i, variant, wandb_logger, agent_dp):
    """Success rate / average return / episode length (Fig. 3 panels). At i == 0 the
    latents are N(0, I), i.e. the unmodified pi0 (SFT baseline)."""
    rng = jax.random.PRNGKey(variant.seed + 456)
    chunk_shape = agent.action_chunk_shape
    returns, highest, successes, lengths, eval_latents = [], [], [], [], []
    for rollout_id in range(variant.eval_episodes):
        obs = _env_reset(env, variant)
        rewards, rollout_latents = [], []
        for t in range(variant.max_timesteps):
            if t % variant.query_freq == 0:
                rng, key = jax.random.split(rng)
                noise = None
                if i == 0:   # unmodified pi0: its own N(0, I) noise over the full 50 x 32 chunk
                    noise = np.asarray(jax.random.normal(key, (1, PI0_HORIZON, chunk_shape[-1])))
                    latent = noise[0, :chunk_shape[0]]
                else:
                    latent = np.reshape(agent.sample_actions(_obs_dict(obs, variant)), chunk_shape)
                rollout_latents.append(np.reshape(latent, -1))
                actions = _decode(agent_dp, obs, latent, chunk_shape, variant, noise=noise)
            obs, reward, done = _env_step(env, actions[t % variant.query_freq], variant)
            rewards.append(reward)
            if done:
                break
        lengths.append(t + 1)
        returns.append(float(np.sum(rewards)))
        highest.append(float(np.max(rewards)))
        successes.append(reward == variant.env_max_reward)
        eval_latents.extend(rollout_latents)
        if variant.save_latents and rollout_id == 0:
            tag = 'successful' if successes[-1] else 'failed'
            np.save(os.path.join(variant.outputdir, 'latents', f'{tag}_eval_latents_step_{i}.npy'),
                    np.array(rollout_latents))

    if variant.save_latents:
        np.save(os.path.join(variant.outputdir, 'latents', f'eval_latents_step_{i}.npy'), np.array(eval_latents))
    wandb_logger.log({'evaluation/success_rate': float(np.mean(successes)),
                      'evaluation/avg_return': float(np.mean(returns)),
                      'evaluation/avg_episode_len': float(np.mean(lengths))}, step=i)
    for r in range(variant.env_max_reward + 1):
        wandb_logger.log({f'evaluation/Reward >= {r}': float(np.mean(np.array(highest) >= r))}, step=i)
    print(f'[eval @ {i}] success rate {np.mean(successes):.3f}  return {np.mean(returns):.2f}  '
          f'length {np.mean(lengths):.1f}')
