"""Sliding-window trajectory buffer for FPO (paper Sec. III-A-a, III-B-d).

Extends ``ReplayBuffer`` with, per transition,
  * the N Monte-Carlo draws ``eps`` [N, D], ``t`` [N, 1] and the CFM loss cached
    under theta_old at rollout time ``initial_cfm_loss`` [N] (Algorithm 1, line 11);
  * ``advantages``, filled by :meth:`compute_gae` once per iteration;
and restricts sampling to the most recent ``window`` transitions, so older
rollouts are evicted from training (``window <= 0`` keeps every transition).
"""

from typing import Callable, Dict, Iterable, Optional

import gym
import numpy as np
from flax.core import frozen_dict

from jaxrl2.agents.pixel_fpo.fpo_math import gae_advantages
from jaxrl2.data.replay_buffer import ReplayBuffer, _init_replay_dict


class FPOReplayBuffer(ReplayBuffer):

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, capacity: int,
                 num_fpo_samples: Optional[int] = None, window: int = 0):
        super().__init__(observation_space, action_space, capacity)
        self.window = int(window)
        self.num_fpo_samples = num_fpo_samples
        self.is_fpo = num_fpo_samples is not None
        if self.is_fpo:
            self.data.update(self._fpo_fields(self.capacity))

    def _fpo_fields(self, n: int) -> Dict[str, np.ndarray]:
        d = int(np.prod(self.action_space.shape))
        return {
            'eps': np.zeros((n, self.num_fpo_samples, d), dtype=np.float32),
            't': np.zeros((n, self.num_fpo_samples, 1), dtype=np.float32),
            'initial_cfm_loss': np.zeros((n, self.num_fpo_samples), dtype=np.float32),
            'advantages': np.zeros((n,), dtype=np.float32),
        }

    # ------------------------------------------------------------------
    def insert(self, data_dict):
        if self.size == self.capacity:  # grow like the base buffer (double the capacity)
            new = {
                'observations': _init_replay_dict(self.observation_space, self.capacity),
                'next_observations': _init_replay_dict(self.observation_space, self.capacity),
                'actions': np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype),
                'next_actions': np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype),
                'rewards': np.empty((self.capacity,), dtype=np.float32),
                'masks': np.empty((self.capacity,), dtype=np.float32),
                'discount': np.empty((self.capacity,), dtype=np.float32),
            }
            if self.is_fpo:
                new.update(self._fpo_fields(self.capacity))
            for k, v in new.items():
                if isinstance(v, dict):
                    for kk in v:
                        self.data[k][kk] = np.concatenate((self.data[k][kk], v[kk]), axis=0)
                else:
                    self.data[k] = np.concatenate((self.data[k], v), axis=0)
            self.capacity *= 2

        for k, v in data_dict.items():
            if k not in self.data:
                continue
            if isinstance(v, dict):
                for kk in v:
                    self.data[k][kk][self.size] = v[kk]
            else:
                self.data[k][self.size] = v
        self.size += 1

    # ------------------------------------------------------------------
    def window_start(self) -> int:
        return 0 if self.window <= 0 else max(0, self.size - self.window)

    def sample(self, batch_size: int, keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:
        if indx is None:
            indx = np.random.randint(self.window_start(), self.size, batch_size)
        out = {}
        for k, v in self.data.items():
            out[k] = {kk: vv[indx] for kk, vv in v.items()} if isinstance(v, dict) else v[indx]
        return frozen_dict.freeze(out)

    # ------------------------------------------------------------------
    def _window_trajectories(self):
        start = self.window_start()
        return [(s, e) for (s, e) in self.traj_bounds.values() if e > start and e > s]

    def _slice_obs(self, key: str, idx: np.ndarray):
        return {k: v[idx] for k, v in self.data[key].items()}

    def _padded_apply(self, fn, idx: np.ndarray, chunk_size: int):
        """Apply ``fn(idx_chunk)`` in fixed-size chunks (padding the last one) so jitted
        value functions compile once."""
        out = []
        for c in range(0, len(idx), chunk_size):
            sl = idx[c:c + chunk_size]
            n = len(sl)
            if n < chunk_size:
                sl = np.concatenate([sl, np.full(chunk_size - n, sl[-1])])
            out.append(np.asarray(fn(sl))[:n])
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def compute_gae(self, value_fn: Callable, fresh_value_fn: Callable, discount: float, lam: float,
                    chunk_size: int = 256):
        """GAE(lambda) for every trajectory that overlaps the window (Sec. III-B-c).

        ``value_fn(obs, latents)`` -> min_i Q_i(s, x) on the stored latents x_t;
        ``fresh_value_fn(obs)``     -> min_i Q_i(s, x'), x' ~ pi_theta(. | s), used to
        bootstrap the observation after the last transition of a non-terminal
        (time-limit) trajectory. Terminal transitions (mask 0) are not bootstrapped.
        """
        trajs = self._window_trajectories()
        if not trajs:
            return
        idx = np.concatenate([np.arange(s, e) for s, e in trajs])
        values = self._padded_apply(
            lambda sl: value_fn(self._slice_obs('observations', sl), self.data['actions'][sl]), idx, chunk_size)
        value_of = dict(zip(idx.tolist(), values.tolist()))

        last = np.array([e - 1 for s, e in trajs])
        open_last = last[self.data['masks'][last] > 0]
        boot_values = self._padded_apply(
            lambda sl: fresh_value_fn(self._slice_obs('next_observations', sl)), open_last, chunk_size)
        boot = dict(zip(open_last.tolist(), boot_values.tolist()))

        for s, e in trajs:
            r = self.data['rewards'][s:e]
            m = self.data['masks'][s:e]
            v = np.array([value_of[i] for i in range(s, e)], dtype=np.float32)
            self.data['advantages'][s:e] = gae_advantages(r, v, boot.get(e - 1, 0.0), m, discount, lam)
