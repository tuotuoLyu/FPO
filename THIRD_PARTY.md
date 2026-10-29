# Third-party components

FPO is released under the MIT licence in [`LICENSE`](LICENSE). Code from other projects is kept
under its original licence.

| Component | Where | Upstream | Licence |
|---|---|---|---|
| Training loop, agents, networks and replay buffer | repository root, `jaxrl2/`, `examples/` | — | MIT, Copyright (c) 2025 Mitsuhiko Nakamoto (retained in `LICENSE`) |
| jaxrl2 (basis of `jaxrl2/`) | `jaxrl2/` | [ikostrikov/jaxrl2](https://github.com/ikostrikov/jaxrl2) | MIT |
| PTR (parts of `jaxrl2/`) | `jaxrl2/` | [Asap7772/PTR](https://github.com/Asap7772/PTR) | see upstream |
| openpi (pi0 model and checkpoints) | `openpi/` (git submodule, not vendored) | [nakamotoo/openpi](https://github.com/nakamotoo/openpi) @ `a6d2400d2534ce32e7bdf8747709b97aaef8ec04`, a fork of [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | Apache-2.0 (see `openpi/LICENSE`) |
| LIBERO (benchmark) | `LIBERO/` (git submodule, not vendored) | [nakamotoo/LIBERO](https://github.com/nakamotoo/LIBERO) @ `87edbd1b823e464bc66557d213cc45e9f88aa543`, a fork of [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | MIT (see `LIBERO/LICENSE`) |
| gym-aloha (ALOHA Transfer Cube) | pip dependency | [huggingface/gym-aloha](https://github.com/huggingface/gym-aloha) | Apache-2.0 |

pi0 checkpoints (`pi0_libero`, `pi0_aloha_sim`) are downloaded by openpi at run time and are
covered by openpi's terms.
