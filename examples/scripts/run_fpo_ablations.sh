#!/bin/bash
# Table II ablations on LIBERO-90 "pick up the butter and put it in the tray".
# Each variant changes exactly one switch of run_fpo_libero.sh (same budget and hyper-parameters).
# Usage: bash examples/scripts/run_fpo_ablations.sh [full|no_ratio|no_clip|one_step|single_critic|all]
set -e
which=${1:-all}
run() { bash examples/scripts/run_fpo_libero.sh --wandb_project FPO_pi0_Libero_ablation "$@"; }

[[ $which == full          || $which == all ]] && run --prefix fpo_libero_full
[[ $which == no_ratio      || $which == all ]] && run --prefix fpo_libero_ablation_sac --algorithm pixel_sac
[[ $which == no_clip       || $which == all ]] && run --prefix fpo_libero_ablation_no_clip --use_ppo_clip 0
[[ $which == one_step      || $which == all ]] && run --prefix fpo_libero_ablation_1_step --num_euler_steps 1
[[ $which == single_critic || $which == all ]] && run --prefix fpo_libero_ablation_single_critic --num_qs 1
exit 0
