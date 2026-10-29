#!/bin/bash
# Table I protocol: one FPO run per task of a LIBERO suite; the suite success rate is the
# mean final evaluation success over its 10 tasks.
# Usage: bash examples/scripts/run_fpo_libero_suite.sh libero_spatial|libero_object|libero_goal|libero_10 [extra flags]
set -e
suite=${1:?usage: run_fpo_libero_suite.sh <suite> [flags]}
shift
for task_id in 0 1 2 3 4 5 6 7 8 9; do
  bash examples/scripts/run_fpo_libero.sh \
    --libero_suite ${suite} --libero_task_id ${task_id} \
    --prefix fpo_${suite}_task${task_id} --wandb_project FPO_pi0_${suite} "$@"
done
