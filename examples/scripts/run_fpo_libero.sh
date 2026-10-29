#!/bin/bash
# FPO on one LIBERO task with the paper's LIBERO setting.
# Default task: LIBERO-90 "pick up the butter and put it in the tray" (Table II, "FPO (complete)").
# Usage: bash examples/scripts/run_fpo_libero.sh [extra flags, e.g. --libero_suite libero_10 --libero_task_id 0]
set -e
proj_name=FPO_pi0_Libero

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-./openpi}   # where pi0 checkpoints are cached
export EXP=${EXP:-./logs/$proj_name}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# CUDA_VISIBLE_DEVICES: --batch_size (210) must be divisible by the number of visible GPUs.

pip install mujoco==3.3.1   # LIBERO needs MuJoCo 3.3.1, gym-aloha 2.3.7

python3 examples/launch_train_fpo.py \
  --algorithm pixel_fpo \
  --env libero \
  --libero_suite libero_90 \
  --libero_task_id 56 \
  --prefix fpo_libero_full \
  --wandb_project ${proj_name} \
  --seed 42 \
  --max_steps 500000 \
  --batch_size 210 \
  --multi_grad_step 40 \
  --start_online_updates 1000 \
  --buffer_window 4000 \
  --resize_image 128 \
  --query_freq 32 \
  --max_timesteps 400 \
  --eval_interval 10000 \
  --eval_episodes 10 \
  --log_interval 500 \
  --num_fpo_samples 4 \
  --num_euler_steps 6 \
  --clip_epsilon 0.2 \
  --use_ppo_clip 1 \
  --cfm_diff_temperature 1.0 \
  --advantage_type gae \
  --gae_lambda 0.95 \
  --discount 0.99 \
  --num_qs 10 \
  --critic_reduction min \
  --hidden_dims 256 \
  --latent_dim 256 \
  --encoder_type resnet_34_v1 \
  --actor_lr 1e-5 \
  --critic_lr 1e-4 \
  "$@"
