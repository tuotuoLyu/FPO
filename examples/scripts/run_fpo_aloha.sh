#!/bin/bash
# FPO on ALOHA Transfer Cube (gym-aloha) with the paper's ALOHA setting (Fig. 4).
set -e
proj_name=FPO_pi0_aloha

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-./openpi}
export EXP=${EXP:-./logs/$proj_name}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# CUDA_VISIBLE_DEVICES: --batch_size (192) must be divisible by the number of visible GPUs.

pip install mujoco==2.3.7   # gym-aloha needs MuJoCo 2.3.7

python3 examples/launch_train_fpo.py \
  --algorithm pixel_fpo \
  --env aloha_cube \
  --prefix fpo_pi0_aloha \
  --wandb_project ${proj_name} \
  --seed 0 \
  --max_steps 3000000 \
  --batch_size 192 \
  --multi_grad_step 20 \
  --start_online_updates 1000 \
  --buffer_window 4000 \
  --resize_image 64 \
  --query_freq 50 \
  --max_timesteps 400 \
  --eval_interval 10000 \
  --eval_episodes 10 \
  --log_interval 500 \
  --num_fpo_samples 8 \
  --num_euler_steps 8 \
  --clip_epsilon 0.2 \
  --use_ppo_clip 1 \
  --cfm_diff_temperature 1.0 \
  --advantage_type gae \
  --gae_lambda 0.95 \
  --q_reduction min \
  --discount 0.99 \
  --num_qs 10 \
  --critic_reduction min \
  --hidden_dims 256 256 256 \
  --latent_dim 256 \
  --encoder_type resnet_34_v1 \
  --use_spatial_softmax 1 \
  --actor_lr 3e-5 \
  --critic_lr 3e-4 \
  --action_magnitude 2.0 \
  --target_entropy auto \
  "$@"
