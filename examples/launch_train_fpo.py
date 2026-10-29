"""Launch FPO (or the latent-SAC ablation) on LIBERO / ALOHA with a frozen pi0.

Defaults follow the paper's LIBERO setting; ``examples/scripts/run_fpo_aloha.sh``
holds the ALOHA setting and ``examples/scripts/run_fpo_ablations.sh`` the Table II
ablations. Every learner hyper-parameter below is also a command-line flag.
"""
import argparse
import inspect
import sys

from jaxrl2.utils.launch_util import parse_training_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ---- run ----
    parser.add_argument('--algorithm', default='pixel_fpo', choices=['pixel_fpo', 'pixel_sac'],
                        help='pixel_fpo = FPO; pixel_sac = "without CFM ratio proxy" ablation (SAC in latent space)')
    parser.add_argument('--env', default='libero', choices=['libero', 'aloha_cube'])
    parser.add_argument('--libero_suite', default='libero_90',
                        help='libero_spatial | libero_object | libero_goal | libero_10 (LIBERO-Long) | libero_90')
    parser.add_argument('--libero_task_id', default=56, type=int,
                        help='task index inside the suite; libero_90 #56 = "pick up the butter and put it in the tray"')
    parser.add_argument('--max_timesteps', default=400, type=int, help='env steps per episode')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--max_steps', default=500000, type=int, help='gradient steps')
    parser.add_argument('--batch_size', default=210, type=int, help='must be divisible by the number of GPUs')
    parser.add_argument('--multi_grad_step', default=40, type=int, help='gradient steps per stored transition (UTD)')
    parser.add_argument('--start_online_updates', default=1000, type=int,
                        help='transitions collected with N(0, I) latents before updates start')
    parser.add_argument('--buffer_window', default=4000, type=int,
                        help='sliding window: sample only the most recent transitions (0 = keep all)')
    parser.add_argument('--query_freq', default=32, type=int, help='env steps executed per pi0 action chunk')
    parser.add_argument('--resize_image', default=128, type=int, help='actor/critic image size')
    parser.add_argument('--add_states', default=1, type=int, help='add proprioception to the critic input')
    parser.add_argument('--clip_base_actions', default=1, type=int, help='clip pi0 actions to [-1, 1]')
    # ---- logging / evaluation ----
    parser.add_argument('--eval_interval', default=10000, type=int)
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--eval_at_start', default=1, type=int, help='evaluate unmodified pi0 at step 0')
    parser.add_argument('--log_interval', default=500, type=int)
    parser.add_argument('--checkpoint_interval', default=-1, type=int)
    parser.add_argument('--save_latents', default=0, type=int,
                        help='save evaluation / exploration latents under <outputdir>/latents (Fig. 5 and 7)')
    parser.add_argument('--use_wandb', default=1, type=int)
    parser.add_argument('--wandb_project', default='FPO_pi0_Libero')
    parser.add_argument('--prefix', default='fpo_pi0_libero')
    parser.add_argument('--suffix', default='')
    parser.add_argument('--launch_group_id', default='')

    # learner keyword arguments (each one becomes a flag via parse_training_args)
    train_args_dict = dict(
        # optimisation
        actor_lr=1e-5,
        critic_lr=1e-4,
        temp_lr=3e-4,                 # SAC ablation only
        # networks (separate actor / critic image encoders)
        hidden_dims=(256,),           # critic MLP
        flow_hidden_dims=(512, 512, 512),
        cnn_features=(32, 32, 32, 32),
        cnn_strides=(2, 1, 1, 1),
        cnn_padding='VALID',
        latent_dim=256,
        encoder_type='resnet_34_v1',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=1.0,
        use_bottleneck=True,
        aug_next=1,
        dropout_rate=0.0,             # SAC ablation only
        num_cameras=1,
        # critic ensemble (Eqs. 5-6)
        discount=0.99,
        tau=0.005,
        num_qs=10,                    # 1 = "single critic" ablation
        critic_reduction='min',
        # FPO (Eqs. 2-4, 7)
        num_fpo_samples=4,            # Monte-Carlo (eps, t) draws per sample
        num_euler_steps=6,            # K; 1 = "single-step integration" ablation
        clip_epsilon=0.2,
        use_ppo_clip=True,            # 0 = "without PPO clipping" ablation
        cfm_diff_temperature=1.0,     # beta in Eq. 3
        cfm_clip=3.0,
        log_ratio_clip=5.0,
        ratio_std_floor=1e-3,         # soft floor on sigma_Delta (numerical noise of the cached loss)
        cfm_loss_type='huber',
        advantage_type='gae',         # gae | q
        gae_lambda=0.95,
        q_reduction='min',
        latent_clip=2.0,
        max_grad_norm=0.5,
        # SAC ablation only
        target_entropy='auto',
        action_magnitude=1.0,
    )

    variant, args = parse_training_args(train_args_dict, parser)

    if variant.algorithm == 'pixel_sac':
        from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
        accepted = set(inspect.signature(PixelSACLearner.__init__).parameters)
        variant['train_kwargs'] = {k: v for k, v in variant['train_kwargs'].items() if k in accepted}

    from examples.train_fpo import main
    print(variant)
    main(variant)
    sys.exit()
