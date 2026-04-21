# PI05 Online PPO Workflow

This document describes the full workflow:
1. Rollout and export offline dataset.
2. Train Value/Keyframe (shared value service).
3. Train exploration perturbation net.
4. Run formal online PPO training.

## 1. Rollout And Export Dataset

Run env websocket server with LeRobot export enabled:

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05

uv run python scripts/serve_robotwin_env_ws.py \
  --task_name beat_block_hammer \
  --task_config demo_clean \
  --instruction_type seen \
  --port 8765 \
  --save_video \
  --save_lerobot \
  --lerobot_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_online_run1 \
  --lerobot_repo_id lerobot_online_run1 \
  --lerobot_overwrite
```

Collect rollout from policy side without PPO/value update (data collection only):

```bash
uv run python scripts/train_pi0_fast_online_v2.py \
  --policy.path checkpoints/pi0_fast_aloha_robotwin_full/beat_block_hammer_pi0fast_full/10000/ \
  --policy.config pi0_fast_aloha_robotwin_ppo \
  --env.ws_url ws://127.0.0.1:8765 \
  --total_updates 1 \
  --rollout_batch_size 12 \
  --mini_batch_size 32 \
  --ppo_epochs 0 \
  --value_epochs 0
```

Dataset output is under:
`/mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_online_run1`

## 2. Keyframe Annotation And Value/Keyframe Training

Annotate keyframes:

```bash
uv run python scripts/annotate_keyframes_lerobot.py \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_online_run1 \
  --repo_id lerobot_online_run1 \
  --out /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_online_run1/keyframes.json \
  --extract_images
```

Start value websocket service (shared backbone for value + keyframe head):

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/serve_value_mc_ws.py \
  --policy.path checkpoints/pi0_fast_aloha_robotwin_full/beat_block_hammer_pi0fast_full/10000/ \
  --policy.config pi0_fast_aloha_robotwin_ppo \
  --host 127.0.0.1 \
  --port 8877 \
  --mc_epochs 1 \
  --mc_batch_size 32 \
  --keyframe_epochs 3 \
  --keyframe_batch_size 64 \
  --keyframe_chunk_size 32 \
  --value_target_mode evorl_normalized \
  --value_c_fail_coef 0.995 \
  --value_length_scale_quantile 0.95
```

Trigger keyframe training:

```bash
uv run python scripts/trigger_keyframe_train_ws.py \
  --ws_url ws://127.0.0.1:8877 \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_online_run1 \
  --repo_id lerobot_online_run1 \
  --annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_online_run1/keyframes.json
```

Trigger value training from the same dataset:

```bash
uv run python scripts/trigger_value_mc_train_ws.py \
  --ws_url ws://127.0.0.1:8877 \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_online_run1 \
  --repo_id lerobot_online_run1
```

## 3. Train Exploration Perturbation Net

Train JAX direction+radius perturbation net (no sigma head, NLL objective):

```bash
uv run python scripts/fit_exploration_net_nll_lerobot.py \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_online_run1 \
  --repo_id lerobot_online_run1 \
  --annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_online_run1/keyframes.json \
  --out checkpoints/explored_net/explore_dir_radius_nll.pkl \
  --chunk_size 32 \
  --dct_k 4 \
  --latent_dim 16 \
  --epochs 30 \
  --batch_size 64
```

The generated checkpoint is a JAX-network parameter checkpoint:
`checkpoints/explored_net/explore_dir_radius_nll.pkl`

## 4. Formal Online PPO Training

Run env server (same as step 1, usually keep it running).

Run online PPO with value ws and exploration net (explicit named args, easy to edit):

```bash
uv run python scripts/train_pi0_fast_online_v2.py \
  --policy.path checkpoints/pi0_fast_aloha_robotwin_full/beat_block_hammer_pi0fast_full/10000/ \
  --policy.config pi0_fast_aloha_robotwin_ppo \
  --env.ws_url ws://127.0.0.1:8765 \
  --rollout_batch_size 64 \
  --mini_batch_size 32 \
  --ppo_epochs 4 \
  --value_epochs 1 \
  --total_updates 100 \
  --value.ws_url ws://127.0.0.1:8877 \
  --explore_mode always \
  --explore_keyframe_threshold 0.5 \
  --explored_chunk_weight 1.0 \
  --non_explored_chunk_weight 1.0 \
  --explore_perturb_backend action_network \
  --explore_action_radius_min 0.0 \
  --explore_action_radius_max 0.15 \
  --explore_action_abs_clip 1.0 \
  --explore_network_mix_alpha 0.5 \
  --explore_network_ckpt checkpoints/explored_net/explore_dir_radius_nll.pkl \
  --explore_network_obs_dim 32 \
  --explore_network_latent_dim 16 \
  --explore_keyframe_gate model_head \
  --explore_action_noise_dims "0,1,2,3,4,5,7,8,9,10,11,12"
```

`--explore_action_noise_dims` lets you exclude gripper dimensions from exploration.

## Notes

1. If you only want rollout data, set `--ppo_epochs 0 --value_epochs 0`.
2. Value and keyframe training logs are printed on the `serve_value_mc_ws.py` terminal.
3. `serve_value_mc_ws.py` must be restarted after code changes.
4. Current `train_aux_triplet_from_lerobot.sh` is optional convenience tooling. For the latest exploration net (`fit_exploration_net_nll_lerobot.py`), use manual commands above.
