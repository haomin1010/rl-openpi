# PI05 Online PPO Workflow

This document describes the full workflow:
1. Rollout and export offline dataset.
2. Train Value/Keyframe (shared value service).
3. Train exploration perturbation net.
4. Run formal online PPO training.

## 0. Runtime Notes

In restricted/sandboxed environments, `uv` and HuggingFace datasets caches may default to read-only locations.
For command-line runs, it is recommended to prepend:

```bash
export UV_CACHE_DIR=/mnt/data1/tmp/pi05_uv_cache
export OPENPI_HF_CACHE_DIR=/mnt/data1/tmp/pi05_hf_cache
```

In practice, `/mnt/data1/tmp` is the preferred location for PI05 temporary files, caches, and smoke-test outputs.
It avoids common `/tmp` space issues during value/keyframe state saving.

The LeRobot-reading scripts in this repo automatically fall back to a writable local HF cache when the default cache
location is not writable, but setting `OPENPI_HF_CACHE_DIR=/mnt/data1/tmp/pi05_hf_cache` explicitly is still
recommended for consistency.

For quick smoke tests without GPU, you can also prepend:

```bash
export JAX_PLATFORMS=cpu
```

This is only for functionality validation. Practical value-service startup and online PPO training are expected to
run on a machine with a working GPU/JAX CUDA stack.

## 1. Rollout And Export Dataset

Policy-side `train_pi0_fast_online_v2.py` now sends an env websocket `finalize_collection_round` command when one
full run finishes successfully. If `--save_lerobot` is enabled, the env server writes one LeRobot dataset per round
under `--lerobot_root/round_XXXXXX/` and then rotates to the next round on the next run.

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
  --lerobot_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_online_rounds \
  --lerobot_repo_id lerobot_online_rounds \
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

Round dataset output is under:
`/mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_online_rounds/round_000000`

Before auxiliary training, merge offline data plus all collected round datasets into one cumulative corpus:

```bash
uv run python scripts/merge_lerobot_round_datasets.py \
  --src_dataset_roots \
    /path/to/lerobot_offline_base \
    /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_online_rounds/round_000000 \
  --src_repo_ids \
    lerobot_offline_base \
    lerobot_online_rounds_round_000000 \
  --out_dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --out_repo_id lerobot_ppo_corpus \
  --overwrite
```

After this merge step, use `lerobot_ppo_corpus` as the dataset root for annotation, value/keyframe training, and
exploration training.

## 2. Keyframe Annotation And Value/Keyframe Training

Annotate keyframes:

```bash
uv run python scripts/annotate_keyframes_lerobot.py \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --out /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/keyframes.json \
  --extract_images
```

Minimal smoke test: annotate one episode then quit immediately.

```bash
uv run python scripts/annotate_keyframes_lerobot.py \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --out /mnt/data1/tmp/pi05_keyframes_smoke.json \
  --start_episode 0 \
  --end_episode 0
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
  --value_length_scale_quantile 0.95 \
  --state_ckpt_dir /mnt/data1/tmp/pi05_value_state
```

Trigger keyframe training:

```bash
uv run python scripts/trigger_keyframe_train_ws.py \
  --ws_url ws://127.0.0.1:8877 \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/keyframes.json
```

Trigger value training from the same dataset:

```bash
uv run python scripts/trigger_value_mc_train_ws.py \
  --ws_url ws://127.0.0.1:8877 \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus
```

## 3. Train Exploration DCT Perturbation Net

Train JAX DCT perturbation net (no sigma head, NLL objective).

Important: the recommended chain is now:

1. build clean action chunks from dataset
2. add perturbation only on selected action dims
3. re-encode perturbed action chunk back to DCT
4. fit the exploration net to the resulting DCT perturbation target

For the EE 14D action definition, the recommended default perturb dims are:

- left arm translation: `0,1,2`
- right arm translation: `7,8,9`

So the practical default for EE mode is:

```text
--action_noise_dims "0,1,2,7,8,9"
```

Train command:

```bash
uv run python scripts/fit_exploration_net_nll_lerobot.py \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/keyframes.json \
  --out checkpoints/explored_net/explore_dir_radius_nll.pkl \
  --chunk_size 32 \
  --dct_k 4 \
  --latent_dim 16 \
  --action_noise_dims "0,1,2,7,8,9" \
  --epochs 30 \
  --batch_size 64
```

Minimal smoke test:

```bash
uv run python scripts/fit_exploration_net_nll_lerobot.py \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/keyframes.json \
  --out /mnt/data1/tmp/pi05_explore/explore_dir_radius_nll.pkl \
  --chunk_size 8 \
  --dct_k 2 \
  --latent_dim 4 \
  --hidden_dim 32 \
  --hidden_depth 1 \
  --action_noise_dims "0,1,2,7,8,9" \
  --epochs 1 \
  --batch_size 2 \
  --max_chunks 4
```

The generated checkpoint is a JAX-network parameter checkpoint:
`checkpoints/explored_net/explore_dir_radius_nll.pkl`

## 4. Formal Online PPO Training

Run env server (same as step 1, usually keep it running).

Run online PPO with value ws and exploration net (explicit named args, easy to edit).

Recommended path:

- exploration net is trained from masked action perturbation targets
- online inference still works in DCT space
- use `dct_network` instead of the older action-space backend chain

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
  --explore_perturb_backend dct_network \
  --explore_action_radius_min 0.0 \
  --explore_action_radius_max 0.15 \
  --explore_action_abs_clip 1.0 \
  --explore_network_mix_alpha 0.5 \
  --explore_network_ckpt checkpoints/explored_net/explore_dir_radius_nll.pkl \
  --explore_network_obs_dim 32 \
  --explore_network_latent_dim 16 \
  --explore_keyframe_gate model_head
```

## Notes

1. If you only want rollout data, set `--ppo_epochs 0 --value_epochs 0`.
2. Value and keyframe training logs are printed on the `serve_value_mc_ws.py` terminal.
3. `serve_value_mc_ws.py` must be restarted after code changes.
4. Current `train_aux_triplet_from_lerobot.sh` is optional convenience tooling. For the latest exploration net (`fit_exploration_net_nll_lerobot.py`), use manual commands above.
5. In the current recommended design, `--explore_action_noise_dims` is mainly a training-target construction concept for the DCT exploration net, not the primary online backend control path.
