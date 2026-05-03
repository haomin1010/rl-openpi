# PI05 EE-Delta SFT + KeyPhase + Online PPO Workflow

This document describes the recommended end-to-end workflow for the `ee_delta`
`pi0_fast` pipeline under the following unified assumptions:

1. FAST action tokens are encoded row-wise.
2. Keyphase annotations are the source of truth for both keyframe and value supervision.
3. Keyframe supervision is phase-aware with 16 distance bins.
4. Value training groups samples by keyphase `name`.
5. Online PPO only perturbs chunks when the current frame is predicted as keyframe.
6. When perturbation is enabled, decoding runs row by row and perturbed row tokens are fed back as prefix for the next row.

The practical order is:

1. Build offline EE dataset
2. Run SFT for `pi0_fast`
3. Collect / merge PPO corpus
4. Annotate keyphases
5. Train 16-Bin Keyframe Head
6. Train phase-conditioned value head
7. Train sigma exploration net
8. Run online PPO

## 0. Important Semantics

### 0.1 Tokenization

For the EE `pi0_fast` configs below, FAST action tokens are row-wise encoded:

- config: `pi0_fast_aloha_robotwin_full_ee_delta`
- config: `pi0_fast_aloha_robotwin_ppo_ee_delta`

Current row layout is:

- `rowwise_layout=action_dim_major`
- action shape is `[32, 14]`
- row-wise encoding treats it as `14` rows, each row containing `32` DCT coefficients

### 0.2 Keyphase supervision

Keyphase annotations use:

```json
{
  "episodes": {
    "12": [
      {"name": "pick_object", "start": 20, "end": 30},
      {"name": "place_object", "start": 60, "end": 70}
    ]
  }
}
```

Semantics:

- keyframe head target:
  `bin 15` -> frame inside keyphase
  `bin 14` -> 1 frame away from nearest phase boundary
  ...
  `bin 1` -> 14 frames away
  `bin 0` -> 15 frames away or farther
- value head target: samples are grouped by keyphase `name`, and value prompt is conditioned on `current phase: <name>`

## 1. Build Offline EE Dataset

### 1.1 Prepare raw RoboTwin data

Example symlink:

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin
ln -sfn /mnt/data/lhm/test/RoboTwin/data/beat_block_hammer data/beat_block_hammer
```

### 1.2 Process raw data into EE-Delta 14D

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
python scripts/process_data.py beat_block_hammer demo_clean 100 --representation ee_delta
```

### 1.3 Convert into LeRobot

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
  --raw-dir processed_data/beat_block_hammer-demo_clean-100-ee_delta \
  --repo-id lerobot-hammer-clean-100-ee_delta \
  --task beat_block_hammer \
  --mode image
```

## 2. Run SFT For `pi0_fast`

### 2.1 Compute norm stats

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
uv run scripts/compute_norm_stats.py --config-name pi0_fast_aloha_robotwin_full_ee_delta
```

### 2.2 Train SFT checkpoint

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
bash finetune.sh pi0_fast_aloha_robotwin_full_ee_delta beat_block_hammer_pi0fast_full_ee_delta 4,5,6,7
```

Output example:

```text
RoboTwin/policy/pi05/checkpoints/pi0_fast_aloha_robotwin_full_ee_delta/beat_block_hammer_pi0fast_full_ee_delta/30000
```

### 2.3 Optional eval

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
bash eval.sh beat_block_hammer demo_clean pi0_fast_aloha_robotwin_full_ee_delta beat_block_hammer_pi0fast_full_ee_delta 0 0
```

## 3. Collect And Merge PPO Corpus

### 3.1 Start env websocket server

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05

uv run python scripts/serve_robotwin_env_ws.py \
  --task_name beat_block_hammer \
  --task_config demo_clean \
  --instruction_type seen \
  --host 127.0.0.1 \
  --port 8765 \
  --control_mode ee_delta \
  --save_video \
  --save_lerobot \
  --lerobot_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_online_rounds \
  --lerobot_repo_id lerobot_online_rounds \
  --lerobot_overwrite
```

### 3.2 Collect one rollout-only round

```bash
uv run python scripts/train_pi0_fast_online_v2.py \
  --policy.path checkpoints/pi0_fast_aloha_robotwin_full_ee_delta/beat_block_hammer_pi0fast_full_ee_delta/30000/ \
  --policy.config pi0_fast_aloha_robotwin_ppo_ee_delta \
  --env.ws_url ws://127.0.0.1:8765 \
  --total_updates 1 \
  --rollout_batch_size 1024 \
  --mini_batch_size 32 \
  --explore_mode never \
  --ppo_epochs 0 \
  --value_epochs 0
```

### 3.3 Merge offline base and online rounds

```bash
uv run python scripts/merge_lerobot_round_datasets.py \
  --src_dataset_roots \
    /mnt/data/Environment/huggingface_cache/lerobot/lerobot-hammer-clean-100-ee_delta \
    /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_online_rounds/round_000000 \
  --src_repo_ids \
    lerobot-hammer-clean-100-ee_delta \
    lerobot_online_rounds_round_000000 \
  --out_dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --out_repo_id lerobot_ppo_corpus \
  --overwrite
```

## 4. Annotate KeyPhases

Use:

```bash
scripts/annotate_keyphases_lerobot.py
```

Example interaction:

```text
phase names (comma-separated, e.g. pick, put)> pick, put
[episode 12] num_frames=116 keyphases> 1:20-30,2:60-70
```

Recommended command:

```bash
uv run python scripts/annotate_keyphases_lerobot.py \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --out /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/keyphases.json \
  --extract_images
```

## 5. Train Binary Keyframe Head

### 5.1 Start value websocket service

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/serve_value_mc_ws.py \
  --policy.config pi05_aloha_full_base_ee \
  --host 127.0.0.1 \
  --port 8877 \
  --mc_epochs 3 \
  --mc_batch_size 64 \
  --num_workers 8 \
  --persistent_workers true \
  --keyframe_epochs 3 \
  --keyframe_batch_size 64 \
  --keyframe_chunk_size 32 \
  --value_target_mode evorl_normalized \
  --value_c_fail_coef 1.0 \
  --state_ckpt_dir /mnt/data1/tmp/pi05_value_state
```

Notes:

- this workflow fixes `serve_value_mc_ws.py` to `pi05` base mode
- `pi05_aloha_full_base_ee` does not require `--policy.path`
- keyframe head now uses `16` bins
- `predict_keyframe_prob` is the expected closeness score in `[0, 1]`

### 5.2 Train keyframe head from keyphases

```bash
uv run python scripts/trigger_keyframe_train_ws.py \
  --ws_url ws://127.0.0.1:8877 \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/keyphases.json
```

## 6. Train Phase-Conditioned Value Head

```bash
uv run python scripts/trigger_value_mc_train_ws.py \
  --ws_url ws://127.0.0.1:8877 \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/keyphases.json
```

Notes:

- providing `--annotations_json` switches value training from whole-episode targets to keyphase-segment targets
- samples from the same phase `name` are trained together
- value input prompt is conditioned on `current phase: <name>`

## 7. Train Sigma Exploration Net

Recommended command:

```bash
uv run python scripts/fit_dimwise_sigma_net_lerobot.py \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/keyphases.json \
  --out checkpoints/explored_net/dimwise_sigma_dct_v2.pkl \
  --chunk_size 32 \
  --action_noise_dims "0,1,2,3,4,5,6" \
  --action_delta_limit 0.1 \
  --epochs 100 \
  --batch_size 128
```

Recommended sanity-check after training:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/analyze_dimwise_sigma_net_lerobot.py \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/keyphases.json \
  --net checkpoints/explored_net/dimwise_sigma_dct_v2.pkl \
  --chunk_size 32 \
  --action_noise_dims "0,1,2,3,4,5,6,7,8,9,10,11,12,13" \
  --num_samples_per_chunk 64 \
  --out_json checkpoints/explored_net/dimwise_sigma_dct_v2_analysis.json
```

Current recommended sigma-net target:

- use the `v2` checkpoint trained by the newer reward-plus-penalty objective
- goal is neighborhood-spread exploration inside the action delta limit, not maximizing raw sigma
- inspect `pre_clip.histogram`, `pre_clip.near_limit_rate_90`, and `pre_clip.near_limit_rate_95` to confirm exploration covers `[0, 0.03]` instead of collapsing near zero or the boundary

## 8. Run Online PPO

Desired semantics:

1. model predicts whether current frame is keyframe
2. if not keyframe: no perturbation
3. if keyframe: decode row by row
4. once one row is complete, apply sigma-net perturbation in DCT space
5. re-encode perturbed row tokens
6. feed perturbed row tokens back as prefix for next row generation

Recommended command:

```bash
uv run python scripts/train_pi0_fast_online_v2.py \
  --policy.path checkpoints/pi0_fast_aloha_robotwin_full_ee_delta/beat_block_hammer_pi0fast_full_ee_delta/30000/ \
  --policy.config pi0_fast_aloha_robotwin_ppo_ee_delta \
  --env.ws_url ws://127.0.0.1:8765 \
  --total_updates 10 \
  --rollout_batch_size 64 \
  --mini_batch_size 32 \
  --ppo_epochs 1 \
  --value_epochs 1 \
  --value.ws_url ws://127.0.0.1:8877 \
  --explore_mode conditional_keyframe \
  --explore_keyframe_threshold 0.5 \
  --explore_keyframe_gate model_head \
  --explore_perturb_backend dct_sigma_network \
  --explore_network_ckpt checkpoints/explored_net/dimwise_sigma_dct_v2.pkl \
  --explore_action_noise_dims "0,1,2,3,4,5,6,7,8,9,10,11,12,13"
```

## 9. Summary Of What Must Match

- SFT config:
  `pi0_fast_aloha_robotwin_full_ee_delta`
- PPO config:
  `pi0_fast_aloha_robotwin_ppo_ee_delta`
- both use row-wise FAST tokenization
- keyframe training and value training must use the same `keyphases.json`
- this workflow fixes value/keyframe service to `pi05_aloha_full_base_ee`
- online PPO still uses the `pi0_fast` SFT/PPO backbone
- sigma exploration net must be trained for the same chunk size and action dim used online

## 10. Current Limitations

- row boundaries are recovered implicitly during online row-wise rollout; no explicit separator token is used
- row-wise perturb-and-reencode rollout is slower than one-shot decoding
- previously saved keyframe states using old multi-bin distance semantics are not compatible with the new binary-keyframe setup
