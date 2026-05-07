# PI05 Online PPO Workflow

This is the current recommended workflow for the `ee_delta` PI05 online pipeline.

The practical order is:

1. Collect / merge dataset
2. Train value head and keyframe head
3. Train exploration network
4. Run online PPO

This document uses the current `ee_delta` setup as the default assumption.

## 0. Runtime Notes

Recommended cache setup:

```bash
export UV_CACHE_DIR=/mnt/data1/tmp/pi05_uv_cache
export OPENPI_HF_CACHE_DIR=/mnt/data1/tmp/pi05_hf_cache
```

Optional CPU-only smoke test:

```bash
export JAX_PLATFORMS=cpu
```

For real value service / PPO training, use a working GPU environment.

## 1. Collect And Merge Dataset

### 1.1 Start env websocket server

This server is the source of truth for `ee_delta` env execution and online LeRobot export.

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

Notes:

- `--control_mode ee_delta` is the important one.
- One completed collection run produces one round dataset under `round_XXXXXX/`.

### 1.2 Collect one round without PPO update

This is the data-collection-only mode.

```bash
uv run python scripts/train_pi0_fast_online_v2.py \
  --policy.path checkpoints/pi0_fast_aloha_robotwin_full_ee_delta/beat_block_hammer_pi0fast_full_ee_delta/30000/ \
  --policy.config pi0_fast_aloha_robotwin_ppo_ee_delta \
  --env.ws_url ws://127.0.0.1:8765 \
  --total_updates 1 \
  --rollout_batch_size 300 \
  --mini_batch_size 32 \
  --explore_mode never \
  --ppo_epochs 0
```

```bash
uv run python scripts/train_pi0_fast_online_v2.py \
  --policy.path checkpoints/pi0_fast_aloha_robotwin_full_ee_delta/beat_block_hammer_pi0fast_full_ee_delta/30000/ \
  --policy.config pi0_fast_aloha_robotwin_ppo_ee_delta \
  --env.ws_url ws://127.0.0.1:8765 \
  --total_updates 1 \
  --rollout_batch_size 300 \
  --mini_batch_size 32 \
  --explore_mode never \
  --ppo_epochs 0
```

Output example:

```text
/mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_online_rounds/round_000000
```

### 1.3 Merge offline base dataset and online rounds

Use the merged corpus as the common input for annotation, value/keyframe training, and exploration fitting.

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

Current behavior:

- the merge script now canonicalizes `ee_delta` feature names
- merged corpus metadata will use EE names such as `left_x`, `left_rotvec_x`, `left_dx`, `left_drotvec_x`
- merge now also writes `meta/value_train_rows.jsonl` (row-level value-training metadata cache)

## 2. Train Value Head And Keyframe Head

### 2.1 Annotate keyphases

`keyphases.json` is now the single source of truth for both heads:

- keyframe head:
  predicts phase class, where `0 = none` and `1..K = annotated phase names`
- value head:
  trains a phase-conditioned phase-success target

Use:

```bash
scripts/annotate_keyphases_lerobot.py
```

Input format is one line per episode:

```text
phase names (comma-separated, e.g. pick, put)> pick, put
[episode 12] num_frames=116 keyphases> 1:20-30,2:60-70
```

Output format:

```json
{
  "dataset_root": "...",
  "repo_id": "...",
  "episodes": {
    "12": [
      {"name": "pick_object", "start": 20, "end": 30},
      {"name": "place_object", "start": 60, "end": 70}
    ]
  }
}
```

Recommended command:

```bash
uv run python scripts/annotate_keyphases_lerobot.py \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --out /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/keyphases.json \
  --extract_images
```

Important semantics:

- the first prompt defines the phase-name vocabulary for this dataset
- later episodes use only index syntax like `1:20-30,2:60-70`
- annotation resumes from the same `--out` file if you stop and restart
- frames after the last annotated phase in an episode are excluded from keyframe training

### 2.2 Start value websocket service

This service hosts:

- value head training / inference
- keyframe head training / inference

You can now start it in two modes:

- `pi05_base` mode:
  use the `pi05` VLM prefix features as the shared backbone for value/keyframe heads.
  This is useful when you do not want to load a finetuned `pi0_fast` action policy checkpoint.
- `pi0_fast` mode:
  use the finetuned `pi0_fast` checkpoint as the shared backbone.

Recommended `pi05_base` `ee_delta` example:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/serve_value_mc_ws.py \
  --policy.config pi05_aloha_full_base_ee \
  --host 127.0.0.1 \
  --port 8877 \
  --mc_epochs 5 \
  --mc_batch_size 32 \
  --num_workers 8 \
  --persistent_workers true \
  --keyframe_epochs 5 \
  --keyframe_batch_size 64 \
  --keyframe_chunk_size 32 \
  --value_target_mode evorl_normalized \
  --value_c_fail_coef 1.0 \
  --state_ckpt_dir /mnt/data1/tmp/pi05_value_state
```

<!-- Equivalent `pi0_fast` `ee_delta` example:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/serve_value_mc_ws.py \
  --policy.path checkpoints/pi0_fast_aloha_robotwin_full_ee_delta/beat_block_hammer_pi0fast_full_ee_delta/30000/ \
  --policy.config pi0_fast_aloha_robotwin_ppo_ee_delta \
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
``` -->

Notes:

- `pi05_aloha_full_base_ee` does not require `--policy.path`; it initializes from the config's built-in `pi05_base` weight loader.
- `pi0_fast_aloha_robotwin_ppo_ee_delta` still requires `--policy.path`.
- value training now prefers `meta/value_train_rows.jsonl` if present (generated by merge), reducing startup metadata scan.
- The saved `state_ckpt_dir/latest.pkl` is backbone-specific.
  Do not mix a `pi05` value/keyframe state file with a `pi0_fast` service, or vice versa.

### 2.3 Train keyframe head and value head together

```bash
uv run python scripts/trigger_value_and_keyframe_train_ws.py \
  --ws_url ws://127.0.0.1:8877 \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/keyphases.json
```

This single request runs:

- keyframe / phase-class training first
- phase-conditioned value training second

Current keyframe-head semantics:

- class `0`: not inside any annotated keyphase
- class `1..K`: inside the corresponding annotated phase
- online `keyframe_prob` is interpreted as `P(class > 0)`

### 2.3B New decoupled path: keyframe for exploration, subtask for value

If you want:

- `keyphases.json` to affect only PPO exploration gating
- a separate `subtask_annotations.json` to define per-frame subtasks for value
- value target to mean within-subtask progress rather than whole-phase success

use this path instead.

`keyphases.json`:

- only trains the keyframe / phase classifier
- only affects whether PPO exploration is enabled for a chunk

`subtask_annotations.json`:

- must cover every frame you want value supervision on
- trains both:
  - a `subtask head` that predicts the current subtask id
  - a subtask-conditioned value head that predicts Evo-style progress within that subtask

Recommended trigger sequence:

Keyframe only:

```bash
uv run python scripts/trigger_keyframe_train_ws.py \
  --ws_url ws://127.0.0.1:8877 \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/keyphases.json
```

Subtask-conditioned value only:

```bash
uv run python scripts/trigger_subtask_value_train_ws.py \
  --ws_url ws://127.0.0.1:8877 \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --subtask_annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/subtask_annotations.json
```

Or trigger both in one command with separate annotation files:

```bash
uv run python scripts/trigger_keyframe_and_subtask_value_train_ws.py \
  --ws_url ws://127.0.0.1:8877 \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --keyframe_annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/keyphases.json \
  --subtask_annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/subtask_annotations.json
```

Chunk semantics in this path:

- PPO exploration gate still uses chunk-start `phase_class`
- value uses chunk-start `subtask_class`
- one chunk is treated as belonging to a single subtask

### 2.4 Optional separate triggers

If you want to rerun only one side, the separate triggers still exist.

Keyframe only:

```bash
uv run python scripts/trigger_keyframe_train_ws.py \
  --ws_url ws://127.0.0.1:8877 \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/keyphases.json
```

<!-- Value only, whole-episode target:

```bash
uv run python scripts/trigger_value_mc_train_ws.py \
  --ws_url ws://127.0.0.1:8877 \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus
``` -->

Optional phase-conditioned value training:

```bash
uv run python scripts/trigger_value_mc_train_ws.py \
  --ws_url ws://127.0.0.1:8877 \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/keyphases.json
```

Value-training notes:

- When `--annotations_json` is provided, value training switches from whole-episode targets to annotated phase targets.
- Samples from phases with the same `name` are trained together by conditioning the value input prompt on `current phase: <name>`.
- The value target is phase-success, not phase-progress:
  - success trajectories: every annotated phase gets the success target
  - failure trajectories: only the last annotated phase gets the failure target
  - earlier phases in a failed trajectory still get the success target
- This means similar-looking frames are no longer forced to regress different values just because they appear at different positions inside the same phase.
- This change only affects value-head training. The action-policy prompt path used by `train_pi0_fast_online_v2.py` is unchanged.

### 2.5 Visualize value / keyframe predictions

Use the overlay renderer to export per-episode videos with:

- `--curves both`: top chart is value prediction, bottom chart is keyframe probability
- `--curves value`: only render the value chart
- `--curves keyframe`: only render the keyframe chart
- `--curves subtask`: only render the subtask-class chart
- `--curves value_subtask`: render subtask-conditioned value plus subtask class
- `--curves all`: render value, keyframe probability, and subtask class together

Recommended command after value/keyframe training:

```bash
uv run python scripts/render_value_overlay_lerobot.py \
  --policy.config pi05_aloha_full_base_ee \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --value_state_file /mnt/data1/tmp/pi05_value_state/latest.pkl \
  --curves both \
  --episodes 0,1,2 \
  --out_dir /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/value_overlay
```

Recommended command for the new decoupled path:

```bash
uv run python scripts/render_value_overlay_lerobot.py \
  --policy.config pi05_aloha_full_base_ee \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --value_state_file /mnt/data1/tmp/pi05_value_state/latest.pkl \
  --subtask_annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/subtask_annotations.json \
  --curves value_subtask \
  --episodes 0,1,2 \
  --out_dir /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/subtask_value_overlay
```

```bash
uv run python scripts/render_keyframe_future_value_overlay.py \
  --policy.config pi05_aloha_full_base_ee \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/keyphases.json \
  --value_state_file /mnt/data1/tmp/pi05_value_state_1/latest.pkl \
  --episodes 0,1,2,100,101,105 \
  --out_dir /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/future_value_overlay
```

Notes:

- `--value_state_file` should point to the `latest.pkl` produced by `serve_value_mc_ws.py`.
- If omitted, the script uses the base checkpoint weights only, without your newly trained value/keyframe heads.
- `--policy.config` must match the backbone used when training the value/keyframe state.
- `pi05_aloha_full_base_ee` does not require `--policy.path`.
- Do not mix a `pi05` `latest.pkl` with a `pi0_fast` overlay policy, or vice versa.
- `--curves` supports `both`, `value`, `keyframe`, `subtask`, `value_subtask`, and `all`.
- When `--subtask_annotations_json` is provided, the renderer can overlay per-frame subtask class and uses the matching subtask prompt for value inference.
- `--episodes` accepts a comma-separated episode list. Empty means render all episodes.
- The script also writes a JSON summary file next to the rendered videos.

Small smoke test:

```bash
uv run python scripts/render_value_overlay_lerobot.py \
  --policy.config pi05_aloha_full_base_ee \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --value_state_file /mnt/data1/tmp/pi05_value_state/latest.pkl \
  --curves value \
  --episodes 0 \
  --out_dir /mnt/data1/tmp/pi05_value_overlay_smoke
```

## 3. Train Exploration Network

Current recommended script:

```bash
scripts/fit_exploration_net_nll_lerobot.py
```

Alternative dimwise sigma-network script:

```bash
scripts/fit_dimwise_sigma_net_lerobot.py
```

Recommended idea:

1. Use keyphase annotations to select meaningful chunks
2. Build clean action chunks from dataset
3. Add perturbation only on selected action dims
4. Convert perturbed chunk to DCT target
5. Fit exploration net on direction + radius target

For 14D EE actions, the recommended perturb dims are:

```text
0,1,2,7,8,9
```

That means:

- left translation dims: `0,1,2`
- right translation dims: `7,8,9`

Training command:

<!-- ```bash
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

Small smoke test:

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
``` -->

Dimwise sigma-network example:

```bash
uv run python scripts/fit_dimwise_sigma_net_lerobot.py \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/keyphases.json \
  --out checkpoints/explored_net/dimwise_sigma_dct.pkl \
  --chunk_size 32 \
  --action_noise_dims "0,1,2,3,4,5" \
  --norm_stats_path assets/pi0_fast_aloha_robotwin_full_ee_delta/lerobot-hammer-clean-100-ee_delta/norm_stats.json \
  --noise_dct_keep_k 4 \
  --epochs 100 \
  --batch_size 128
```

Output checkpoint example:

```text
checkpoints/explored_net/dimwise_sigma_dct.pkl
```

Recommended sanity-check before online PPO:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/analyze_dimwise_sigma_net_lerobot.py \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/keyphases.json \
  --net checkpoints/explored_net/dimwise_sigma_dct.pkl \
  --chunk_size 32 \
  --action_noise_dims "0,1,2,3,4,5" \
  --num_samples_per_chunk 64 \
  --out_json checkpoints/explored_net/dimwise_sigma_dct_analysis.json
```

Compare explored action in videos:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/compare_sigma_rollout_videos.py \
  --policy.path checkpoints/pi0_fast_aloha_robotwin_full_ee_delta/beat_block_hammer_pi0fast_full_ee_delta/30000/ \
  --policy.config pi0_fast_aloha_robotwin_ppo_ee_delta \
  --env.ws_url ws://127.0.0.1:8765 \
  --value.ws_url ws://127.0.0.1:8877 \
  --env.seed 100000 \
  --max_chunks 64 \
  --seed 0 \
  --sigma_ckpt checkpoints/explored_net/dimwise_sigma_dct.pkl \
  --explore_mode conditional_keyframe \
  --explore_keyframe_threshold 0.5 \
  --explore_action_noise_dims "0,1,2,3,4,5" \
  --out_json checkpoints/explored_net/compare_sigma_rollout_videos_ws8877.json
```

What to look for in the analysis JSON:

- `pre_clip.histogram`: should cover multiple bins inside `[0, 0.03]`, not collapse only near zero or only near the upper edge.
- `pre_clip.exceed_rate`: small tail exceedance is acceptable; the current `v2` target is neighborhood exploration, not exact hard-constraint satisfaction before clip.
- `pre_clip.near_limit_rate_90` and `pre_clip.near_limit_rate_95`: should stay modest if you want exploration spread across the neighborhood rather than concentrated at the boundary.

## 4. Run Online PPO

Keep the env websocket server running from step 1.

Keep the value websocket service running from step 2.

Then run PPO:

<!-- ```bash
uv run python scripts/train_pi0_fast_online_v2.py \
  --policy.path checkpoints/pi0_fast_aloha_robotwin_full_ee_delta/beat_block_hammer_pi0fast_full_ee_delta/30000/ \
  --policy.config pi0_fast_aloha_robotwin_ppo_ee_delta \
  --env.ws_url ws://127.0.0.1:8765 \
  --rollout_batch_size 64 \
  --mini_batch_size 32 \
  --ppo_epochs 4 \
  --value_epochs 1 \
  --total_updates 100 \
  --value.ws_url ws://127.0.0.1:8877 \
  --explore_mode never \
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
  --explore_network_latent_dim 16
``` -->

Using the dimwise sigma DCT perturbation network:

```bash
uv run python scripts/train_pi0_fast_online_v2.py \
  --policy.path checkpoints/pi0_fast_aloha_robotwin_full_ee_delta/beat_block_hammer_pi0fast_full_ee_delta/30000/ \
  --policy.config pi0_fast_aloha_robotwin_ppo_ee_delta \
  --env.ws_url ws://127.0.0.1:8765 \
  --total_updates 100 \
  --buffer_capacity 64 \
  --mini_batch_size 8 \
  --ppo_epochs 4 \
  --value.ws_url ws://127.0.0.1:8877 \
  --explore_mode conditional_keyframe \
  --explore_keyframe_threshold 0.5 \
  --explore_perturb_backend dct_sigma_network \
  --explore_network_ckpt checkpoints/explored_net/dimwise_sigma_dct.pkl \
  --explore_action_noise_dims "0,1,2,3,4,5" \
  --timing_log_file /tmp/pi05_ppo_timing.jsonl

```

Notes for this sigma-net path:

- `dct_sigma_network` now runs on the standard full-chunk decode path.
- PPO first decodes the full chunk, then perturbs the selected low-frequency DCT block, then recomputes logprobs on the executed tokens.
- Phase and value are both served by `serve_value_mc_ws.py`. For one chunk, rollout fixes the phase to the chunk-start phase when querying the executed chunk's `next_value`.
- With `--explore_mode conditional_keyframe`, sigma perturbation is only enabled when the websocket keyframe probability exceeds `--explore_keyframe_threshold`.
- The checkpoint stores `action_delta_limit`, and the runtime sampler will still rescale a sampled chunk if any action-space perturbation exceeds that limit.
- `--noise_dct_keep_k 4` means sigma is predicted only on the first 4 DCT rows.
- `--explore_action_noise_dims "0,1,2,3,4,5,6"` means sigma is modeled only on those action dimensions.
- Current recommended checkpoint path in this workflow is `dimwise_sigma_dct.pkl`.

Convenience wrapper also exists:

```bash
bash scripts/run_pi0_fast_online_v2.sh \
  checkpoints/pi0_fast_aloha_robotwin_full_ee_delta/beat_block_hammer_pi0fast_full_ee_delta/30000 \
  ws://127.0.0.1:8765 \
  pi0_fast_aloha_robotwin_ppo_ee_delta
```

## Practical Notes

1. If you only want rollout data, set `--ppo_epochs 0`.
2. Value and keyframe training logs appear in the `serve_value_mc_ws.py` terminal.
3. The env server is responsible for correct `ee_delta` execution and online LeRobot export.
4. The PPO policy script now supports `--policy.config auto`, but being explicit is still clearer in important runs.
5. `train_aux_triplet_from_lerobot.sh` is optional convenience tooling, not the main recommended path.

## Regenerate / Retrain Guidance

You usually need to redo only the steps downstream of the thing that changed.

- If you only changed workflow docs: no rerun needed.
- If you only fixed dataset feature names in metadata: re-merge is enough, full re-collection is not required.
- If old online rounds were collected with wrong env mode or wrong PPO config: recollect those rounds.
- If PPO training used bad online rounds: rebuild merged corpus and retrain from the corrected corpus.
- If you only changed eval compatibility: no need to regenerate dataset or retrain.
