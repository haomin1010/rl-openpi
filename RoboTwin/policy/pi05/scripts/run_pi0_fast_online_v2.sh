#!/bin/bash
set -euo pipefail

# Example:
# bash scripts/run_pi0_fast_online_v2.sh \
#   checkpoints/pi0_fast_aloha_robotwin_ppo/beat_block_hammer_pi0fast_full/30000 \
#   ws://127.0.0.1:8765

CKPT_PATH=${1}
ENV_WS_URL=${2:-ws://127.0.0.1:8765}
POLICY_CONFIG=${3:-pi0_fast_aloha_robotwin_ppo}
REPO_ROOT=${4:-}

UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache} uv run scripts/train_pi0_fast_online_v2.py \
  --policy.path "${CKPT_PATH}" \
  --policy.config "${POLICY_CONFIG}" \
  --env.ws_url "${ENV_WS_URL}" \
  --repo_root "${REPO_ROOT}" \
  --fixed_reward 0.0 \
  --total_updates 100 \
  --rollout_batch_size 64 \
  --mini_batch_size 32 \
  --ppo_epochs 4 \
  --explore_dct_dims 4
