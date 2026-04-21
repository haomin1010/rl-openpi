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
VALUE_EPOCHS=${5:-1}
VALUE_WS_URL=${6:-}
EXPLORE_MODE=${7:-always}
EXPLORE_KEYFRAME_THRESHOLD=${8:-0.5}
EXPLORED_CHUNK_WEIGHT=${9:-1.0}
NON_EXPLORED_CHUNK_WEIGHT=${10:-1.0}
EXPLORE_PERTURB_BACKEND=${11:-dct_gaussian}
EXPLORE_ACTION_RADIUS_MIN=${12:-0.0}
EXPLORE_ACTION_RADIUS_MAX=${13:-0.15}
EXPLORE_ACTION_ABS_CLIP=${14:-1.0}
EXPLORE_NETWORK_MIX_ALPHA=${15:-0.5}
EXPLORE_NETWORK_CKPT=${16:-}
EXPLORE_NETWORK_OBS_DIM=${17:-32}
EXPLORE_NETWORK_LATENT_DIM=${18:-16}
EXPLORE_KEYFRAME_GATE=${19:-model_head}
EXPLORE_KEYFRAME_NET_CKPT=${20:-}
EXPLORE_ACTION_NOISE_DIMS=${21:-}

CMD=(
  uv run scripts/train_pi0_fast_online_v2.py
  --policy.path "${CKPT_PATH}"
  --policy.config "${POLICY_CONFIG}"
  --env.ws_url "${ENV_WS_URL}"
  --repo_root "${REPO_ROOT}"
  --total_updates 100
  --rollout_batch_size 64
  --mini_batch_size 32
  --ppo_epochs 4
  --value_epochs "${VALUE_EPOCHS}"
  --explore_dct_dims 3
  --explore_mode "${EXPLORE_MODE}"
  --explore_keyframe_threshold "${EXPLORE_KEYFRAME_THRESHOLD}"
  --explore_keyframe_gate "${EXPLORE_KEYFRAME_GATE}"
  --explore_perturb_backend "${EXPLORE_PERTURB_BACKEND}"
  --explore_action_radius_min "${EXPLORE_ACTION_RADIUS_MIN}"
  --explore_action_radius_max "${EXPLORE_ACTION_RADIUS_MAX}"
  --explore_action_abs_clip "${EXPLORE_ACTION_ABS_CLIP}"
  --explore_network_mix_alpha "${EXPLORE_NETWORK_MIX_ALPHA}"
  --explore_network_obs_dim "${EXPLORE_NETWORK_OBS_DIM}"
  --explore_network_latent_dim "${EXPLORE_NETWORK_LATENT_DIM}"
  --explored_chunk_weight "${EXPLORED_CHUNK_WEIGHT}"
  --non_explored_chunk_weight "${NON_EXPLORED_CHUNK_WEIGHT}"
)

if [[ -n "${EXPLORE_ACTION_NOISE_DIMS}" ]]; then
  CMD+=(--explore_action_noise_dims "${EXPLORE_ACTION_NOISE_DIMS}")
fi

if [[ -n "${EXPLORE_NETWORK_CKPT}" ]]; then
  CMD+=(--explore_network_ckpt "${EXPLORE_NETWORK_CKPT}")
fi

if [[ -n "${EXPLORE_KEYFRAME_NET_CKPT}" ]]; then
  CMD+=(--explore_keyframe_net_ckpt "${EXPLORE_KEYFRAME_NET_CKPT}")
fi

if [[ -n "${VALUE_WS_URL}" ]]; then
  CMD+=(--value.ws_url "${VALUE_WS_URL}")
fi

UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache} "${CMD[@]}"
