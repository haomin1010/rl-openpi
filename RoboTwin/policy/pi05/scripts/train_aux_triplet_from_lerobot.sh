#!/bin/bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

# One-shot manual pipeline:
# 1) train keyframe head on value ws (shared backbone) from human annotations
# 2) fit exploration net using keyframe-focused samples (annotations)
# 3) trigger external value MC training from the same dataset
#
# Example:
# bash scripts/train_aux_triplet_from_lerobot.sh \
#   --dataset_root /path/to/lerobot_dataset_root \
#   --value_ws_url ws://127.0.0.1:8877 \
#   --annotations_json /path/to/keyframes.json \
#   --explore_out /tmp/openpi_value_sync/explore_net_latest.npz

DATASET_ROOT=""
REPO_ID="lerobot-hammer-online"
VALUE_WS_URL=""
ANNOTATIONS_JSON=""
EXPLORE_OUT=""

OBS_KEY="observation.state"
ACTION_KEY="action"
OBS_DIM=32
LATENT_DIM=16
RADIUS_MIN=0.0
RADIUS_MAX=0.15
MAX_SAMPLES=10000
KEYFRAME_CHUNK_SIZE=32

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_root) DATASET_ROOT="$2"; shift 2 ;;
    --repo_id) REPO_ID="$2"; shift 2 ;;
    --value_ws_url) VALUE_WS_URL="$2"; shift 2 ;;
    --annotations_json) ANNOTATIONS_JSON="$2"; shift 2 ;;
    --explore_out) EXPLORE_OUT="$2"; shift 2 ;;
    --obs_key) OBS_KEY="$2"; shift 2 ;;
    --action_key) ACTION_KEY="$2"; shift 2 ;;
    --obs_dim) OBS_DIM="$2"; shift 2 ;;
    --latent_dim) LATENT_DIM="$2"; shift 2 ;;
    --radius_min) RADIUS_MIN="$2"; shift 2 ;;
    --radius_max) RADIUS_MAX="$2"; shift 2 ;;
    --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
    --keyframe_chunk_size) KEYFRAME_CHUNK_SIZE="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${VALUE_WS_URL}" ]]; then
  echo "[aux] --value_ws_url is empty: keyframe/value training stages will be skipped."
fi
if [[ -z "${DATASET_ROOT}" || ! -d "${DATASET_ROOT}" ]]; then
  echo "--dataset_root is required and must exist." >&2
  exit 3
fi

echo "[aux] dataset_root=${DATASET_ROOT}"
echo "[aux] repo_id=${REPO_ID}"

if [[ -n "${ANNOTATIONS_JSON}" && -n "${VALUE_WS_URL}" ]]; then
  echo "[aux] stage=keyframe_train (ws)"
  uv run python scripts/trigger_keyframe_train_ws.py \
    --ws_url "${VALUE_WS_URL}" \
    --dataset_root "${DATASET_ROOT}" \
    --repo_id "${REPO_ID}" \
    --annotations_json "${ANNOTATIONS_JSON}"
else
  echo "[aux] stage=keyframe_train skipped (need both --annotations_json and --value_ws_url)"
fi

if [[ -n "${EXPLORE_OUT}" ]]; then
  echo "[aux] stage=exploration_fit"
  CMD=(
    uv run python scripts/fit_exploration_net_lerobot.py
    --dataset_root "${DATASET_ROOT}"
    --repo_id "${REPO_ID}"
    --out "${EXPLORE_OUT}"
    --obs_key "${OBS_KEY}"
    --action_key "${ACTION_KEY}"
    --obs_dim "${OBS_DIM}"
    --latent_dim "${LATENT_DIM}"
    --max_samples "${MAX_SAMPLES}"
    --radius_min "${RADIUS_MIN}"
    --radius_max "${RADIUS_MAX}"
    --keyframe_chunk_size "${KEYFRAME_CHUNK_SIZE}"
  )
  if [[ -n "${ANNOTATIONS_JSON}" ]]; then
    CMD+=(--keyframe_source annotations --keyframe_annotations_json "${ANNOTATIONS_JSON}")
  else
    CMD+=(--keyframe_source all)
  fi
  "${CMD[@]}"
else
  echo "[aux] stage=exploration_fit skipped (--explore_out is empty)"
fi

if [[ -n "${VALUE_WS_URL}" ]]; then
  echo "[aux] stage=value_mc_train (ws)"
  uv run python scripts/trigger_value_mc_train_ws.py \
    --ws_url "${VALUE_WS_URL}" \
    --dataset_root "${DATASET_ROOT}" \
    --repo_id "${REPO_ID}"
else
  echo "[aux] stage=value_mc_train skipped (--value_ws_url is empty)"
fi

echo "[aux] done."
