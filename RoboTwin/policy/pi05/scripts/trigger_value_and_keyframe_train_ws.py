#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from openpi_online_ppo.rl.value_ws_client import ValueWebsocketClient


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trigger keyframe training and phase-conditioned value training together on value websocket service."
    )
    p.add_argument("--ws_url", type=str, default="ws://127.0.0.1:8877")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--repo_id", type=str, required=True)
    p.add_argument("--annotations_json", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    client = ValueWebsocketClient(args.ws_url)
    try:
        metrics = client.train_value_and_keyframe_from_lerobot(
            dataset_root=args.dataset_root,
            repo_id=args.repo_id,
            annotations_json=args.annotations_json,
        )
        print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    finally:
        client.close()


if __name__ == "__main__":
    main()
