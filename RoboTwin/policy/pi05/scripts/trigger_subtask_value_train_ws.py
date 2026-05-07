#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from openpi_online_ppo.rl.value_ws_client import ValueWebsocketClient


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trigger subtask-conditioned value training on value websocket service.")
    p.add_argument("--ws_url", type=str, default="ws://127.0.0.1:8877")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--repo_id", type=str, required=True)
    p.add_argument("--subtask_annotations_json", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    client = ValueWebsocketClient(args.ws_url)
    try:
        metrics = client.train_subtask_value_from_lerobot(
            dataset_root=args.dataset_root,
            repo_id=args.repo_id,
            subtask_annotations_json=args.subtask_annotations_json,
        )
        print(json.dumps({"stage": "subtask_value_train", "metrics": metrics}, ensure_ascii=False, sort_keys=True))
    finally:
        client.close()


if __name__ == "__main__":
    main()
