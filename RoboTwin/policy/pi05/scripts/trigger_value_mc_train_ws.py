#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from openpi_online_ppo.rl.value_ws_client import ValueWebsocketClient


def main() -> None:
    p = argparse.ArgumentParser(description="Trigger external value MC training from a local LeRobot dataset.")
    p.add_argument("--ws_url", type=str, required=True, help="Value service websocket url, e.g. ws://127.0.0.1:8877")
    p.add_argument("--dataset_root", type=str, required=True, help="Root dir that contains meta_data/ and data/ .")
    p.add_argument("--repo_id", type=str, default="local/openpi_mc")
    p.add_argument(
        "--annotations_json",
        type=str,
        default=None,
        help="Optional keyphase annotations json. When provided, value training switches to phase-conditioned mode.",
    )
    args = p.parse_args()

    client = ValueWebsocketClient(args.ws_url)
    try:
        metrics = client.train_mc_from_lerobot(
            dataset_root=args.dataset_root,
            repo_id=args.repo_id,
            annotations_json=args.annotations_json,
        )
    finally:
        client.close()
    print(json.dumps(metrics, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
