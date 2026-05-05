from __future__ import annotations

from typing import Any

import websockets.sync.client

from openpi_client import msgpack_numpy


class ValueWebsocketClient:
    """Sync websocket client for remote value service."""

    def __init__(self, ws_url: str):
        self._packer = msgpack_numpy.Packer()
        self._conn = websockets.sync.client.connect(
            ws_url,
            compression=None,
            max_size=None,
        )
        self._try_drain_handshake()

    def _try_drain_handshake(self) -> None:
        try:
            self._conn.recv(timeout=0.05)
        except Exception:
            pass

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._conn.send(self._packer.pack(payload))
        resp = self._conn.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"value ws server error: {resp}")
        data = msgpack_numpy.unpackb(resp)
        if not isinstance(data, dict):
            raise TypeError(f"Unexpected value ws response type: {type(data)}")
        return data

    def predict(self, transformed_obs: dict[str, Any]) -> float:
        resp = self._request({"cmd": "predict", "observation": transformed_obs})
        return float(resp["value"])

    def predict_phase_value(self, obs: dict[str, Any], phase_class: int) -> float:
        resp = self._request(
            {
                "cmd": "predict_phase_value",
                "observation": obs,
                "phase_class": int(phase_class),
            }
        )
        return float(resp["value"])

    def predict_phase_value_batch(self, obs_batch: list[dict[str, Any]], phase_class: int) -> list[float]:
        resp = self._request(
            {
                "cmd": "predict_phase_value_batch",
                "observations": list(obs_batch),
                "phase_class": int(phase_class),
            }
        )
        values = resp.get("values", [])
        if not isinstance(values, list):
            raise TypeError(f"Unexpected batch value response type: {type(values)}")
        return [float(v) for v in values]

    def predict_keyframe(self, transformed_obs: dict[str, Any]) -> float:
        resp = self._request({"cmd": "predict_keyframe", "observation": transformed_obs})
        return float(resp["keyframe_prob"])

    def predict_keyframe_class(self, transformed_obs: dict[str, Any]) -> int:
        resp = self._request({"cmd": "predict_keyframe_class", "observation": transformed_obs})
        return int(resp["phase_class"])

    def sync_params_from_file(self, params_file: str) -> None:
        self._request({"cmd": "sync_params_from_file", "params_file": params_file})

    def save_state_to_file(self, state_file: str, *, include_optimizer_state: bool = True) -> str:
        resp = self._request(
            {
                "cmd": "save_state_to_file",
                "state_file": state_file,
                "include_optimizer_state": bool(include_optimizer_state),
            }
        )
        return str(resp.get("state_file", state_file))

    def load_state_from_file(self, state_file: str, *, load_optimizer_state: bool = True) -> None:
        self._request(
            {
                "cmd": "load_state_from_file",
                "state_file": state_file,
                "load_optimizer_state": bool(load_optimizer_state),
            }
        )

    @staticmethod
    def _coerce_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in metrics.items():
            key = str(k)
            try:
                out[key] = float(v)
            except (TypeError, ValueError):
                out[key] = v
        return out

    def train_mc_from_lerobot(
        self,
        *,
        dataset_root: str,
        repo_id: str,
        annotations_json: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cmd": "train_mc_from_lerobot",
            "dataset_root": dataset_root,
            "repo_id": repo_id,
        }
        if annotations_json:
            payload["annotations_json"] = annotations_json
        resp = self._request(payload)
        metrics = resp.get("metrics", {})
        if not isinstance(metrics, dict):
            return {}
        return self._coerce_metrics(metrics)

    def train_keyframe_from_lerobot(
        self,
        *,
        dataset_root: str,
        repo_id: str,
        annotations_json: str,
    ) -> dict[str, Any]:
        resp = self._request(
            {
                "cmd": "train_keyframe_from_lerobot",
                "dataset_root": dataset_root,
                "repo_id": repo_id,
                "annotations_json": annotations_json,
            }
        )
        metrics = resp.get("metrics", {})
        if not isinstance(metrics, dict):
            return {}
        return self._coerce_metrics(metrics)

    def train_value_and_keyframe_from_lerobot(
        self,
        *,
        dataset_root: str,
        repo_id: str,
        annotations_json: str,
    ) -> dict[str, Any]:
        resp = self._request(
            {
                "cmd": "train_value_and_keyframe_from_lerobot",
                "dataset_root": dataset_root,
                "repo_id": repo_id,
                "annotations_json": annotations_json,
            }
        )
        out: dict[str, Any] = {}
        for key in ("keyframe_metrics", "value_metrics"):
            metrics = resp.get(key, {})
            if isinstance(metrics, dict):
                out[key] = self._coerce_metrics(metrics)
            else:
                out[key] = {}
        return out

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
