from __future__ import annotations

from typing import Any

import numpy as np
import transforms3d as t3d


def _as_pose7(arr: Any) -> np.ndarray:
    pose = np.asarray(arr, dtype=np.float64).reshape(-1)
    if pose.shape[0] != 7:
        raise ValueError(f"Expected pose shape (7,), got {pose.shape}")
    return pose


def _normalize_quat_wxyz(quat: Any) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"Expected quaternion shape (4,), got {q.shape}")
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = q / n
    if q[0] < 0.0:
        q = -q
    return q


def rotvec_from_quat_delta(ref_quat_wxyz: Any, quat_wxyz: Any) -> np.ndarray:
    q_ref = _normalize_quat_wxyz(ref_quat_wxyz)
    q = _normalize_quat_wxyz(quat_wxyz)
    q_rel = _normalize_quat_wxyz(t3d.quaternions.qmult(t3d.quaternions.qinverse(q_ref), q))
    axis, angle = t3d.quaternions.quat2axangle(q_rel)
    if float(abs(angle)) <= 1e-12:
        return np.zeros((3,), dtype=np.float32)
    return np.asarray(np.asarray(axis, dtype=np.float64) * float(angle), dtype=np.float32)


def quat_from_rotvec(rotvec: Any) -> np.ndarray:
    r = np.asarray(rotvec, dtype=np.float64).reshape(-1)
    if r.shape[0] != 3:
        raise ValueError(f"Expected rotvec shape (3,), got {r.shape}")
    angle = float(np.linalg.norm(r))
    if angle <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = r / angle
    return _normalize_quat_wxyz(t3d.quaternions.axangle2quat(axis, angle))


def compose_quat_delta(quat_wxyz: Any, rotvec: Any) -> np.ndarray:
    q = _normalize_quat_wxyz(quat_wxyz)
    q_delta = quat_from_rotvec(rotvec)
    return _normalize_quat_wxyz(t3d.quaternions.qmult(q, q_delta))


def ee_obs14_from_episode_ref(
    *,
    left_pose7: Any,
    left_grip: float,
    right_pose7: Any,
    right_grip: float,
    left_ref_quat_wxyz: Any,
    right_ref_quat_wxyz: Any,
) -> np.ndarray:
    left_pose = _as_pose7(left_pose7)
    right_pose = _as_pose7(right_pose7)
    left_rot = rotvec_from_quat_delta(left_ref_quat_wxyz, left_pose[3:])
    right_rot = rotvec_from_quat_delta(right_ref_quat_wxyz, right_pose[3:])
    state = np.concatenate(
        [
            left_pose[:3].astype(np.float32),
            left_rot.astype(np.float32),
            np.asarray([left_grip], dtype=np.float32),
            right_pose[:3].astype(np.float32),
            right_rot.astype(np.float32),
            np.asarray([right_grip], dtype=np.float32),
        ],
        axis=0,
    )
    if state.shape != (14,):
        raise ValueError(f"Expected 14D ee obs, got {state.shape}")
    return state.astype(np.float32)


def ee_action14_from_pose_pair(
    *,
    left_pose7_t: Any,
    left_grip_t1: float,
    left_pose7_t1: Any,
    right_pose7_t: Any,
    right_grip_t1: float,
    right_pose7_t1: Any,
) -> np.ndarray:
    left_t = _as_pose7(left_pose7_t)
    left_t1 = _as_pose7(left_pose7_t1)
    right_t = _as_pose7(right_pose7_t)
    right_t1 = _as_pose7(right_pose7_t1)
    left_dp = (left_t1[:3] - left_t[:3]).astype(np.float32)
    right_dp = (right_t1[:3] - right_t[:3]).astype(np.float32)
    left_dr = rotvec_from_quat_delta(left_t[3:], left_t1[3:])
    right_dr = rotvec_from_quat_delta(right_t[3:], right_t1[3:])
    action = np.concatenate(
        [
            left_dp,
            left_dr.astype(np.float32),
            np.asarray([left_grip_t1], dtype=np.float32),
            right_dp,
            right_dr.astype(np.float32),
            np.asarray([right_grip_t1], dtype=np.float32),
        ],
        axis=0,
    )
    if action.shape != (14,):
        raise ValueError(f"Expected 14D ee action, got {action.shape}")
    return action.astype(np.float32)


def ee_exec_action16_from_delta_action14(
    *,
    left_pose7_now: Any,
    left_grip_now: float,
    right_pose7_now: Any,
    right_grip_now: float,
    delta_action14: Any,
) -> np.ndarray:
    del left_grip_now, right_grip_now
    left_pose = _as_pose7(left_pose7_now)
    right_pose = _as_pose7(right_pose7_now)
    action = np.asarray(delta_action14, dtype=np.float64).reshape(-1)
    if action.shape[0] != 14:
        raise ValueError(f"Expected 14D delta-EE action, got {action.shape}")

    left_target_pos = left_pose[:3] + action[:3]
    left_target_quat = compose_quat_delta(left_pose[3:], action[3:6])
    left_target_grip = float(action[6])

    right_target_pos = right_pose[:3] + action[7:10]
    right_target_quat = compose_quat_delta(right_pose[3:], action[10:13])
    right_target_grip = float(action[13])

    out = np.concatenate(
        [
            left_target_pos.astype(np.float32),
            left_target_quat.astype(np.float32),
            np.asarray([left_target_grip], dtype=np.float32),
            right_target_pos.astype(np.float32),
            right_target_quat.astype(np.float32),
            np.asarray([right_target_grip], dtype=np.float32),
        ],
        axis=0,
    )
    if out.shape != (16,):
        raise ValueError(f"Expected 16D EE execution action, got {out.shape}")
    return out.astype(np.float32)


def extract_raw_ee_from_env_obs(raw_obs: dict[str, Any]) -> dict[str, np.ndarray | float]:
    endpose = raw_obs.get("endpose")
    if not isinstance(endpose, dict):
        raise KeyError("Raw env observation does not contain `endpose`.")
    left_pose = _as_pose7(endpose["left_endpose"]).astype(np.float32)
    right_pose = _as_pose7(endpose["right_endpose"]).astype(np.float32)
    left_grip = float(endpose["left_gripper"])
    right_grip = float(endpose["right_gripper"])
    return {
        "left_pose7": left_pose,
        "left_grip": left_grip,
        "right_pose7": right_pose,
        "right_grip": right_grip,
    }

