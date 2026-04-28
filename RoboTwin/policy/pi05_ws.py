import os
import sys

import numpy as np


def _resolve_control_mode(usr_args):
    mode = str(usr_args.get("control_mode", "ee_delta")).strip().lower()
    if mode == "auto":
        train_config_name = str(usr_args.get("train_config_name", "")).lower()
        model_name = str(usr_args.get("model_name", "")).lower()
        if "ee_delta" in train_config_name or "ee_delta" in model_name:
            return "ee_delta"
        return "qpos"
    if mode not in {"qpos", "ee_delta"}:
        raise ValueError(f"Unsupported control_mode `{mode}`.")
    return mode


def _import_websocket_client_policy():
    """Import openpi websocket client with a local fallback path."""
    try:
        from openpi_client import websocket_client_policy as _websocket_client_policy
        return _websocket_client_policy
    except ModuleNotFoundError:
        current_file_path = os.path.abspath(__file__)
        policy_root = os.path.dirname(current_file_path)
        fallback_path = os.path.join(policy_root, "pi05", "packages", "openpi-client", "src")
        if fallback_path not in sys.path:
            sys.path.append(fallback_path)
        from openpi_client import websocket_client_policy as _websocket_client_policy
        return _websocket_client_policy


_websocket_client_policy = _import_websocket_client_policy()


def _import_ee_delta_helpers():
    try:
        from openpi_online_ppo.ee_delta import ee_exec_action16_from_delta_action14
        from openpi_online_ppo.ee_delta import ee_obs14_from_episode_ref
        from openpi_online_ppo.ee_delta import extract_raw_ee_from_env_obs
        return ee_exec_action16_from_delta_action14, ee_obs14_from_episode_ref, extract_raw_ee_from_env_obs
    except ModuleNotFoundError:
        current_file_path = os.path.abspath(__file__)
        policy_root = os.path.dirname(current_file_path)
        fallback_path = os.path.join(policy_root, "pi05", "src")
        if fallback_path not in sys.path:
            sys.path.append(fallback_path)
        from openpi_online_ppo.ee_delta import ee_exec_action16_from_delta_action14
        from openpi_online_ppo.ee_delta import ee_obs14_from_episode_ref
        from openpi_online_ppo.ee_delta import extract_raw_ee_from_env_obs
        return ee_exec_action16_from_delta_action14, ee_obs14_from_episode_ref, extract_raw_ee_from_env_obs


(
    ee_exec_action16_from_delta_action14,
    ee_obs14_from_episode_ref,
    extract_raw_ee_from_env_obs,
) = _import_ee_delta_helpers()


class PI05Websocket:
    def __init__(self, ws_host: str, ws_port: int, pi0_step: int, control_mode: str):
        self.client = _websocket_client_policy.WebsocketClientPolicy(host=ws_host, port=ws_port)
        self.pi0_step = int(pi0_step)
        self.control_mode = str(control_mode)
        self.observation_window = None
        self.instruction = None
        self.left_ref_quat = None
        self.right_ref_quat = None

    def set_language(self, instruction: str):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left = img_arr[0], img_arr[1], img_arr[2]

        # RoboTwin env images are HWC; openpi expects CHW.
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": np.asarray(state, dtype=np.float32),
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.client.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        self.client.reset()
        print("successfully unset obs and language intruction")


def encode_obs(observation):
    raise RuntimeError("encode_obs(observation) without a model is no longer supported.")


def _ee_state_from_obs(observation, model):
    raw_ee = extract_raw_ee_from_env_obs(observation)
    if model.left_ref_quat is None:
        model.left_ref_quat = np.asarray(raw_ee["left_pose7"][3:], dtype=np.float32)
    if model.right_ref_quat is None:
        model.right_ref_quat = np.asarray(raw_ee["right_pose7"][3:], dtype=np.float32)
    return ee_obs14_from_episode_ref(
        left_pose7=raw_ee["left_pose7"],
        left_grip=float(raw_ee["left_grip"]),
        right_pose7=raw_ee["right_pose7"],
        right_grip=float(raw_ee["right_grip"]),
        left_ref_quat_wxyz=model.left_ref_quat,
        right_ref_quat_wxyz=model.right_ref_quat,
    )


def encode_obs_with_model(observation, model):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    if model.control_mode == "ee_delta":
        input_state = _ee_state_from_obs(observation, model)
    else:
        input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


def get_model(usr_args):
    ws_host = usr_args.get("ws_host", "127.0.0.1")
    ws_port = int(usr_args.get("ws_port", 8000))
    pi0_step = int(usr_args.get("pi0_step", 50))
    control_mode = _resolve_control_mode(usr_args)
    print(f"Connecting websocket policy server: ws://{ws_host}:{ws_port}")
    return PI05Websocket(ws_host=ws_host, ws_port=ws_port, pi0_step=pi0_step, control_mode=control_mode)


def eval(TASK_ENV, model, observation):
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs_with_model(observation, model)
    model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()[: model.pi0_step]

    for action in actions:
        if model.control_mode == "ee_delta":
            raw_ee = extract_raw_ee_from_env_obs(observation)
            exec_action = ee_exec_action16_from_delta_action14(
                left_pose7_now=raw_ee["left_pose7"],
                left_grip_now=float(raw_ee["left_grip"]),
                right_pose7_now=raw_ee["right_pose7"],
                right_grip_now=float(raw_ee["right_grip"]),
                delta_action14=np.asarray(action, dtype=np.float32),
            )
            TASK_ENV.take_action(exec_action, action_type="ee")
        else:
            TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs_with_model(observation, model)
        model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model):
    model.left_ref_quat = None
    model.right_ref_quat = None
    model.reset_obsrvationwindows()
