import os
import sys

import numpy as np


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


class PI05Websocket:
    def __init__(self, ws_host: str, ws_port: int, pi0_step: int):
        self.client = _websocket_client_policy.WebsocketClientPolicy(host=ws_host, port=ws_port)
        self.pi0_step = int(pi0_step)
        self.observation_window = None
        self.instruction = None

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
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


def get_model(usr_args):
    ws_host = usr_args.get("ws_host", "127.0.0.1")
    ws_port = int(usr_args.get("ws_port", 8000))
    pi0_step = int(usr_args.get("pi0_step", 50))
    print(f"Connecting websocket policy server: ws://{ws_host}:{ws_port}")
    return PI05Websocket(ws_host=ws_host, ws_port=ws_port, pi0_step=pi0_step)


def eval(TASK_ENV, model, observation):
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()[: model.pi0_step]

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model):
    model.reset_obsrvationwindows()
