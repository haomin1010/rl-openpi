import numpy as np
import torch
import dill
import os, sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)
sys.path.append(os.path.join(parent_directory, "src"))

from pi_model import *
from openpi_online_ppo.ee_delta import ee_exec_action16_from_delta_action14
from openpi_online_ppo.ee_delta import ee_obs14_from_episode_ref
from openpi_online_ppo.ee_delta import extract_raw_ee_from_env_obs


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


# Encode observation for the model
def encode_obs(observation, model):
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
    train_config_name, model_name, checkpoint_id, pi0_step = (usr_args["train_config_name"], usr_args["model_name"],
                                                              usr_args["checkpoint_id"], usr_args["pi0_step"])
    control_mode = _resolve_control_mode(usr_args)
    return PI0(train_config_name, model_name, checkpoint_id, pi0_step, control_mode=control_mode)


def eval(TASK_ENV, model, observation):

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation, model)
    model.update_observation_window(input_rgb_arr, input_state)

    # ======== Get Action ========

    actions = model.get_action()[:model.pi0_step]

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
        input_rgb_arr, input_state = encode_obs(observation, model)
        model.update_observation_window(input_rgb_arr, input_state)

    # ============================


def reset_model(model):
    model.left_ref_quat = None
    model.right_ref_quat = None
    model.reset_obsrvationwindows()
