import sys

import os
import pathlib
import h5py
import numpy as np
import pickle
import cv2
import argparse
import yaml, json

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openpi_online_ppo.ee_delta import ee_action14_from_pose_pair
from openpi_online_ppo.ee_delta import ee_obs14_from_episode_ref


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        left_endpose = root["/endpose/left_endpose"][()]
        left_ee_gripper = root["/endpose/left_gripper"][()]
        right_endpose = root["/endpose/right_endpose"][()]
        right_ee_gripper = root["/endpose/right_gripper"][()]
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return (
        left_gripper,
        left_arm,
        right_gripper,
        right_arm,
        left_endpose,
        left_ee_gripper,
        right_endpose,
        right_ee_gripper,
        image_dict,
    )


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def get_task_config(task_name):
    with open(f"./task_config/{task_name}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return args


def data_transform(path, episode_num, save_path, representation):
    begin = 0
    floders = os.listdir(path)
    # assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):

        desc_type = "seen"
        instruction_data_path = os.path.join(path, "instructions", f"episode{i}.json")
        with open(instruction_data_path, "r") as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = instruction_dict[desc_type]
        save_instructions_json = {"instructions": instructions}

        os.makedirs(os.path.join(save_path, f"episode_{i}"), exist_ok=True)

        with open(
                os.path.join(os.path.join(save_path, f"episode_{i}"), "instructions.json"),
                "w",
        ) as f:
            json.dump(save_instructions_json, f, indent=2)

        (
            left_gripper_all,
            left_arm_all,
            right_gripper_all,
            right_arm_all,
            left_endpose_all,
            left_ee_gripper_all,
            right_endpose_all,
            right_ee_gripper_all,
            image_dict,
        ) = load_hdf5(os.path.join(path, "data", f"episode{i}.hdf5"))
        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        last_state = None
        for j in range(0, left_gripper_all.shape[0]):

            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )

            joint_state = np.array(left_arm.tolist() + [left_gripper] + right_arm.tolist() + [right_gripper], dtype=np.float32)

            if representation == "ee_delta":
                left_pose = np.asarray(left_endpose_all[j], dtype=np.float32)
                right_pose = np.asarray(right_endpose_all[j], dtype=np.float32)
                left_grip_obs = float(left_ee_gripper_all[j])
                right_grip_obs = float(right_ee_gripper_all[j])
                left_ref_quat = np.asarray(left_endpose_all[0], dtype=np.float32)[3:]
                right_ref_quat = np.asarray(right_endpose_all[0], dtype=np.float32)[3:]
                state = ee_obs14_from_episode_ref(
                    left_pose7=left_pose,
                    left_grip=left_grip_obs,
                    right_pose7=right_pose,
                    right_grip=right_grip_obs,
                    left_ref_quat_wxyz=left_ref_quat,
                    right_ref_quat_wxyz=right_ref_quat,
                )
            else:
                state = joint_state.astype(np.float32)

            if j != left_gripper_all.shape[0] - 1:
                qpos.append(state)

                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_resized = cv2.resize(camera_high, (640, 480))
                cam_high.append(camera_high_resized)

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            if j != 0:
                if representation == "ee_delta":
                    action = ee_action14_from_pose_pair(
                        left_pose7_t=np.asarray(left_endpose_all[j - 1], dtype=np.float32),
                        left_grip_t1=float(left_ee_gripper_all[j]),
                        left_pose7_t1=np.asarray(left_endpose_all[j], dtype=np.float32),
                        right_pose7_t=np.asarray(right_endpose_all[j - 1], dtype=np.float32),
                        right_grip_t1=float(right_ee_gripper_all[j]),
                        right_pose7_t1=np.asarray(right_endpose_all[j], dtype=np.float32),
                    )
                else:
                    action = state
                actions.append(action)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        hdf5path = os.path.join(save_path, f"episode_{i}/episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")
            cam_high_enc, len_high = images_encoding(cam_high)
            cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
            cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
            image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
            image.create_dataset("cam_right_wrist", data=cam_right_wrist_enc, dtype=f"S{len_right}")
            image.create_dataset("cam_left_wrist", data=cam_left_wrist_enc, dtype=f"S{len_left}")

        begin += 1
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        default="beat_block_hammer",
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("setting", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        default=50,
        help="Number of episodes to process (e.g., 50)",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="joint",
        choices=("joint", "ee_delta"),
        help="Output representation for observations/actions.",
    )
    args = parser.parse_args()

    task_name = args.task_name
    setting = args.setting
    expert_data_num = args.expert_data_num
    representation = str(args.representation)

    load_dir = os.path.join("../../data", str(task_name), str(setting))

    begin = 0
    print(f'read data from path:{os.path.join("data", load_dir)}')

    target_dir = f"processed_data/{task_name}-{setting}-{expert_data_num}-{representation}"
    begin = data_transform(
        load_dir,
        expert_data_num,
        target_dir,
        representation,
    )
