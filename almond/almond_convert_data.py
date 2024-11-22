import argparse
from itertools import chain
import os

import cv2
import torch

from lerobot.common.datasets import populate_dataset

DIR_PATH = os.path.dirname(__file__)
TRAINING_POSITIONS = ["center", "left_arm", "right_arm"]

def merge_columns(nested_list: list[list[str]]) -> list[str]:
    return [",".join(row[i] for row in nested_list) for i in range(len(nested_list[0]))]

def convert_dataset(almond_repo: str, lerobot_repo: str):
    dataset = populate_dataset.init_dataset(
        repo_id=f"shawnptl8/{lerobot_repo}",
        root="data",
        force_override=False,
        fps=50,
        video=True,
        write_images=True,
        num_image_writer_processes=0,
        num_image_writer_threads=4, # 4 threads per camera
    )

    almond_dataset_dir = os.path.join(DIR_PATH, "..", "almond_data", almond_repo)

    with os.scandir(almond_dataset_dir) as episodes:
        for episode in episodes:
            if not episode.is_dir():
                continue

            frames_dir = os.path.join(episode.path, "frames")

            observation_positions_paths = [os.path.join(episode.path, "positions", "observation", f"{name}_positions.txt") for name in TRAINING_POSITIONS]
            action_positions_paths = [os.path.join(episode.path, "positions", "action", f"{name}_positions.txt") for name in TRAINING_POSITIONS]

            observation_position_files = [open(path, "r") for path in observation_positions_paths]
            action_position_files = [open(path, "r") for path in action_positions_paths]

            observation_positions = merge_columns([f.readlines() for f in observation_position_files])
            action_positions = merge_columns([f.readlines() for f in action_position_files])

            for i, (observation_position, action_position) in enumerate(zip(observation_positions, action_positions)):
                if i == 0:
                    continue

                image_path = os.path.join(frames_dir, f"{i}.jpg")
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)
                image = torch.from_numpy(image)

                observation_values = observation_position.split(",")
                observation = torch.tensor(observation_values)

                action_values = action_position.split(",")
                action = torch.tensor(action_values)

                observation = {
                    "observation.images.fpv": image,
                    "observation.state": observation,
                }

                action = {
                    "action": action,
                }

                populate_dataset.add_frame(
                    dataset=dataset,
                    observation=observation,
                    action=action,
                )

            for position in observation_position_files:
                position.close()
            for position in action_position_files:
                position.close()

            populate_dataset.save_current_episode(dataset)

    populate_dataset.create_lerobot_dataset(
        dataset=dataset,
        run_compute_stats=False,
        push_to_hub=False,
        tags=None,
        play_sounds=False,
    )

def main():
    parser = argparse.ArgumentParser(description="Convert AlmondDataset to LeRobotDataset")
    parser.add_argument("--almond-repo", help="Repo ID of the AlmondDataset.")
    parser.add_argument("--lerobot-repo", help="Repo ID of the LeRobotDataset.")
    args = parser.parse_args()

    convert_dataset(args.almond_repo, args.lerobot_repo)

if __name__ == "__main__":
    main()