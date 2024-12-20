from argparse import ArgumentParser
import asyncio
import json
import os
import socket

import numpy as np
import torch
from torch import Tensor
import websockets
from websockets import WebSocketClientProtocol

from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

DEFAULT_DEVICE = "mps"

client: WebSocketClientProtocol = None

async def connect():
    global client
    uri = "ws://raspberrypi.local:8000"

    try:
        print("Connecting to Almond rPi")
        client = await websockets.connect(uri, extra_headers={"Almond-Client-Type": "inference"})
    except socket.gaierror:
        print("Failed to connect to Almond rPi")
        exit(1)

    print("Connected to Almond rPi")

def run_inference(policy: ACTPolicy | DiffusionPolicy, pictures: dict[str, list], positions: list[float]) -> list[float]:
    # Read the follower state and access the frames from the cameras
    observation: dict[str, Tensor] = {}
    observation["observation.state"] = torch.as_tensor(positions)
    for name, picture in pictures.items():
        observation[f"observation.images.{name}"] = torch.from_numpy(np.array(picture))

    # Convert to pytorch format: chann`el first and float32 in [0,1]
    # with batch dimension
    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(DEFAULT_DEVICE)

    # Compute the next action with the policy
    # based on the current observation
    action = policy.select_action(observation)
    # Remove batch dimension
    action = action.squeeze(0)
    # Move to cpu, if not already the case
    if DEFAULT_DEVICE != "cpu":
        action = action.to("cpu")
    # Order the robot to move
    return list(action.numpy())

async def inference_loop(model: str, model_path: str):
    if model == "act":
        policy = ACTPolicy.from_pretrained(model_path)
    elif model == "diffusion":
        policy = DiffusionPolicy.from_pretrained(model_path)

    policy.to(DEFAULT_DEVICE)

    async for data in client:
        data = json.loads(data)

        inference = run_inference(policy, data["pictures"], data["positions"])
        client.send(json.dumps({"inference": inference}))

async def main(model: str, model_path: str):
    if not os.path.isfile(model_path):
        print(f"Model file not found: {model_path}")
        exit(1)

    await connect()
    await inference_loop(model, model_path)

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Almond Intellignece",
        description="Control Almond rPi with AI"
    )

    parser.add_argument("--model", type=str.lower, choices=["act", "diffusion"], required=True, help="Model to use for inference.")
    parser.add_argument("--model_path", required=True, help="Path to the model file.")
    args = vars(parser.parse_args())

    asyncio.run(main(**args))