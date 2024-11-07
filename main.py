import asyncio
import json
import numpy as np
import torch
from torch import Tensor
import websockets
from websockets import WebSocketClientProtocol

from lerobot.common.policies.act.modeling_act import ACTPolicy

DEFAULT_DEVICE = "mps"

client: WebSocketClientProtocol = None

async def connect():
    global client

    print("Connecting to Almond rPi")
    uri = "ws://raspberrypi.local:8000"
    client = await websockets.connect(uri)
    client.send(json.dumps({"type": "inference"}))
    print("Connected to Almond rPi")

def run_inference(pictures: dict[str, list], positions: list[float], model_path: str) -> list[float]:
    policy = ACTPolicy.from_pretrained(model_path)
    policy.to(DEFAULT_DEVICE)

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

async def inference_loop(model_path: str):
    async for data in client:
        data = json.loads(data)

        inference = run_inference(data["pictures"], data["positions"], model_path)
        client.send(json.dumps({"inference": inference}))

async def main(model_path: str):
    await connect()
    await inference_loop()

if __name__ == "__main__":
    asyncio.run(main())