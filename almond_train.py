import argparse
import asyncio
import requests
import os
import time

import paramiko
from paramiko import SSHClient
from scp import SCPClient

import env

DIR_PATH = os.path.dirname(__file__)
USERNAME = "shawnptl8"

CPUS = 2
RAM = 4
STORAGE = 32
GPUS = 1
VRAM = 24
GPU_MODEL = "geforcertx4090-pcie-24gb"
MIN_UPTIME = 0.999
OPERATING_SYSTEM = "Ubuntu 22.04 LTS"

# FLOW
# 1. Ask what data to use
# 2. Create server
# 3. Check if needs restart to activate GPU
# 4. Create ssh key
# 5. Add ssh key to GitHub
# 6. Clone repo
# 7. Follow LeRobot instructions & transfer training data
# 8. Add wandb key
# 9. Move systemd script to /etc/systemd/system/
# 10. Start service
# 11. Save checkpoint models to Google Drive
# 12. Stop service when training is complete
# 13. Delete server

hostnode = None

port = None
user = None
ip = None

ssh_client = None
scp_client = None

def connect_to_server():
    assert ip is not None and port is not None and user is not None, "Server not started."
    global ssh_client, scp_client

    ssh_client = SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=ip, port=port, username=user)

    scp_client = SCPClient(ssh_client.get_transport())

def available_datasets() -> str:
    dataset_path = os.path.join(DIR_PATH, "data", USERNAME)
    datasets = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith("eval_")]
    return datasets

def get_cheapest_matching_server():
    available_servers_url = "https://marketplace.tensordock.com/api/v0/client/deploy/hostnodes"
    available_servers_params = {
        "minvCPUs": CPUS,
        "minRAM": RAM,
        "minStorage": STORAGE,
        "minGPUCount": GPUS,
        "maxGPUCount": GPUS,
        "minVRAM": VRAM,
    }

    available_servers = requests.get(available_servers_url, data=available_servers_params).json()
    matching_servers = {k: v for k, v in available_servers["hostnodes"].items() if GPU_MODEL in v["specs"]["gpu"] and v["status"]["uptime"] >= MIN_UPTIME}
    if not matching_servers:
        print("No matching servers found.")
        exit(1)

    def server_price(server: tuple[str, dict]) -> float:
        specs = server[1]["specs"]

        cpu_price = specs["cpu"]["price"] * CPUS
        ram_price = specs["ram"]["price"] * RAM
        storage_price = specs["storage"]["price"] * STORAGE
        gpu_price = specs["gpu"][GPU_MODEL]["price"] * GPUS
        
        return cpu_price + ram_price + storage_price + gpu_price

    cheapest_matching_server = sorted(matching_servers.items(), key=server_price)[0]

    global hostnode, port

    hostnode = cheapest_matching_server[0]
    port = cheapest_matching_server[1]["networking"]["ports"][0]

def print_server_runtime():
    balance_url = "https://marketplace.tensordock.com/api/v0/billing/balance"
    balance_params = {
        "api_key": env.TENSORDOCK_AUTH_KEY,
        "api_token": env.TENSORDOCK_AUTH_TOKEN,
    }

    balance = requests.post(balance_url, data=balance_params).json()
    
    remaining_balance = balance["balance"]
    hourly_cost = balance["hourly_cost"]
    runtime = remaining_balance / hourly_cost

    print(f"Balance: ${remaining_balance:.2f}")
    print(f"Hourly cost: ${hourly_cost:.2f}")
    print(f"Runtime: {runtime:.2f} hours")

def wait_for_server_start():
    server_details_url = "https://marketplace.tensordock.com/api/v0/client/get/single"
    server_details_params = {
        "api_key": env.TENSORDOCK_AUTH_KEY,
        "api_token": env.TENSORDOCK_AUTH_TOKEN,
        "server": hostnode,
    }

    while True:
        server_details = requests.post(server_details_url, data=server_details_params).json()
        if "virtualmachines" in server_details and server_details["virtualmachines"]["status"].lower() == "running":
            global user, ip
            user = server_details["virtualmachines"]["default_user"]
            ip = server_details["virtualmachines"]["ip_address"]

        time.sleep(5)

def create_server():
    get_cheapest_matching_server()

    ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519.pub")
    if not os.path.exists(ssh_key_path):
        print("SSH key not found. Please generate a SSH key.\nssh-keygen -t ed25519 -C \"your_email@example.com\"")
        exit(1)

    with open(ssh_key_path) as f:
        ssh_key = f.read().strip()

    deploy_server_url = "https://marketplace.tensordock.com/api/v0/client/deploy/single"
    deploy_server_params = {
        "api_key": env.TENSORDOCK_AUTH_KEY,
        "api_token": env.TENSORDOCK_AUTH_TOKEN,
        "ssh_key": ssh_key,
        "name": "almond-intelligence",
        "gpu_count": GPUS,
        "gpu_model": GPU_MODEL,
        "vcpus": CPUS,
        "ram": RAM,
        "storage": STORAGE,
        "hostnode": hostnode,
        "operating_system": OPERATING_SYSTEM,
        "external_ports": f"{{{port}}}",
        "internal_ports": "{22}",
    }

    response = requests.post(deploy_server_url, data=deploy_server_params)
    assert response.ok, f"Failed to create server: {response.json()}"

    print("Requested server")
    print_server_runtime()

    wait_for_server_start()
    print("Server started")

    connect_to_server()
    print("Connected to server")

    # Check if needs restart to activate GPU
    _, stdout, _ = ssh_client.exec_command("nvidia-smi")
    if stdout.read().decode().strip().lower().startswith("failed"):
        print("GPU not detected. Rebooting server...")
        ssh_client.exec_command("sudo reboot")
    
        ssh_client.close()
        scp_client.close()

        time.sleep(5)
        
        wait_for_server_start()
        print("Server started")
        connect_to_server()
        print("Connected to server")

async def main():
    datasets = available_datasets()

    parser = argparse.ArgumentParser(description="Train Almond Intelligence Model")
    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start", help="Start training service.")
    start_parser.add_argument("--dataset", choices=datasets, help="Dataset to use for training.")

    train_parser = subparsers.add_parser("train", help="Train model.")

    stop_parser = subparsers.add_parser("stop", help="Stop training service.")

    args = parser.parse_args()

    if args.command == "start":
        global ip, port, user
        ip = "70.62.164.136"
        port = "28206"
        user = "user"
        create_server()

if __name__ == "__main__":
    asyncio.run(main())