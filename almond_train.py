import argparse
import requests
import os
import time

import paramiko
from paramiko import SSHClient
from paramiko.ssh_exception import NoValidConnectionsError
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
# 4. Clone repo with fine-grained personal access token
# 5. Follow LeRobot instructions & transfer training data
# 6. Add wandb key
# 7. Move systemd script to /etc/systemd/system/
# 8. Start service
# 9. Save checkpoint models to Google Drive
# 10. Stop service when training is complete
# 11. Delete server

dataset = None

hostnode = None

port = None
user = None
ip = None

ssh_client = None
scp_client = None

current_scp_file = None

def scp_progress(filename: bytes, size: int, sent: int):
    global current_scp_file
    filename = filename.decode()

    if filename != current_scp_file:
        if current_scp_file is not None:
            print()

        current_scp_file = filename
        print(f"{filename}: ", end="")

    percent = int(sent / size * 100)
    print(f"\r{filename}: {percent}% complete", end="")

def exec_remote_command(command: str | list[str], single: bool = False) -> int:
    assert ssh_client is not None, "SSH client not connected."

    if isinstance(command, str):
        command = [command]
    elif single:
        command = [" && ".join(command)]

    for cmd in command:
        _, stdout, _ = ssh_client.exec_command(cmd)
        exit_status = stdout.channel.recv_exit_status()

        if exit_status != 0:
            return exit_status

def transfer_file(local_path: str, remote_path: str):
    assert scp_client is not None, "SCP client not connected."
    global current_scp_file

    recursive = os.path.isdir(local_path)
    scp_client.put(local_path, remote_path, recursive)
    current_scp_file = None

# MARK: Create and connect to server

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

def connect_to_server():
    server_details_url = "https://marketplace.tensordock.com/api/v0/client/list"
    server_details_params = {
        "api_key": env.TENSORDOCK_AUTH_KEY,
        "api_token": env.TENSORDOCK_AUTH_TOKEN,
    }

    global user, ip
    while True:
        server_details = requests.post(server_details_url, data=server_details_params).json()
        if "virtualmachines" in server_details:
            vms = server_details["virtualmachines"]
            vm = next(iter(vms.values()))

            if vm["status"].lower() == "running":
                user = vm["default_user"]
                ip = vm["ip_address"]

                break

        time.sleep(5)

    assert ip is not None and port is not None and user is not None, "Server not started."
    global ssh_client, scp_client

    if ssh_client is not None:
        ssh_client.close()
    if scp_client is not None:
        scp_client.close()


    ssh_client = SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    while ssh_client.get_transport() is None or not ssh_client.get_transport().is_active():
        try:
            ssh_client.connect(hostname=ip, port=port, username=user)
        except NoValidConnectionsError as e:
            time.sleep(5)

    scp_client = SCPClient(ssh_client.get_transport(), progress=scp_progress)

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

    print("Waiting for server to start")
    connect_to_server()
    print("Connected to server")

    # Check if needs restart to activate GPU
    _, stdout, _ = ssh_client.exec_command("nvidia-smi")
    if stdout.read().decode().strip().lower().startswith("failed"):
        print("GPU not detected. Rebooting server...")
        ssh_client.exec_command("sudo reboot")
    
        time.sleep(5)

        print("Waiting for server to start")
        connect_to_server()
        print("Connected to server")

# MARK: Clone & setup repo

def clone_repo():
    exec_remote_command(f"git clone https://{env.GITHUB_TOKEN}@github.com/Almond-Mart/almond-intelligence.git")

def install_miniconda():
    exec_remote_command(
        [
            "mkdir -p ~/miniconda3",
            "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh",
            "bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3",
            "rm ~/miniconda3/miniconda.sh",
            "source ~/miniconda3/bin/activate",
            "conda init --all"
        ],
        single=True
    )

def install_dependencies():
    exec_remote_command(
        [
            "cd almond-intelligence",
            "source ~/miniconda3/bin/activate",
            "conda create -y -n lerobot python=3.10",
            "conda activate lerobot", 
            "pip install -e .",
            f"wandb login {env.WANDB_API_KEY}"
        ],
        single=True
    )

def transfer_training_data():
    exec_remote_command(f"mkdir -p ~/almond-intelligence/data/{USERNAME}")

    transfer_file(
        os.path.join(DIR_PATH, "data", USERNAME, dataset),
        os.path.join("~", "almond-intelligence", "data", USERNAME)
    )

def main():
    datasets = available_datasets()

    parser = argparse.ArgumentParser(description="Train Almond Intelligence Model")
    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start", help="Start training service.")
    start_parser.add_argument("--dataset", choices=datasets, help="Dataset to use for training.")

    train_parser = subparsers.add_parser("train", help="Train model.")

    stop_parser = subparsers.add_parser("stop", help="Stop training service.")

    args = parser.parse_args()

    if args.command == "start":
        global dataset
        dataset = args.dataset

        create_server()

        print("Cloning repo and installing miniconda")
        clone_repo()
        install_miniconda()

        print("Installing dependencies and transferring training data")
        install_dependencies()
        transfer_training_data()

if __name__ == "__main__":
    main()