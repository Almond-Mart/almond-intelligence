import argparse
import requests
import os

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
# 3. Create ssh key
# 4. Add ssh key to GitHub
# 5. Clone repo
# 6. Follow LeRobot instructions & transfer training data
# 7. Add wandb key
# 8. Move systemd script to /etc/systemd/system/
# 9. Start service
# 10. Save checkpoint models to Google Drive
# 11. Stop service when training is complete
# 12. Delete server

def available_datasets() -> str:
    dataset_path = os.path.join(DIR_PATH, "data", USERNAME)
    datasets = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    return datasets

def get_cheapest_matching_server() -> tuple[str, str]:
    available_servers_url = "https://marketplace.tensordock.com/api/v0/client/deploy/hostnodes"
    available_servers_params = {
        "minvCPUs": CPUS,
        "minRAM": RAM,
        "minStorage": STORAGE,
        "minGPUCount": GPUS,
        "maxGPUCount": GPUS,
        "minVRAM": VRAM,
    }

    available_servers = requests.get(available_servers_url, params=available_servers_params).json()
    matching_servers = {k: v for k, v in available_servers["hostnodes"].items() if GPU_MODEL in v["specs"]["gpu"] and v["status"]["uptime"] >= MIN_UPTIME}

    def server_price(server: tuple[str, dict]) -> float:
        specs = server[1]["specs"]

        cpu_price = specs["cpu"]["price"] * CPUS
        ram_price = specs["ram"]["price"] * RAM
        storage_price = specs["storage"]["price"] * STORAGE
        gpu_price = specs["gpu"][GPU_MODEL]["price"] * GPUS
        
        return cpu_price + ram_price + storage_price + gpu_price

    cheapest_matching_server = sorted(matching_servers.items(), key=server_price)[0]
    hostnode = cheapest_matching_server[0]
    first_port = cheapest_matching_server[1]["networking"]["ports"][0]

    return hostnode, first_port

def print_server_runtime():
    balance_url = "https://marketplace.tensordock.com/api/v0/billing/balance"
    balance_params = {
        "api_key": env.TENSORDOCK_AUTH_KEY,
        "api_token": env.TENSORDOCK_AUTH_TOKEN,
    }

    balance = requests.get(balance_url, params=balance_params).json()
    
    remaining_balance = balance["balance"]
    hourly_rate = balance["hourly_rate"]
    runtime = remaining_balance / hourly_rate

    print(f"Remaining balance: ${remaining_balance:.2f}")
    print(f"Hourly rate: ${hourly_rate:.2f}")
    print(f"Runtime: {runtime:.2f} hours")

def create_server():
    hostnode, first_port = get_cheapest_matching_server()

    ssh_key_path = os.path.join("~", ".ssh", "id_ed25519.pub")
    if not os.path.exists(ssh_key_path):
        print("SSH key not found. Please generate an SSH key.\nssh-keygen -t ed25519 -C \"your_email@example.com\"")
        exit(1)

    with open(ssh_key_path) as f:
        ssh_key = f.read()

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
        "external_ports": f"{{{first_port}}}",
        "internal_ports": "{22}",
    }

    response = requests.post(deploy_server_url, params=deploy_server_params)
    assert response.ok, f"Failed to create server: {response.json()}"

    print("Started server")
    print_server_runtime()

def main():
    datasets = available_datasets()

    parser = argparse.ArgumentParser(description="Train Almond Intelligence Model")
    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start", help="Start training service.")
    start_parser.add_argument("dataset", choices=datasets, help="Dataset to use for training.")

    train_parser = subparsers.add_parser("train", help="Train model.")

    stop_parser = subparsers.add_parser("stop", help="Stop training service.")

    args = parser.parse_args()

    if args.command == "start":
        create_server()

if __name__ == "__main__":
    main()