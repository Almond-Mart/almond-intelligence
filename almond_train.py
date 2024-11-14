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
GPU_TYPE = "geforcertx4090-pcie-24gb"
MIN_UPTIME = 0.999

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

def create_server():
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
    matching_servers = {k: v for k, v in available_servers["hostnodes"].items() if GPU_TYPE in v["specs"]["gpu"] and v["status"]["uptime"] >= MIN_UPTIME}

    def server_price(server: tuple[str, dict]) -> float:
        specs = server[1]["specs"]

        cpu_price = specs["cpu"]["price"] * CPUS
        ram_price = specs["ram"]["price"] * RAM
        storage_price = specs["storage"]["price"] * STORAGE
        gpu_price = specs["gpu"][GPU_TYPE]["price"] * GPUS
        
        return cpu_price + ram_price + storage_price + gpu_price

    cheapest_matching_server = sorted(matching_servers.items(), key=server_price)[0]

    print(cheapest_matching_server)

def main():
    datasets = available_datasets()

    parser = argparse.ArgumentParser(description="Train Almond Intelligence Model")
    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start", help="Start training service.")
    start_parser.add_argument("dataset", choices=datasets, help="Dataset to use for training.")

    train_parser = subparsers.add_parser("train", help="Train model.")

    stop_parser = subparsers.add_parser("stop", help="Stop training service.")

    args = parser.parse_args()

    create_server()

if __name__ == "__main__":
    main()