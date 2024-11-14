import os

import dotenv

DIR_PATH = os.path.dirname(__file__)

dotenv.load_dotenv(os.path.join(DIR_PATH, ".env"))

TENSORDOCK_AUTH_KEY = os.getenv("TENSORDOCK_AUTH_KEY")
TENSORDOCK_AUTH_TOKEN = os.getenv("TENSORDOCK_AUTH_TOKEN")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
