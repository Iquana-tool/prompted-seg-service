from os import getenv
import os

# General paths
ROOT = os.path.dirname(os.path.realpath(__file__))
HYDRA_CONFIGS_DIR = getenv("HYDRA_CONFIGS_DIR", os.path.join(ROOT, "configs"))
HUGGINGFACE_TOKEN = getenv("HF_ACCESS_TOKEN")
LOG_DIR = getenv("LOG_DIR", "logs")
TEMP_DIR = getenv("TEMP_DIR", "temp")
TEMP_IMAGE_DIR = getenv("TEMP_IMAGE_DIR", "./temp/images")
MLFLOW_URL = getenv("MLFLOW_URL", "http://localhost:5000")
REDIS_URL = getenv("REDIS_URL", "redis://localhost:6739")
