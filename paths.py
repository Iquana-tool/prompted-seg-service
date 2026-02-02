from os import getenv
import os

# General paths
ROOT = os.path.dirname(os.path.realpath(__file__))
HYDRA_CONFIGS_DIR = getenv("HYDRA_CONFIGS_DIR", os.path.join(ROOT, "configs"))
HUGGINGFACE_TOKEN = getenv("HUGGINGFACE_TOKEN")
LOG_DIR = getenv("LOG_DIR", "logs")
TEMP_DIR = getenv("TEMP_DIR", "temp")
TEMP_IMAGE_DIR = getenv("TEMP_IMAGE_DIR", "./temp/images")
REDIS_URL = os.environ.get("REDIS_URL", "localhost:6739")

# Weights and configs paths
SAM2_TINY_WEIGHTS = getenv("SAM2_TINY_WEIGHTS", "./weights/sam2.1_hiera_tiny.pt")
SAM2_TINY_CONFIG = getenv("SAM2_TINY_CONFIG", "sam2.1_hiera_tiny") 
SAM2_SMALL_WEIGHTS = getenv("SAM2_SMALL_WEIGHTS", "./weights/sam2.1_hiera_small.pt")
SAM2_SMALL_CONFIG = getenv("SAM2_SMALL_CONFIG", "sam2.1_hiera_small")  
SAM2_BASE_WEIGHTS = getenv("SAM2_BASE_WEIGHTS", "./weights/sam2.1_hiera_base_plus.pt")
SAM2_BASE_CONFIG = getenv("SAM2_BASE_CONFIG", "sam2.1_hiera_base_plus")  
SAM2_LARGE_WEIGHTS = getenv("SAM2_LARGE_WEIGHTS", "./weights/sam2.1_hiera_large.pt")
SAM2_LARGE_CONFIG = getenv("SAM2_LARGE_CONFIG", "sam2.1_hiera_large") 