from os import getenv
import os

# General paths
ROOT = os.path.dirname(os.path.realpath(__file__))
HYDRA_CONFIGS_DIR = getenv("HYDRA_CONFIGS_DIR", os.path.join(ROOT, "configs"))
LOG_DIR = getenv("LOG_DIR", "logs")
TEMP_DIR = getenv("TEMP_DIR", "temp")
TEMP_IMAGE_DIR = getenv("TEMP_IMAGE_DIR", "./temp/images")

# Weights and configs paths
SAM2_TINY_WEIGHTS = getenv("SAM2_TINY_WEIGHTS", "./weights/sam2.1_hiera_tiny.pt")
SAM2_TINY_CONFIG = getenv("SAM2_TINY_CONFIG", os.path.join(HYDRA_CONFIGS_DIR, "sam2.1_hiera_tiny.yaml"))
SAM2_SMALL_WEIGHTS = getenv("SAM2_SMALL_WEIGHTS", "./weights/sam2.1_hiera_small.pt")
SAM2_SMALL_CONFIG = getenv("SAM2_SMALL_CONFIG", os.path.join(HYDRA_CONFIGS_DIR, "sam2.1_hiera_small.yaml"))
SAM2_BASE_WEIGHTS = getenv("SAM2_BASE_WEIGHTS", "./weights/sam2.1_hiera_baseplus.pt")
SAM2_BASE_CONFIG = getenv("SAM2_BASE_CONFIG", os.path.join(HYDRA_CONFIGS_DIR, "sam2.1_hiera_baseplus.yaml"))
SAM2_LARGE_WEIGHTS = getenv("SAM2_LARGE_WEIGHTS", "./weights/sam2.1_hiera_large.pt")
SAM2_LARGE_CONFIG = getenv("SAM2_LARGE_CONFIG", os.path.join(HYDRA_CONFIGS_DIR, "sam2.1_hiera_large.yaml"))