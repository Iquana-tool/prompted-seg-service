from os import getenv


SAM2_TINY_WEIGHTS = getenv("SAM2_TINY_WEIGHTS", "weights/sam2.1_hiera_tiny.pt")
SAM2_TINY_CONFIG = getenv("SAM2_TINY_CONFIG", "configs/sam2.1_hiera_tiny.yaml")
SAM2_SMALL_WEIGHTS = getenv("SAM2_SMALL_WEIGHTS", "weights/sam2.1_hiera_small.pt")
SAM2_SMALL_CONFIG = getenv("SAM2_SMALL_CONFIG", "configs/sam2.1_hiera_small.yaml")
SAM2_BASE_WEIGHTS = getenv("SAM2_BASE_WEIGHTS", "weights/sam2.1_hiera_base.pt")
SAM2_BASE_CONFIG = getenv("SAM2_BASE_CONFIG", "configs/sam2.1_hiera_base.yaml")
SAM2_LARGE_WEIGHTS = getenv("SAM2_LARGE_WEIGHTS", "weights/sam2.1_hiera_large.pt")
SAM2_LARGE_CONFIG = getenv("SAM2_LARGE_CONFIG", "configs/sam2.1_hiera_large.yaml")