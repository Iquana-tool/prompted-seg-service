import logging
from typing import Callable

from iquana_toolbox.mlflow import MLFlowModelRegistry

from models.sam2 import SAMPrompted

logger = logging.getLogger(__name__)

MODEL_REGISTRY_CONFIG = [
    {
        "model_identifier": "sam2-1-tiny",
        "model_factory": lambda: SAMPrompted("facebook/sam2.1-hiera-tiny"),
        "desc": "Segment Anything Model 2.1 - Tiny variant. The smallest and fastest model with lowest memory footprint. Suitable for real-time inference but with reduced accuracy. Supports point and box prompts.",
        "tags": {
            "task": "prompted-segmentation",
            "status": "ready",
            "pretrained": "true",
            "trainable": "false",
            "finetunable": "false",
            "model_size": "tiny",
            "inference_speed": "fastest",
            "accuracy_level": "low",
            "prompt_types_supported": "point,box",
            "refinement_supported": "true",
            "requires_gpu": "false",
        }
    },
    {
        "model_identifier": "sam2-1-small",
        "model_factory": lambda: SAMPrompted("facebook/sam2.1-hiera-small"),
        "desc": "Segment Anything Model 2.1 - Small variant. Provides a good balance between inference speed and segmentation accuracy. Ideal for production use cases requiring reasonable performance. Supports point and box prompts.",
        "tags": {
            "task": "prompted-segmentation",
            "status": "ready",
            "pretrained": "true",
            "trainable": "false",
            "finetunable": "false",
            "model_size": "small",
            "inference_speed": "fast",
            "accuracy_level": "medium",
            "prompt_types_supported": "point,box",
            "refinement_supported": "true",
            "requires_gpu": "true",
        }
    },
    {
        "model_identifier": "sam2-1-base-plus",
        "model_factory": lambda: SAMPrompted("facebook/sam2.1-hiera-base-plus"),
        "desc": "Segment Anything Model 2.1 - Base+ variant. Larger model with improved accuracy compared to small variant. Good choice for accuracy-critical applications. Supports point and box prompts with refinement capabilities.",
        "tags": {
            "task": "prompted-segmentation",
            "status": "ready",
            "pretrained": "true",
            "trainable": "false",
            "finetunable": "false",
            "model_size": "base-plus",
            "inference_speed": "medium",
            "accuracy_level": "high",
            "prompt_types_supported": "point,box",
            "refinement_supported": "true",
            "requires_gpu": "true",
        }
    },
    {
        "model_identifier": "sam2-1-large",
        "model_factory": lambda: SAMPrompted("facebook/sam2.1-hiera-large"),
        "desc": "Segment Anything Model 2.1 - Large variant. The largest and most accurate SAM2 model. Best segmentation quality but requires more VRAM and slower inference. Recommended for offline and accuracy-critical workflows. Supports point and box prompts.",
        "tags": {
            "task": "prompted-segmentation",
            "status": "ready",
            "pretrained": "true",
            "trainable": "false",
            "finetunable": "false",
            "model_size": "large",
            "inference_speed": "slowest",
            "accuracy_level": "highest",
            "prompt_types_supported": "point,box",
            "refinement_supported": "true",
            "requires_gpu": "true",
        }
    },
]


def register_models(model_registry: MLFlowModelRegistry):
    """Lazily register models only if they don't already exist in MLflow.
    
    This avoids expensive model instantiation at startup if the models are already
    registered in the MLflow registry.
    """
    for config in MODEL_REGISTRY_CONFIG:
        model_id = config["model_identifier"]

        if model_registry.check_registered(model_id):
            continue

        logger.info(f"Registering model '{model_id}' (not found in registry)...")
        try:
            model = config["model_factory"]()
            model_registry.register_model(
                model_identifier=model_id,
                model=model,
                desc=config["desc"],
                tags=config["tags"]
            )
            logger.info(f"Successfully registered model '{model_id}'.")
        except Exception as e:
            logger.error(f"Failed to register model '{model_id}': {e}")
            raise
