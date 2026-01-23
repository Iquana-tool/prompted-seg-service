from models.model_registry import ModelRegistry
from schemas.models import PromptedSegmentationModels
from models.sam2 import SAM2ModelLoader
from paths import *


def register_models(model_registry: ModelRegistry):
    """ This function registers all models in the MODEL_REGISTRY. You can extend it to add custom models. """
    model_registry.register_model(
        model_info=PromptedSegmentationModels(
            registry_key="sam2.1_tiny",
            name="SAM2.1 Tiny",
            description="Segment Anything Model 2.1 - Tiny version. The smallest and fastest model. Accuracy is lower than other models.",
            tags=["Fast", "General Purpose"],
            number_of_parameters=0,
            pretrained=True,
            trainable=False,
            finetunable=False,
            prompt_types_supported=["point", "box"],
            refinement_supported=True,
        ),
        model_loader=SAM2ModelLoader(
            weights=SAM2_TINY_WEIGHTS,
            config=SAM2_TINY_CONFIG)
    )
    model_registry.register_model(
        model_info=PromptedSegmentationModels(
            registry_key="sam2.1_small",
            name="SAM2.1 Small",
            description="Segment Anything Model 2 - Small version. A good balance between speed and accuracy.",
            tags=["Sam2", "Small", "Fast", "General Purpose"],
            number_of_parameters=0,
            pretrained=True,
            trainable=False,
            finetunable=False,
            prompt_types_supported=["point", "box"],
            refinement_supported=True,
         ),
        model_loader=SAM2ModelLoader(
            weights=SAM2_SMALL_WEIGHTS,
            config=SAM2_SMALL_CONFIG)
    )
    model_registry.register_model(
        model_info=PromptedSegmentationModels(
            registry_key="sam2.1_baseplus",
            name="SAM2.1 Base+",
            description="Segment Anything Model 2.1 - Base+ version. A larger model with better accuracy.",
            tags=["Sam2", "Accurate", "Medium", "General Purpose"],
            number_of_parameters=0,
            pretrained=True,
            trainable=False,
            finetunable=False,
            prompt_types_supported=["point", "box"],
            refinement_supported=True,
         ),
        model_loader=SAM2ModelLoader(
            weights=SAM2_BASE_WEIGHTS,
            config=SAM2_BASE_CONFIG)
    )
    model_registry.register_model(
        model_info=PromptedSegmentationModels(
            registry_key="sam2.1_large",
            name="SAM2.1 Large",
            description="Segment Anything Model 2 - Large version. The largest SAM2 model with the best accuracy. Requires more VRAM and is slower.",
            tags=["Slow", "General Purpose"],
            number_of_parameters=0,
            pretrained=True,
            trainable=False,
            finetunable=False,
            prompt_types_supported=["point", "box"],
            refinement_supported=True,
        ),
        model_loader=SAM2ModelLoader(
            weights=SAM2_LARGE_WEIGHTS,
            config=SAM2_LARGE_CONFIG)
    )