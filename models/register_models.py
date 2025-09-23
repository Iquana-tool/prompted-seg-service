from models.model_registry import ModelRegistry, ModelInfo, ModelLoader


def register_models(model_registry: ModelRegistry):
    """ This function registers all models in the MODEL_REGISTRY. You can extend it to add custom models. """
    model_registry.register_model(
        model_info=ModelInfo(
            identifier_str="sam2_tiny",
            name="SAM2 Tiny",
            description="Segment Anything Model 2 - Tiny version",
            tags=["Sam2", "Tiny", "Fast", "General Purpose"], ),
        model_loader=ModelLoader(
            loader_function=None,
            weights_path="models/sam2_tiny/sam2_tiny.pth",
            configs_path="models/sam2_tiny/sam2_tiny.yaml")
    )