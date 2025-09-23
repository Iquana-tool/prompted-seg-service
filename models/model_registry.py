import os


class ModelInfo:
    def __init__(self,
                 identifier_str: str,
                 name: str,
                 description: str,
                 weights_path: str,
                 configs_path: str,
                 tags: list[str],
                 supported_prompt_types: list[str] = ["point", "box"],
                 supports_refinement: bool = False):
        """Class to hold information about a segmentation model.
        Args:
            identifier_str: str - Unique identifier for the model.
            name: str - Human-readable name for the model.
            description: str - Brief description of the model.
            weights_path: str - Path to the model weights file.
            configs_path: str - Path to the model configuration file.
            tags: list[str] - List of tags associated with the model.
            supported_prompt_types: list[str] - List of supported prompt types (default: ["point", "box"]).
                Possible values are "point", "box", "circle", "polygon". Can be extended in the future.
            supports_refinement: bool - Whether the model supports refinement meaning that you can pass a previous mask
                to be refined with new annotations (default: False).

        """
        self.identifier_str = identifier_str
        self.name = name
        self.description = description
        self.weights_path = weights_path
        self.configs_path = configs_path
        self.tags = tags
        self.supported_prompt_types = supported_prompt_types
        self.supports_refinement = supports_refinement

    def to_json(self):
        """Convert the model information to a JSON-serializable dictionary."""
        return {
            "identifier_str": self.identifier_str,
            "name": self.name,
            "description": self.description,
            "weights_path": self.weights_path,
            "configs_path": self.configs_path,
            "tags": self.tags,
            "supported_prompt_types": self.supported_prompt_types,
            "supports_refinement": self.supports_refinement
        }

    def check_paths(self, raise_error: bool = False) -> bool:
        """Check if the weights and configs paths exist. Only checks configs path if it is not None meaning that the
            model requires a config file.
        Args:
            raise_error: bool - Whether to raise an error if paths do not exist (default: False).
        Returns:
            bool - True if both paths exist, False otherwise.
        """
        config_necessary = self.configs_path is not None
        if not os.path.exists(self.weights_path) and config_necessary and not os.path.exists(self.configs_path):
            if raise_error:
                raise FileNotFoundError(
                    f"Both weights and configs paths do not exist: {self.weights_path}, {self.configs_path}")
            return False
        elif not os.path.exists(self.weights_path):
            if raise_error:
                raise FileNotFoundError(f"Weights path does not exist: {self.weights_path}")
            return False
        elif config_necessary and not os.path.exists(self.configs_path):
            if raise_error:
                raise FileNotFoundError(f"Configs path does not exist: {self.configs_path}")
            return False
        else:
            return True


class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register_model(self, model_info: ModelInfo):
        if model_info.identifier_str in self.models:
            raise ValueError(f"Model with identifier {model_info.identifier_str} is already registered.")
        self.models[model_info.identifier_str] = model_info

    def get_model(self, identifier_str: str) -> ModelInfo:
        if identifier_str not in self.models:
            raise KeyError(f"Model with identifier {identifier_str} is not registered.")
        return self.models[identifier_str]

    def check_model(self, identifier_str: str) -> bool:
        model = self.get_model(identifier_str)
        return model.check_paths()

    def list_models(self, only_available: bool = True) -> list[ModelInfo]:
        if only_available:
            return [model for model in self.models.values() if model.check_paths()]
        return list(self.models.values())


MODEL_REGISTRY = ModelRegistry()
MODEL_REGISTRY.register_model(ModelInfo(
    identifier_str="sam2_tiny",
    name="SAM2 Tiny",
    description="Segment Anything Model 2 - Tiny version",
    weights_path="./sam2_tiny/sam2_tiny.pth",
    configs_path="models/sam2_tiny/sam2_tiny.yaml",
    tags=["Sam2", "Tiny", "Fast", "General Purpose"]
))
MODEL_REGISTRY.register_model(ModelInfo(
    identifier_str="sam2_base",
    name="SAM2 Base",
    description="Segment Anything Model 2 - Base version",
    weights_path="models/sam2_base/sam2_base.pth",
    configs_path="models/sam2_base/sam2_base.yaml",
    tags=["Sam2", "Base", "Balanced", "General Purpose"]
))
MODEL_REGISTRY.register_model(ModelInfo(
    identifier_str="sam2_large",
    name="SAM2 Large",
    description="Segment Anything Model 2 - Large version",
    weights_path="models/sam2_large/sam2_large.pth",
    configs_path="models/sam2_large/sam2_large.yaml",
    tags=["Sam2", "Large", "Accurate", "General Purpose"]
))