import os


class ModelInfo:
    def __init__(self, identifier_str: str, name: str, description: str, weights_path: str, configs_path: str,
                 tags: list[str]):
        self.identifier_str = identifier_str
        self.name = name
        self.description = description
        self.weights_path = weights_path
        self.configs_path = configs_path
        self.tags = tags

    def to_json(self):
        return {
            "identifier_str": self.identifier_str,
            "name": self.name,
            "description": self.description,
            "weights_path": self.weights_path,
            "configs_path": self.configs_path,
            "tags": self.tags,
        }

    def check_paths(self):
        if not os.path.exists(self.weights_path) and not os.path.exists(self.configs_path):
            raise FileNotFoundError(
                f"Both weights and configs paths do not exist: {self.weights_path}, {self.configs_path}")
        elif not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Weights path does not exist: {self.weights_path}")
        elif not os.path.exists(self.configs_path):
            raise FileNotFoundError(f"Configs path does not exist: {self.configs_path}")
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

    def list_models(self):
        return list(self.models.values())


MODEL_REGISTRY = ModelRegistry()
MODEL_REGISTRY.register_model(ModelInfo(
    identifier_str="sam2_tiny",
    name="SAM2 Tiny",
    description="Segment Anything Model 2 - Tiny version",
    weights_path="models/sam2_tiny/sam2_tiny.pth",
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