from models.model_registry import ModelInfo


class Prompted2DBaseModel:
    def __init__(self, model_info, device: str = "auto"):
        # Initialize model parameters here
        pass

    def set_image(self, image):
        # Placeholder for setting the image in the model. This enables caching the image.
        pass

    def segment(self, prompts: Prompts):
        # Placeholder for segmentation logic
        pass
