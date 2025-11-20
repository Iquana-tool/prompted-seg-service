from typing import Literal

import numpy as np
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from logging import getLogger
from models.base_models import Prompted2DBaseModel
from models.model_registry import ModelLoader
from app.schemas.prompts import Prompts


logger = getLogger(__name__)


class SAM3ModelLoader(ModelLoader):
    def __init__(self, weights: str, config: str, device: str):
        """
        Initialize the SAM3 model loader.
        :param weights: Path to the model weights.
        :param config: Path to the model config file.
        :param device: Device to run the model on. 'auto' will use GPU if available, otherwise CPU.
        """
        super().__init__(weights)
        self.weights = weights
        self.config = config
        self.device = device

    def is_loadable(self):
        return True


class SAM3Prompted(Prompted2DBaseModel):
    def __init__(self, checkpoint_path: str, device: Literal["cpu", "cuda", "auto"] = "auto"):
        """
        Initialize the prompted SAM3 model.
        :param weights: Path to the model weights.
        :param config: Path to the model config file.
        :param device: Device to run the model on. 'auto' will use GPU if available, otherwise CPU.
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.model = build_sam3_image_model(checkpoint_path=self.checkpoint_path, device=self.device)
        self.processor = Sam3Processor(self.model)
        self.set_image = None
        self.inference_state = None

    def process_prompted_request(self, image, prompts: Prompts, previous_mask=None):
        if self.set_image is None or image.shape != self.set_image.shape or np.any(image != self.set_image):
            logger.debug("Setting new image for SAM3 model.")
            # Set the image only if it is different from the previous one
            self.set_image = image
            self.processor.set_image(image)
        # continue to use inference state if previous mask is provided, else reset all prompts.
        if previous_mask is not None:
            inference_state = self.inference_state
        else:
            inference_state = self.processor.reset_all_prompts(self.inference_state)
        if prompts.noun_prompt is not None:
            inference_state = self.processor.set_text_prompt(
                prompts.noun_prompt,
                inference_state
            )
        if prompts.box_prompt is not None:
            inference_state = self.processor.add_geometric_prompt(
                state=inference_state,
                box=prompts.box_prompt.to_cxywh(),
                label=True,
            )
        self.inference_state = inference_state
        return