from typing import Literal

import numpy as np
import torch
from transformers import Sam3Processor, Sam3Model
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
        self.model: Sam3Model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.processor: Sam3Processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.set_image = None
        self.inference_state = None

    def process_prompted_request(self, image, prompts: Prompts, previous_mask=None):
        if self.set_image is None or image.shape != self.set_image.shape or np.any(image != self.set_image):
            logger.debug("Setting new image for SAM3 model.")
            # Set the image only if it is different from the previous one
            self.set_image = image
            self.inference_state = self.processor.set_image(image)
        if previous_mask is not None:
            inference_state = self.inference_state
        else:
            inference_state = self.processor.reset_all_prompts(self.inference_state)
        inference_state = self.processor(
            images=image,
            text=prompts.noun_prompt,
            input_boxes=[prompts.box_prompt.to_min_max_box()],
            input_boxes_labels=[[1]],
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**inference_state)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inference_state.get("original_sizes").tolist(),
        )[0]