import cv2
import torch.cuda
from logging import getLogger
from models.base_models import Prompted2DBaseModel
from models.model_registry import ModelLoader
from app.schemas.prompts import Prompts
from models.prompts import Prompts
from util.image_loading import load_image_from_upload
import numpy as np
from sam2.build_sam import build_sam2 as build, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
import os


logger = getLogger(__name__)


class SAM2ModelLoader(ModelLoader):
    def __init__(self, weights: str, config: str, device='auto'):
        """
        Initialize the SAM2 model loader.
        :param weights: Path to the model weights.
        :param config: Path to the model config file.
        :param device: Device to run the model on. 'auto' will use GPU if available, otherwise CPU.
        """
        super().__init__(SAM2Prompted, weights=weights, config=config, device=device)
        self.weights = weights
        self.config = config

    def is_loadable(self) -> bool:
        """ Check if the model is loadable by verifying the existence of the weights and config files.
        :return: True if the model is loadable, False otherwise.
        """
        weights_exist = os.path.isfile(self.weights)
        config_exist = os.path.isfile(self.config)
        if not weights_exist:
            print(f"Model weights not found at {self.weights}")
        if not config_exist:
            print(f"Model config not found at {self.config}")
        return weights_exist and config_exist


class SAM2Prompted(Prompted2DBaseModel):
    def __init__(self,  weights, config, device='auto'):
        """
        Initialize the prompted SAM2 model.
        :param weights: Path to the model weights.
        :param config: Path to the model config file.
        :param device: Device to run the model on. 'auto' will use GPU if available, otherwise CPU.
        """
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build(ckpt_path=weights,
                           config_file=config,
                           device=self.device)
        self.prompt_predictor = SAM2ImagePredictor(self.model)
        self.set_image = None # To track the current image being processed
        self.set_image_name = None

    def process_prompted_request(self, image, prompts: Prompts, previous_mask=None):
        # This works because the third condition is only checked if the first two are false meaning
        # we have a set image and it has the same shape as the uploaded image
        if self.set_image is None and image.shape != self.set_image or np.all(image == self.set_image):
            logger.debug("Setting new image for SAM2 model.")
            # Set the image only if it is different from the previous one
            self.set_image = image
            self.prompt_predictor.set_image(image)
        mask = None
        if previous_mask is not None:
            # resize it to 256x256
            mask = cv2.resize(previous_mask, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        mask, scores, _ = self.prompt_predictor.predict(**prompts.to_SAM2_input(),
                                                        multimask_output=False,  # We only want one mask
                                                        mask_input=mask,  # The previous mask for refinement
                                                        normalize_coords=False)  # Because they are already normalized
        return mask, scores