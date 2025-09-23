import cv2

from models.base_models import Prompted2DBaseModel
from models.model_registry import ModelInfo
from app.schemas.segment_2D import Prompted2DSegmentationRequest
from models.prompts import Prompts
from util.image_loading import load_image_from_upload
import numpy as np
from sam2.build_sam import build_sam2 as build, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor


class SAM2Prompted(Prompted2DBaseModel):
    def __init__(self,  model_info: ModelInfo, device='auto'):
        """ Initialize the prompted SAM2 model.
            Args:
                path_to_weights (str): Path to the model weights file.
                path_to_config (str): Path to the model configuration file.
                device (str): The device to run the model on. Can be 'cpu', 'cuda', or 'auto'.
        """
        super().__init__(model_info, device)
        self.model_info = model_info
        self.device = device
        self.model = build(ckpt_path=model_info.weights_path,
                           config_file=model_info.configs_path,
                           device=self.device)
        self.prompt_predictor = SAM2ImagePredictor(self.model)
        self.set_image = None # To track the current image being processed

    def process_prompted_request(self, request: Prompted2DSegmentationRequest) -> tuple[np.ndarray, np.ndarray]:
        """ Process the prompted_segmentation request.
            Args:
                request (PromptedSegmentationRequest): The prompted_segmentation request containing the image and prompts.

            Returns:
                tuple: A tuple containing an array of masks and an array of predicted iou scores.
        """
        prompts = Prompts()
        prompts.from_segmentation_request(request)
        # This could be optimized in some way, where the image does not have to be uploaded if it stays the same, but
        # not sure how to do that with FastAPI
        image = load_image_from_upload(request.image)
        # This works because the third condition is only checked if the first two are false meaning
        # we have a set image and it has the same shape as the uploaded image
        if self.set_image is not None and image.shape != self.set_image or np.all(image == self.set_image):
            # Set the image only if it is different from the previous one
            self.set_image = image
            self.prompt_predictor.set_image(image)
        mask = None
        if request.previous_mask is not None:
            # Load the mask and resize it to 256x256
            mask = load_image_from_upload(request.previous_mask)
            mask = cv2.resize(mask, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        mask, scores, _ = self.prompt_predictor.predict(**prompts.to_SAM2_input(),
                                                        multimask_output=False,  # We only want one mask
                                                        mask_input=mask,  # The previous mask for refinement
                                                        normalize_coords=False)  # Because they are already normalized
        return mask, scores