import os
from logging import getLogger

import numpy as np
import torch
from torchvision.transforms.functional import resize
from iquana_toolbox.schemas.prompts import Prompts
from transformers import Sam2Model, Sam2Processor
from paths import HUGGINGFACE_TOKEN
from models.base_models import Prompted2DBaseModel

logger = getLogger(__name__)


class SAMPrompted(Prompted2DBaseModel):
    def __init__(self, model_name_or_path, device='auto'):
        """
        Initialize the prompted SAM model using Transformers.
        """
        super().__init__()
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load processor and model from transformers
        self.processor = Sam2Processor.from_pretrained(
            model_name_or_path,
            token=HUGGINGFACE_TOKEN,
        )
        self.model = Sam2Model.from_pretrained(
            model_name_or_path,
            token=HUGGINGFACE_TOKEN,
        ).to(self.device)

    def process_prompted_request(self, image, prompts: Prompts, previous_mask=None):
        """
        Process the image with points or box prompts using the transformers pipeline.
        """
        # 1. Prepare Prompts
        point_coords = None # Image x Object x point x coords
        point_labels = None
        if prompts.point_prompts:
            point_coords = [[[[int(p.x * image.shape[1]), int(p.y * image.shape[0])] for p in prompts.point_prompts]]]
            point_labels = [[[p.label for p in prompts.point_prompts]]]

        box_coords = None
        if prompts.box_prompt:
            # Transformers SAM expects [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = prompts.box_prompt.xyxy
            xmin = int(xmin * image.shape[1])
            ymin = int(ymin * image.shape[0])
            xmax = int(xmax * image.shape[1])
            ymax = int(ymax * image.shape[0])
            box_coords = [[[xmin, ymin, xmax, ymax]]]

        # 2. Pre-process Image and Prompts
        # The processor handles resizing and normalization
        inputs = self.processor(
            [image],
            input_points=point_coords,
            input_labels=point_labels,
            input_boxes=box_coords,
            return_tensors="pt"
        ).to(self.device)
        _previous_mask = None
        if previous_mask is not None:
            _previous_mask = torch.from_numpy(previous_mask).unsqueeze(0).unsqueeze(0).to(self.device).float()
            _previous_mask = resize(_previous_mask, [256, 256])

        # 3. Inference
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                input_masks=_previous_mask,
                multimask_output=True,
            )

        # 4. Post-processing
        # Convert outputs (low-res masks) to original image size
        batches = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu()
        )

        # scores: [batch_size, 1, num_masks]
        scores = outputs.iou_scores.cpu().numpy().squeeze()
        best_index = np.argmax(scores) # take the mask with the best score

        # masks[0] is [1, 3, H, W] -> taking the first batch and usually the highest score mask
        masks = batches[0].squeeze()
        final_mask = masks[best_index].numpy().astype(np.uint8) * 255
        return [final_mask], [scores[best_index]]