from logging import getLogger
from typing import Any

import numpy as np
import torch
from torchvision.transforms.functional import resize
from transformers import Sam2Model, Sam2Processor

from iquana_toolbox.ai.base_classes import (
    PromptedSegmentationModel,
    PromptedSegmentationModelInfo,
)
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.networking.http.services import PromptedSegmentationRequest
from iquana_toolbox.schemas.prompts import Prompts
from iquana_service_core import register_model

from paths import HUGGINGFACE_TOKEN

logger = getLogger(__name__)


# One entry per registered SAM 2.1 variant. The model is fully self-describing:
# each instance builds its own model_info from the entry keyed by registry_key.
_VARIANTS: dict[str, dict] = {
    "sam2-1-tiny": {
        "checkpoint": "facebook/sam2.1-hiera-tiny",
        "name": "SAM 2.1 Tiny",
        "description": (
            "Segment Anything Model 2.1 - Tiny variant. The smallest and fastest model "
            "with lowest memory footprint. Suitable for real-time inference but with "
            "reduced accuracy. Supports point and box prompts."
        ),
        "model_size": "tiny",
        "inference_speed": "fastest",
        "accuracy_level": "low",
        "requires_gpu": "false",
    },
    "sam2-1-small": {
        "checkpoint": "facebook/sam2.1-hiera-small",
        "name": "SAM 2.1 Small",
        "description": (
            "Segment Anything Model 2.1 - Small variant. Provides a good balance between "
            "inference speed and segmentation accuracy. Ideal for production use cases "
            "requiring reasonable performance. Supports point and box prompts."
        ),
        "model_size": "small",
        "inference_speed": "fast",
        "accuracy_level": "medium",
        "requires_gpu": "true",
    },
    "sam2-1-base-plus": {
        "checkpoint": "facebook/sam2.1-hiera-base-plus",
        "name": "SAM 2.1 Base+",
        "description": (
            "Segment Anything Model 2.1 - Base+ variant. Larger model with improved "
            "accuracy compared to the small variant. Good choice for accuracy-critical "
            "applications. Supports point and box prompts with refinement capabilities."
        ),
        "model_size": "base-plus",
        "inference_speed": "medium",
        "accuracy_level": "high",
        "requires_gpu": "true",
    },
    "sam2-1-large": {
        "checkpoint": "facebook/sam2.1-hiera-large",
        "name": "SAM 2.1 Large",
        "description": (
            "Segment Anything Model 2.1 - Large variant. The largest and most accurate "
            "SAM2 model. Best segmentation quality but requires more VRAM and slower "
            "inference. Recommended for offline and accuracy-critical workflows. Supports "
            "point and box prompts."
        ),
        "model_size": "large",
        "inference_speed": "slowest",
        "accuracy_level": "highest",
        "requires_gpu": "true",
    },
}


class SAMPrompted(PromptedSegmentationModel):
    """Prompted segmentation backed by SAM 2.1 (HuggingFace Transformers).

    One class, four registered variants (tiny/small/base-plus/large). The variant
    is selected by ``registry_key``; the model derives its full ``model_info`` from
    the matching entry in :data:`_VARIANTS`.
    """

    def __init__(self, registry_key: str, device: str = "auto"):
        cfg = _VARIANTS[registry_key]
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_info = PromptedSegmentationModelInfo(
            registry_key=registry_key,
            name=cfg["name"],
            description=cfg["description"],
            usage_tip="Provide point and/or box prompts; supports iterative refinement using the previous mask.",
            tags={
                "task": "prompted-segmentation",
                "status": "ready",
                "pretrained": "true",
                "finetunable": "false",
                "model_size": cfg["model_size"],
                "inference_speed": cfg["inference_speed"],
                "accuracy_level": cfg["accuracy_level"],
                "requires_gpu": cfg["requires_gpu"],
            },
            status="ready",
            trainable=False,
            prompt_types_supported=["point", "box"],
            refinement_supported=True,
        )

        self.processor = Sam2Processor.from_pretrained(cfg["checkpoint"], token=HUGGINGFACE_TOKEN)
        self.model = Sam2Model.from_pretrained(cfg["checkpoint"], token=HUGGINGFACE_TOKEN).to(self.device)

    def predict(
        self,
        context: Any,
        model_input: list[PromptedSegmentationRequest],
        params: dict[str, Any] | None = None,
    ) -> list[Contour]:
        """Segment one image per request using its 2D prompts."""
        # MLflow's PyFuncModel.predict(data) passes ``data`` straight through; be
        # tolerant of either a single request or a list of them.
        requests = model_input if isinstance(model_input, list) else [model_input]
        contours: list[Contour] = []
        for request in requests:
            previous_mask = request.previous_mask.mask if request.previous_mask else None
            mask, score = self._segment(request.image, request.prompts, previous_mask)
            contours.append(
                Contour.from_binary_mask(
                    binary_mask=mask,
                    only_return_biggest_contour=True,  # one prompted object per request
                    confidence=score,
                    added_by=request.model_registry_key,
                )
            )
        return contours

    def train(self, request, **kwargs):
        raise NotImplementedError("SAMPrompted is a pretrained model and is not trainable.")

    def _segment(self, image, prompts: Prompts, previous_mask=None) -> tuple[np.ndarray, float]:
        """Run SAM2 on a single image with point/box prompts; return (mask, score)."""
        # 1. Prepare prompts (coords are normalised in the request; scale to pixels).
        point_coords = None  # Image x Object x point x coords
        point_labels = None
        if prompts.point_prompts:
            point_coords = [[[[int(p.x * image.shape[1]), int(p.y * image.shape[0])] for p in prompts.point_prompts]]]
            point_labels = [[[p.label for p in prompts.point_prompts]]]

        box_coords = None
        if prompts.box_prompt:
            xmin, ymin, xmax, ymax = prompts.box_prompt.xyxy
            xmin = int(xmin * image.shape[1])
            ymin = int(ymin * image.shape[0])
            xmax = int(xmax * image.shape[1])
            ymax = int(ymax * image.shape[0])
            box_coords = [[[xmin, ymin, xmax, ymax]]]

        # 2. Pre-process image + prompts (the processor handles resize/normalisation).
        inputs = self.processor(
            [image],
            input_points=point_coords,
            input_labels=point_labels,
            input_boxes=box_coords,
            return_tensors="pt",
        ).to(self.device)

        _previous_mask = None
        if previous_mask is not None:
            _previous_mask = torch.from_numpy(previous_mask).unsqueeze(0).unsqueeze(0).to(self.device).float()
            _previous_mask = resize(_previous_mask, [256, 256])

        # 3. Inference.
        with torch.no_grad():
            outputs = self.model(**inputs, input_masks=_previous_mask, multimask_output=True)

        # 4. Post-process: upscale to original size and pick the best-scoring mask.
        batches = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())
        scores = outputs.iou_scores.cpu().numpy().squeeze()
        best_index = int(np.argmax(scores))

        masks = batches[0].squeeze()
        final_mask = masks[best_index].numpy().astype(np.uint8) * 255
        return final_mask, float(scores[best_index])


# --- Registered variants: each a zero-arg factory the catalog auto-discovers. ---
@register_model
def sam2_1_tiny() -> SAMPrompted:
    return SAMPrompted("sam2-1-tiny")


@register_model
def sam2_1_small() -> SAMPrompted:
    return SAMPrompted("sam2-1-small")


@register_model
def sam2_1_base_plus() -> SAMPrompted:
    return SAMPrompted("sam2-1-base-plus")


@register_model
def sam2_1_large() -> SAMPrompted:
    return SAMPrompted("sam2-1-large")
