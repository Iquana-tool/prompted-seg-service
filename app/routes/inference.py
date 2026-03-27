from logging import getLogger

from fastapi import APIRouter, HTTPException
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.networking.http.services import PromptedSegmentationRequest

from app.state import MODEL_REGISTRY

logger = getLogger(__name__)
router = APIRouter()


@router.post("/inference", tags=["inference"])
async def inference(request: PromptedSegmentationRequest):
    """Segment an image using 2D prompts.
    
    :param request: PromptedSegmentationRequest containing image_url, user_id, model_identifier, prompts and an optional previous mask.
    :return: Segmentation result with contour.
    """
    # Load model from registry
    model = MODEL_REGISTRY.get_model_by_alias(request.model_registry_key, "latest")

    # Extract previous mask if provided
    previous_mask = request.previous_mask.mask if request.previous_mask else None

    # Run inference
    masks, scores = model.process_prompted_request(
        request.image,
        request.prompts,
        previous_mask,
    )

    # Convert masks and scores to proper format
    if not isinstance(masks, list):
        masks = [masks]
    if not isinstance(scores, list):
        scores = [scores]

    # Create contour result
    result = Contour.from_binary_mask(
        binary_mask=masks[0],
        only_return_biggest_contour=True,  # We only want one contour
        confidence=float(scores[0]),
        added_by=request.model_registry_key,
    )

    return {
        "success": True,
        "message": "Successfully performed prompted segmentation.",
        "result": result
    }
