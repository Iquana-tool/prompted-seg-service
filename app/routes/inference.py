from logging import getLogger

from fastapi import APIRouter
from iquana_toolbox.schemas.contours import Contour
from iquana_toolbox.schemas.service_requests import PromptedSegmentationRequest

from app.state import MODEL_REGISTRY, MODEL_CACHE, IMAGE_CACHE

logger = getLogger(__name__)
session_router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])
router = APIRouter()


@session_router.post("/run")
async def segment_image_with_prompts(
        request: PromptedSegmentationRequest
    ):
    """Segment an image using 2D prompts.
    :param request: Segment2DRequest containing user_uid, model_identifier, prompts and an optional previous mask.
    :return: Segmentation result.
    """
    try:
        model = MODEL_CACHE.get(request.user_id)
    except KeyError:
        logger.info(f"Cache miss for user {request.user_id}. Loading model.")
        model = MODEL_REGISTRY.load_model(request.model_registry_key)
        MODEL_CACHE.put(request.user_id, request.model_registry_key, model)
    if request.user_id not in IMAGE_CACHE:
        IMAGE_CACHE.set(request.user_id, request.image)
    image = IMAGE_CACHE.get(request.user_id)
    masks, scores = model.process_prompted_request(
        image,
        request.prompts,
        request.previous_mask.mask if request.previous_mask else None,
    )
    return {
        "success": True,
        "message": "Successfully performed prompted segmentation.",
        "result": Contour.from_binary_mask(
            binary_mask=masks[0],
            only_return_biggest_contour=True,  # We only want one contour
            confidence=scores[0],
            added_by=request.model_registry_key,
        )
    }
