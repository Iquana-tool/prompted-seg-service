from fastapi import APIRouter
from app.schemas.segment_2D import Prompted2DSegmentationRequest
from app.state import MODEL_REGISTRY, MODEL_CACHE
from logging import getLogger


router = APIRouter(prefix="/segment", tags=["segment"])
logger = getLogger(__name__)


@router.post("/image_with_prompts")
async def segment_image_with_prompts(request: Prompted2DSegmentationRequest):
    """Segment an image using 2D prompts.
    :param request: Request object.
    :return: Segmentation result.
    """
    model_identifier = request.model_identifier
    try:
        model = MODEL_CACHE.get(model_identifier)
    except KeyError:
        logger.info(f"Cache miss for model {model_identifier}. Loading model.")
        model = MODEL_REGISTRY.load_model(model_identifier)
        MODEL_CACHE.put(model_identifier, model)
    result = model.process_prompted_request(request)
    return {
        "success": True,
        "message": "Segmentation completed successfully.",
        "result": result
    }

