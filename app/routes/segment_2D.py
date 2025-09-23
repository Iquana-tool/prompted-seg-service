from fastapi import APIRouter, UploadFile, File
from app.schemas.segment_2D import Prompted2DSegmentationRequest
from app import MODEL_CACHE

router = APIRouter("/segment", tags=["segment"])


@router.post("/image_with_prompts")
async def segment_image_with_prompts(request: Prompted2DSegmentationRequest):
    """Segment an image using 2D prompts.
    :param request: Request object.
    :return: Segmentation result.
    """
    model_identifier = request.model_identifier
    model = MODEL_CACHE.get(model_identifier)
    result = model.process_prompted_request(request)
    return {
        "success": True,
        "message": "Segmentation completed successfully.",
        "result": result
    }

