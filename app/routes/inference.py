from logging import getLogger

from fastapi import APIRouter
from iquana_toolbox.schemas.networking.http.services import PromptedSegmentationRequest

from app.state import MODEL_REGISTRY

logger = getLogger(__name__)
router = APIRouter()


@router.post("/inference", tags=["inference"])
async def inference(request: PromptedSegmentationRequest):
    """Segment an image using 2D prompts.

    :param request: PromptedSegmentationRequest with image_url, user_id,
        model_registry_key, prompts and an optional previous mask.
    :return: Segmentation result with the contour.
    """
    model = MODEL_REGISTRY.get_model_by_alias(request.model_registry_key, "latest")
    # model is an MLflow PyFuncModel; predict(data) forwards to the model's
    # predict(context, model_input=data, params), which returns a list[Contour].
    contours = model.predict([request])
    return {
        "success": True,
        "message": "Successfully performed prompted segmentation.",
        "result": contours[0] if contours else None,
    }
