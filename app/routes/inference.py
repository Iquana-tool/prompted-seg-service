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
    # Return all candidates so the backend can pick the best one (e.g. discard a
    # candidate that just re-segments the focussed parent and keep the next best).
    contours = model.predict([request])
    return {
        "success": True,
        "message": f"Successfully performed prompted segmentation. Found {len(contours)} candidate(s).",
        "result": contours,
    }
