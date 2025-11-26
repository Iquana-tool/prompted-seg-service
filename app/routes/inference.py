from logging import getLogger

import numpy as np
from fastapi import Response

from app.routes import session_router
from app.schemas.segment_2D import Segment2DRequest
from app.state import MODEL_REGISTRY, MODEL_CACHE, IMAGE_CACHE

logger = getLogger(__name__)


@session_router.post("/segment_image_with_prompts")
async def segment_image_with_prompts(
        request: Segment2DRequest
    ):
    """Segment an image using 2D prompts.
    :param request: Segment2DRequest containing user_uid, model_identifier, prompts and an optional previous mask.
    :return: Segmentation result.
    """
    try:
        model = MODEL_CACHE.get(request.user_id)
    except KeyError:
        logger.info(f"Cache miss for user {request.user_id}. Loading model.")
        model = MODEL_REGISTRY.load_model(request.model_identifier)
        MODEL_CACHE.put(request.user_id, model)
    if request.user_id not in IMAGE_CACHE:
        return {"success": False, "message": "No image uploaded for this user. Please upload an image first."}
    image = IMAGE_CACHE.get(request.user_id)
    prompts = request.prompts
    if request.previous_mask is not None:
        previous_mask = np.array(request.previous_mask, dtype=np.uint8)
    else:
        previous_mask = None
    masks, scores = model.process_prompted_request(image, prompts, previous_mask)
    mask = masks[0].astype(np.uint8)
    score = scores[0]
    # Convert the mask to raw bytes
    mask_bytes = mask.tobytes()

    # Return the raw bytes with metadata in headers
    return Response(
        content=mask_bytes,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": "attachment; filename=mask.bin",
            "X-Mask-Shape": f"{mask.shape[0]},{mask.shape[1]}",  # e.g., "256,256"
            "X-Mask-Dtype": str(mask.dtype),  # e.g., "uint8"
            "X-Score": str(score)  # Optional: Include the score
        }
    )
