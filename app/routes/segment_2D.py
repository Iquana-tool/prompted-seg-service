from logging import getLogger

import numpy as np
from fastapi import APIRouter, UploadFile, File, Response

from app.schemas.prompts import Prompts as PromptsRequest
from app.state import MODEL_REGISTRY, MODEL_CACHE, IMAGE_CACHE
from models.prompts import Prompts
from util.image_loading import load_image_from_upload

router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])
logger = getLogger(__name__)


@router.post("/open_image")
async def open_image(user: str, image: UploadFile = File(...)):
    """Endpoint to upload an image and an optional previous mask.
    This is a placeholder endpoint to demonstrate file upload functionality.
    In a real application, you might want to store the image and return an ID or URL.
    """
    image = load_image_from_upload(image)
    IMAGE_CACHE.set(user, image)
    return {
        "success": True,
        "message": f"Image uploaded successfully for user {user}.",
        "image_shape": image.shape
    }


@router.get("/focus_crop/min_x={min_x}&min_y={min_y}&max_x={max_x}&max_y={max_y}&user_uid={user_uid}")
async def focus_crop(
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        user_uid: str,
    ):
    """Crop the uploaded image to the specified bounding box and update the cached image.
    :param min_x: Minimum x-coordinate of the bounding box.
    :param min_y: Minimum y-coordinate of the bounding box.
    :param max_x: Maximum x-coordinate of the bounding box.
    :param max_y: Maximum y-coordinate of the bounding box.
    :param user_uid: Unique identifier for the user to retrieve their cached image.
    :return: Success message with new image shape or error message.
    """
    if user_uid not in IMAGE_CACHE:
        return {"success": False, "message": "No image uploaded for this user. Please upload an image first."}
    IMAGE_CACHE.set_focused_crop(user_uid, min_x, min_y, max_x, max_y)
    return {
        "success": True,
        "message": f"Image cropped successfully for user {user_uid}.",
    }


@router.get("/unfocus_crop/user_uid={user_uid}")
async def unfocus_crop(user_uid: str):
    """Revert the cached image to the original uploaded image.
    :param user_uid: Unique identifier for the user to retrieve their cached image.
    :return: Success message with new image shape or error message.
    """
    if user_uid not in IMAGE_CACHE:
        return {"success": False, "message": "No image uploaded for this user. Please upload an image first."}
    IMAGE_CACHE.set_uncropped(user_uid)
    return {
        "success": True,
        "message": f"Image reverted to original successfully for user {user_uid}.",
    }


@router.get("/close_image/user_uid={user_uid}")
async def close_image(user_uid: str):
    """Clear the cached image for the specified user.
    :param user_uid: Unique identifier for the user to clear their cached image.
    :return: Success message or error message.
    """
    if user_uid not in IMAGE_CACHE:
        return {"success": False, "message": "No image uploaded for this user. Please upload an image first."}
    IMAGE_CACHE.delete(user_uid)
    return {
        "success": True,
        "message": f"Image cache cleared successfully for user {user_uid}.",
    }


@router.post("/segment_image_with_prompts")
async def segment_image_with_prompts(
        prompts_request: PromptsRequest,
        model_identifier: str,
        user_uid: str,
    ):
    """Segment an image using 2D prompts.
    :param image: The image to be segmented.
    :param previous_mask: An optional previous mask to provide context. Must be a binary mask image file. This does not work with every model.
    :param form_data: Form data containing model identifier and prompts.
    :return: Segmentation result.
    """
    try:
        model = MODEL_CACHE.get(model_identifier)
    except KeyError:
        logger.info(f"Cache miss for model {model_identifier}. Loading model.")
        model = MODEL_REGISTRY.load_model(model_identifier)
        MODEL_CACHE.put(model_identifier, model)
    if user_uid not in IMAGE_CACHE:
        return {"success": False, "message": "No image uploaded for this user. Please upload an image first."}
    image = IMAGE_CACHE.get(user_uid)
    prompts = Prompts()
    for point in prompts_request.point_prompts:
        prompts.add_point_annotation(point.x, point.y, point.label)
    if prompts_request.box_prompt:
        box_prompt = prompts_request.box_prompt
        prompts.add_box_annotation(box_prompt.min_x,
                                   box_prompt.min_y,
                                   box_prompt.max_x,
                                   box_prompt.max_y)
    # prompts.add_point_annotation(0.5, 0.5, 1) # Example point prompt in the center
    masks, scores = model.process_prompted_request(image, prompts, None)
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

