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
    try:
        # Load model from registry
        model = MODEL_REGISTRY.get_model(request.model_registry_key, version_or_alias="latest")
        
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
    
    except HTTPException:
        raise
    except KeyError as e:
        logger.error(f"Model not found: {request.model_registry_key}")
        raise HTTPException(status_code=404, detail=f"Model '{request.model_registry_key}' not found")
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
