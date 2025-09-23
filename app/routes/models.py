from fastapi import APIRouter
from app.util.model_registry import ModelRegistry, ModelInfo


api_router = APIRouter("/models", tags=["models"])


@api_router.get("/available")
async def list_models():
    available_models = [model_info.to_json() for model_info in MODEL_REGISTRY.values() if model_info.check_paths()]
    return {"available_models": available_models}
