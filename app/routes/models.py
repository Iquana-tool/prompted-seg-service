from fastapi import APIRouter
from models.model_registry import MODEL_REGISTRY


api_router = APIRouter("/models", tags=["models"])


@api_router.get("/available")
async def list_models():
    available_models = [model_info.to_json() for model_info in MODEL_REGISTRY.values() if model_info.check_paths()]
    return {"available_models": available_models}


@api_router.get("/load_model/{model_id}")
async def load_model(model_id: str):
