import logging

from iquana_service_core import create_service_app

from app.state import MODEL_REGISTRY
from app.routes.inference import router as inference_router

logger = logging.getLogger(__name__)


def create_app():
    return create_service_app(
        title="IQUANA Prompted Segmentation API",
        description="FastAPI backend for interactive coral prompted segmentation",
        task="prompted-segmentation",
        registry=MODEL_REGISTRY,
        models_package="models",
        inference_routers=[inference_router],
        hf_login=False,
    )
