from logging import getLogger
from logging import DEBUG
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir
from models.model_cache import ModelCache
from models.model_registry import ModelRegistry
from models.register_models import register_models
from contextlib import asynccontextmanager
logger = getLogger(__name__)
logger.setLevel(DEBUG)

MODEL_CACHE = ModelCache()
MODEL_REGISTRY = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    logger.debug("Starting up the Prompted Segmentation Service")
    logger.debug("Registering models in the MODEL_REGISTRY")
    register_models(MODEL_REGISTRY)
    yield
    # Shutdown code
    logger.debug("Shutting down the Prompted Segmentation Service")


def create_app():
    logger.debug("Creating FastAPI application")
    # Load environment variables
    load_dotenv()

    # Get allowed origins from environment variable
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

    app = FastAPI(
        title="Coral Segmentation API",
        lifespan=lifespan,
        description="FastAPI backend for interactive coral prompted_segmentation",
        version="0.1.0",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the routers


    return app