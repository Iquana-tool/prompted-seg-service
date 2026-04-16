import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from app import create_app
from iquana_toolbox.schemas.networking.http.services import PromptedSegmentationRequest
from iquana_toolbox.schemas.prompts import PointPrompt, BoxPrompt, Prompts
from models.register_models import MODEL_REGISTRY_CONFIG


@pytest.fixture
def app():
    """Create test FastAPI app."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create a mock model that implements Prompted2DBaseModel interface."""
    model = Mock()
    # Simulate successful segmentation: return masks and scores
    model.process_prompted_request = Mock(
        return_value=(
            [np.ones((512, 512), dtype=np.uint8) * 255],  # Binary mask
            [0.95]  # Confidence score
        )
    )
    return model


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def point_prompts():
    """Create sample point prompts."""
    return Prompts(
        point_prompts=[
            PointPrompt(x=0.5, y=0.5, label=1),
            PointPrompt(x=0.3, y=0.3, label=0),
        ]
    )


@pytest.fixture
def box_prompts():
    """Create sample box prompts."""
    return Prompts(
        box_prompt=BoxPrompt(min_x=0.2, min_y=0.2, max_x=0.8, max_y=0.8)
    )


@pytest.fixture
def segmentation_request_with_points(sample_image, point_prompts):
    """Create a valid PromptedSegmentationRequest with point prompts."""
    return PromptedSegmentationRequest(
        image_url="http://example.com/image.jpg",
        user_id="test_user_123",
        model_registry_key="sam2-1-tiny",
        prompts=point_prompts,
        previous_mask=None,
    )


@pytest.fixture
def segmentation_request_with_box(sample_image, box_prompts):
    """Create a valid PromptedSegmentationRequest with box prompt."""
    return PromptedSegmentationRequest(
        image_url="http://example.com/image.jpg",
        user_id="test_user_456",
        model_registry_key="sam2-1-small",
        prompts=box_prompts,
        previous_mask=None,
    )


class TestInferenceEndpoint:
    """Test suite for the /inference endpoint."""

    @patch("app.routes.inference._load_image_from_url")
    @patch("app.routes.inference.MODEL_REGISTRY")
    def test_inference_with_point_prompts(self, mock_registry, mock_load_image, client, mock_model, segmentation_request_with_points):
        """Test inference endpoint with point prompts."""
        # Setup mocks
        mock_load_image.return_value = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mock_registry.get_model.return_value = mock_model

        # Make request
        response = client.post(
            "/inference",
            json={
                "image_url": segmentation_request_with_points.image_url,
                "user_id": segmentation_request_with_points.user_id,
                "model_registry_key": "sam2-1-tiny",
                "prompts": segmentation_request_with_points.prompts.model_dump(),
            }
        )

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Successfully performed prompted segmentation" in data["message"]
        assert data["result"] is not None
        
        # Verify model was called with correct parameters
        mock_registry.get_model.assert_called_once_with("sam2-1-tiny", version_or_alias="latest")
        mock_model.process_prompted_request.assert_called_once()

    @patch("app.routes.inference._load_image_from_url")
    @patch("app.routes.inference.MODEL_REGISTRY")
    def test_inference_with_box_prompt(self, mock_registry, mock_load_image, client, mock_model, segmentation_request_with_box):
        """Test inference endpoint with box prompt."""
        # Setup mocks
        mock_load_image.return_value = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mock_registry.get_model.return_value = mock_model

        # Make request
        response = client.post(
            "/inference",
            json={
                "image_url": segmentation_request_with_box.image_url,
                "user_id": segmentation_request_with_box.user_id,
                "model_registry_key": "sam2-1-small",
                "prompts": segmentation_request_with_box.prompts.model_dump(),
            }
        )

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "result" in data
        
        # Verify model was called
        mock_registry.get_model.assert_called_once_with("sam2-1-small", version_or_alias="latest")


    @patch("app.routes.inference._load_image_from_url")
    @patch("app.routes.inference.MODEL_REGISTRY")
    def test_inference_model_not_found(self, mock_registry, mock_load_image, client, segmentation_request_with_points):
        """Test inference endpoint when model is not found."""
        # Setup mocks
        mock_load_image.return_value = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mock_registry.get_model.side_effect = KeyError("Model not found")

        # Make request
        response = client.post(
            "/inference",
            json={
                "image_url": segmentation_request_with_points.image_url,
                "user_id": segmentation_request_with_points.user_id,
                "model_registry_key": "nonexistent-model",
                "prompts": segmentation_request_with_points.prompts.model_dump(),
            }
        )

        # Assertions
        assert response.status_code == 404

    @patch("app.routes.inference._load_image_from_url")
    @patch("app.routes.inference.MODEL_REGISTRY")
    def test_inference_model_inference_fails(self, mock_registry, mock_load_image, client, segmentation_request_with_points):
        """Test inference endpoint when model inference fails."""
        # Setup mocks
        mock_load_image.return_value = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mock_model = Mock()
        mock_model.process_prompted_request.side_effect = RuntimeError("GPU out of memory")
        mock_registry.get_model.return_value = mock_model

        # Make request
        response = client.post(
            "/inference",
            json={
                "image_url": segmentation_request_with_points.image_url,
                "user_id": segmentation_request_with_points.user_id,
                "model_registry_key": "sam2-1-tiny",
                "prompts": segmentation_request_with_points.prompts.model_dump(),
            }
        )

        # Assertions
        assert response.status_code == 500

    @patch("app.routes.inference._load_image_from_url")
    @patch("app.routes.inference.MODEL_REGISTRY")
    def test_all_registered_models(self, mock_registry, mock_load_image, client):
        """Test inference with all registered models."""
        mock_load_image.return_value = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mock_model = Mock()
        mock_model.process_prompted_request = Mock(
            return_value=(
                [np.ones((512, 512), dtype=np.uint8) * 255],
                [0.95]
            )
        )
        mock_registry.get_model.return_value = mock_model
        
        sample_image_url = "http://example.com/image.jpg"
        user_id = "test_user"
        point_prompts = Prompts(point_prompts=[PointPrompt(x=0.5, y=0.5, label=1)])
        
        for config in MODEL_REGISTRY_CONFIG:
            model_id = config["model_identifier"]
            
            response = client.post(
                "/inference",
                json={
                    "image_url": sample_image_url,
                    "user_id": user_id,
                    "model_registry_key": model_id,
                    "prompts": point_prompts.model_dump(),
                }
            )
            
            assert response.status_code == 200, f"Failed for model {model_id}: {response.text}"
            assert response.json()["success"] is True


class TestRegisteredModels:
    """Test that all models in MODEL_REGISTRY_CONFIG are properly defined."""

    def test_all_models_have_required_fields(self):
        """Test that all registered models have required configuration fields."""
        required_fields = {"model_identifier", "model_factory", "desc", "tags"}
        
        for config in MODEL_REGISTRY_CONFIG:
            assert required_fields.issubset(config.keys()), \
                f"Model config missing required fields: {config}"
            
            # Validate model_identifier is non-empty
            assert config["model_identifier"], "model_identifier must be non-empty"
            
            # Validate model_factory is callable
            assert callable(config["model_factory"]), \
                f"model_factory for {config['model_identifier']} must be callable"
            
            # Validate description is non-empty
            assert config["desc"], f"Description missing for {config['model_identifier']}"
            
            # Validate tags
            assert isinstance(config["tags"], dict), \
                f"Tags for {config['model_identifier']} must be a dict"
            assert config["tags"], "Tags dict must be non-empty"

    def test_model_tags_consistency(self):
        """Test that all models have consistent tag keys."""
        expected_tag_keys = {
            "task",
            "pretrained",
            "trainable",
            "finetunable",
            "model_size",
            "inference_speed",
            "accuracy_level",
            "prompt_types_supported",
            "refinement_supported",
            "requires_gpu",
        }
        
        for config in MODEL_REGISTRY_CONFIG:
            tags = config["tags"]
            missing_keys = expected_tag_keys - set(tags.keys())
            assert not missing_keys, \
                f"Model {config['model_identifier']} missing tag keys: {missing_keys}"

    def test_model_identifiers_are_unique(self):
        """Test that all model identifiers are unique."""
        identifiers = [config["model_identifier"] for config in MODEL_REGISTRY_CONFIG]
        assert len(identifiers) == len(set(identifiers)), \
            "Model identifiers must be unique"

    def test_model_factories_valid(self):
        """Test that model factories are properly defined."""
        for config in MODEL_REGISTRY_CONFIG:
            model_id = config["model_identifier"]
            factory = config["model_factory"]
            
            # Verify factory is a callable lambda
            assert callable(factory), \
                f"model_factory for {model_id} is not callable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])





