# Inference Endpoint Unit Tests

## Overview

Comprehensive unit test suite for the `/inference` endpoint of the Prompted Segmentation Service. Tests verify functionality, error handling, and integration with registered models.

## Test Files

- **`tests/test_inference.py`**: Main test suite (9 tests)
- **`pytest.ini`**: Pytest configuration
- **`tests/__init__.py`**: Test package marker

## Test Results

✅ **All 9 tests pass**

```
tests/test_inference.py::TestInferenceEndpoint::test_inference_with_point_prompts PASSED
tests/test_inference.py::TestInferenceEndpoint::test_inference_with_box_prompt PASSED
tests/test_inference.py::TestInferenceEndpoint::test_inference_model_not_found PASSED
tests/test_inference.py::TestInferenceEndpoint::test_inference_model_inference_fails PASSED
tests/test_inference.py::TestInferenceEndpoint::test_all_registered_models PASSED
tests/test_inference.py::TestRegisteredModels::test_all_models_have_required_fields PASSED
tests/test_inference.py::TestRegisteredModels::test_model_tags_consistency PASSED
tests/test_inference.py::TestRegisteredModels::test_model_identifiers_are_unique PASSED
tests/test_inference.py::TestRegisteredModels::test_model_factories_valid PASSED
```

## Test Coverage

### TestInferenceEndpoint (5 tests)

#### 1. **test_inference_with_point_prompts**
- **Purpose**: Verify point-based prompts work correctly
- **Mocks**: `MODEL_REGISTRY.get_model()`, `_load_image_from_url()`
- **Validates**:
  - Successful 200 response
  - Proper model loading with `version_or_alias="latest"`
  - Model inference called with correct parameters
  - Result contains contour data

#### 2. **test_inference_with_box_prompt**
- **Purpose**: Verify box-based prompts work correctly
- **Mocks**: Same as above
- **Validates**:
  - Successful inference with bounding box prompts
  - Correct model registry key routing
  - Result structure with contour

#### 3. **test_inference_model_not_found**
- **Purpose**: Handle missing model gracefully
- **Mocks**: `MODEL_REGISTRY.get_model()` raises `KeyError`
- **Validates**:
  - 404 status code when model doesn't exist
  - Proper error response

#### 4. **test_inference_model_inference_fails**
- **Purpose**: Handle model inference errors gracefully
- **Mocks**: Model raises `RuntimeError` during inference
- **Validates**:
  - 500 status code when inference fails
  - Proper error handling and logging

#### 5. **test_all_registered_models**
- **Purpose**: Verify all registered models work with the endpoint
- **Data-driven**: Tests all 4 SAM2 variants (tiny, small, base-plus, large)
- **Validates**:
  - Endpoint works for every registered model
  - Consistent success response across all models

### TestRegisteredModels (4 tests)

#### 6. **test_all_models_have_required_fields**
- **Purpose**: Validate model registry config structure
- **Validates**:
  - All models have: `model_identifier`, `model_factory`, `desc`, `tags`
  - Non-empty values for each field
  - `model_factory` is callable

#### 7. **test_model_tags_consistency**
- **Purpose**: Ensure consistent metadata across models
- **Validates**: All models have these tag keys:
  - `task`, `pretrained`, `trainable`, `finetunable`
  - `model_size`, `inference_speed`, `accuracy_level`
  - `prompt_types_supported`, `refinement_supported`, `requires_gpu`

#### 8. **test_model_identifiers_are_unique**
- **Purpose**: Prevent model ID collisions
- **Validates**: No duplicate model identifiers in registry

#### 9. **test_model_factories_valid**
- **Purpose**: Verify model instantiation functions
- **Validates**: Each model factory is callable

## Endpoint Tested

```
POST /inference
Content-Type: application/json

{
  "image_url": "string (file path or URL)",
  "user_id": "string or integer",
  "model_registry_key": "sam2-1-tiny|sam2-1-small|sam2-1-base-plus|sam2-1-large",
  "prompts": {
    "point_prompts": [
      {"x": 0.0-1.0, "y": 0.0-1.0, "label": 0|1|True|False}
    ],
    "box_prompt": {
      "min_x": 0.0-1.0,
      "min_y": 0.0-1.0,
      "max_x": 0.0-1.0,
      "max_y": 0.0-1.0
    }
  },
  "previous_mask": null  // optional: BinaryMask with RLE encoding
}

Response (200 OK):
{
  "success": true,
  "message": "Successfully performed prompted segmentation.",
  "result": {
    "contours": [...],
    "confidence": 0.95,
    "added_by": "sam2-1-tiny"
  }
}
```

## Dependencies

Test dependencies added to `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.11.0",
]
```

Install with: `uv sync --extra dev`

## Running Tests

```bash
# Run all tests
python -m pytest tests/test_inference.py -v

# Run specific test class
python -m pytest tests/test_inference.py::TestInferenceEndpoint -v

# Run specific test
python -m pytest tests/test_inference.py::TestInferenceEndpoint::test_inference_with_point_prompts -v

# Run with verbose output and short traceback
python -m pytest tests/test_inference.py -v --tb=short

# Run all tests (project-wide)
python -m pytest
```

## Key Testing Patterns

1. **Mocking**: Uses `unittest.mock.patch()` to isolate endpoint logic from external dependencies
2. **Fixtures**: Reusable test data (prompts, requests, models)
3. **Data-driven**: Single test for all registered models
4. **Error handling**: Explicit tests for 404 and 500 scenarios
5. **Schema validation**: Tests confirm model metadata is complete and consistent

## Files Modified

### New Files
- `tests/test_inference.py` (260+ lines, 9 test cases)
- `tests/__init__.py`
- `pytest.ini`

### Updated Files
- `app/routes/inference.py`: Fixed to use correct schema fields (`image_url`, `user_id`) and added proper error handling
- `pyproject.toml`: Added dev dependencies

## Notes

- Tests use FastAPI's `TestClient` for synchronous endpoint testing
- Image loading is mocked to avoid file I/O in unit tests
- Model inference is mocked to avoid expensive GPU operations
- All tests are isolated and can run independently
- Tests validate both happy path and error scenarios


