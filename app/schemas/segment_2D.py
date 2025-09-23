from pydantic import BaseModel, field_validator, Field
from fastapi import UploadFile
from app import MODEL_REGISTRY
from app.schemas.prompts import PointPrompt, BoxPrompt, PolygonPrompt, CirclePrompt


class Prompted2DSegmentationRequest(BaseModel):
    """ Model for validating a prompted segmentation request. One request represents one object to be segmented."""
    image: UploadFile = Field(..., description="The image to be segmented as a file upload.")
    previous_mask: UploadFile | None = Field(default=None,
                                            description="An optional previous mask to provide context. Must be a binary "
                                                        "mask image file. This does not work with every model.")
    model_identifier: str = Field(description="The model identifier. Must be one of the registered available models.")
    point_prompts: list[PointPrompt] | None = Field(default=None,
                                             description="A list of point prompts. Each point prompt must have x, y, and label.")
    box_prompt: BoxPrompt | None = Field(default=None,
                                         description="A bounding box prompt. Must have min_x, min_y, max_x, and max_y.")
    circle_prompt: CirclePrompt | None = Field(default=None,
                                               description="A circle prompt. Must have center_x, center_y, and radius.")
    polygon_prompt: PolygonPrompt | None = Field(default=None,
                                                 description="A polygon prompt. Must have a list of vertices.")

    @field_validator('model_identifier', mode='before')
    def validate_model_identifier(cls, value, values):
        model_info = MODEL_REGISTRY.get_model_info(value)
        given_prompt_types = []
        if values.get('point_prompts'):
            if 'point' in model_info.supported_prompt_types:
                return value
            given_prompt_types.append('point')
        elif values.get('box_prompt'):
            if 'box' in model_info.supported_prompt_types:
                return value
            given_prompt_types.append('box')
        elif values.get('circle_prompt'):
            if 'circle' in model_info.supported_prompt_types:
                return value
            given_prompt_types.append('circle')
        elif values.get('polygon_prompt'):
            if 'polygon' in model_info.supported_prompt_types:
                return value
            given_prompt_types.append('polygon')
        raise ValueError(f"Model expects at least one of the following prompt types: {', '.join(model_info.supported_prompt_types)}\n"
                         f"But got: {', '.join(given_prompt_types) if given_prompt_types else 'no prompts'}")
