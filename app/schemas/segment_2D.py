import os

import cv2
from pydantic import BaseModel, field_validator, Field
from fastapi import File, UploadFile
from models.model_registry import MODEL_REGISTRY


class PointPrompt(BaseModel):
    """ Model for validating a point annotation. """
    x: float = Field(description="X coordinate of the point in relative coordinates. Must be between 0 and 1.")
    y: float = Field(description="Y coordinate of the point in relative coordinates. Must be between 0 and 1.")
    label: bool = Field(
        description="Label of the point. 0 for background (negative annotation), 1 for foreground (positive annotation).")

    @field_validator('label')
    def validate_label(cls, value):
        if value not in [True, False]:
            raise ValueError("Label must be 0 (background) or 1 (foreground).")
        return value

    @field_validator('x', 'y')
    def validate_coordinates(cls, value):
        if not (0 <= value <= 1):
            raise ValueError("Coordinates must be between 0 and 1.")
        return value


class BoxPrompt(BaseModel):
    """ Model for validating a bounding box annotation. """
    min_x: float = Field(
        description="Minimum x coordinate of the box in relative coordinates. Must be between 0 and 1.")
    min_y: float = Field(
        description="Minimum y coordinate of the box in relative coordinates. Must be between 0 and 1.")
    max_x: float = Field(
        description="Maximum x coordinate of the box in relative coordinates. Must be between 0 and 1.")
    max_y: float = Field(
        description="Maximum y coordinate of the box in relative coordinates. Must be between 0 and 1.")

    @field_validator('min_x', 'min_y', 'max_x', 'max_y')
    def validate_coordinates(cls, value):
        if not (0 <= value <= 1):
            raise ValueError("Box coordinates must be between 0 and 1.")
        return value


class PolygonPrompt(BaseModel):
    """ Model for validating a polygon annotation. """
    vertices: list[list[float]] = Field(default_factory=list[list],
                                        description="The vertices of the polygon. A list of (x, y) coordinates.")

    @field_validator('vertices')
    def validate_vertices(cls, value):
        if len(value) < 3:
            raise ValueError("Polygon must have at least 3 vertices.")
        for vertex in value:
            if len(vertex) != 2:
                raise ValueError("Each vertex must have exactly 2 coordinates.")
            if not all(0 <= coord <= 1 for coord in vertex):
                raise ValueError("Coordinates must be between 0 and 1.")
        return value


class CirclePrompt(BaseModel):
    """ Model for validating a circle annotation. """
    center_x: float = Field(
        description="X coordinate of the circle center in relative coordinates. Must be between 0 and 1.")
    center_y: float = Field(
        description="Y coordinate of the circle center in relative coordinates. Must be between 0 and 1.")
    radius: float = Field(
        description="Radius of the circle. This is a relative value, e.g. 0.1 means 10% of the image width/height.")

    @field_validator('center_x', 'center_y', 'radius')
    def validate_coordinates(cls, value):
        if not (0 <= value <= 1):
            raise ValueError("Coordinates must be between 0 and 1.")
        return value


class Prompted2DSegmentationRequest(BaseModel):
    """ Model for validating a prompted segmentation request. One request represents one object to be segmented."""
    image: UploadFile = Field(..., description="The image to be segmented as a file upload.")
    previous_mask: UploadFile | None = Field(default=None,
                                            description="An optional previous mask to provide context. Must be a binary "
                                                        "mask image file. This does not work with every model.")
    model_identifier: str = Field(description="The model identifier. Must be one of the registered available models.")
    point_prompts: list[PointPrompt] = Field(default_factory=None,
                                             description="A list of point prompts. Each point prompt must have x, y, and label.")
    box_prompt: BoxPrompt | None = Field(default=None,
                                         description="A bounding box prompt. Must have min_x, min_y, max_x, and max_y.")
    circle_prompt: CirclePrompt | None = Field(default=None,
                                               description="A circle prompt. Must have center_x, center_y, and radius.")
    polygon_prompt: PolygonPrompt | None = Field(default=None,
                                                 description="A polygon prompt. Must have a list of vertices.")

    @field_validator('model_identifier', mode='before')
    def validate_model_identifier(cls, value, values):
        if not MODEL_REGISTRY.check_model(value):
            raise ValueError(f"Model with identifier {value} is not available.")
        else:
            model_info = MODEL_REGISTRY.get_model(value)
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
