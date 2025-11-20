from pydantic import BaseModel, field_validator, Field


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

    @property
    def width(self):
        return self.max_x - self.min_x

    @property
    def height(self):
        return self.max_y - self.min_y

    def to_xywh(self):
        return [self.min_x, self.min_y, self.width, self.height]

    def to_cxywh(self):
        cx = (self.min_x + self.max_x) / 2.0
        cy = (self.min_y + self.max_y) / 2.0
        return [cx, cy, self.width, self.height]

    def to_min_max_box(self):
        return [self.min_x, self.min_y, self.max_x, self.max_y]


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


class Prompts(BaseModel):
    """ Model for validating a prompted segmentation request. One request represents one object to be segmented."""
    point_prompts: list[PointPrompt] | None = Field(default=None,
                                             description="A list of point prompts. Each point prompt must have x, y, and label.")
    box_prompt: BoxPrompt | None = Field(default=None,
                                         description="A bounding box prompt. Must have min_x, min_y, max_x, and max_y.")
    noun_prompt: str | None = Field(default=None,
                                    description="A noun prompt. Describes the object to be segmented.")

    @property
    def point_coords(self) -> list[list[float]]:
        return [[pt.x, pt.y] for pt in self.point_prompts]

    @property
    def point_labels(self):
        return [pt.label for pt in self.point_prompts]

