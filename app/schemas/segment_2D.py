from pydantic import BaseModel, Field
from app.schemas.prompts import Prompts

class Segment2DRequest(BaseModel):
    """ Model for 2D segmentation form data. """
    model_identifier: str = Field(..., title="Model identifier", description="Model identifier string. "
                                                                             "Used to select the model.")
    user_id: str = Field(..., title="User ID", description="Unique identifier for the user.")
    prompts: Prompts = Field(..., title="Prompts", description="Prompts for segmentation")
    previous_mask: list[list[bool]] | None = Field(None, title="Previous Mask",
                                                  description="Optional previous mask to provide context.")