from pydantic import BaseModel, Field
from app.schemas.prompts import Prompts

class Segment2DFormData(BaseModel):
    """ Model for 2D segmentation form data. """
    model_identifier: str = Field(..., title="Model identifier", description="Model identifier string. "
                                                                             "Used to select the model.")
    prompts: Prompts = Field(..., title="Prompts", description="Prompts for segmentation")