from app.schemas.segment_2D import Prompted2DSegmentationRequest
from abc import ABC, abstractmethod


class Prompted2DBaseModel(ABC):
    """ Abstract base class for 2D prompted segmentation models. """
    @abstractmethod
    def process_prompted_request(self, request: Prompted2DSegmentationRequest):
        # Process the prompted segmentation request
        pass
