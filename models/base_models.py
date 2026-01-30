from iquana_toolbox.schemas.prompts import Prompts
from abc import ABC, abstractmethod


class Prompted2DBaseModel(ABC):
    """ Abstract base class for 2D prompted segmentation models. """
    @abstractmethod
    def process_prompted_request(self, image, prompts: Prompts, previous_mask=None):
        """ Process a prompted segmentation request.
        :param image: The input image to be segmented.
        :param prompts: The prompts to guide the segmentation.
        :param previous_mask: An optional previous mask to provide context.
        :return: A tuple containing a mask and their corresponding quality score.
        """
        pass
