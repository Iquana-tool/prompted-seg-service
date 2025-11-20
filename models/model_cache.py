import threading
from collections import OrderedDict
from logging import getLogger


logger = getLogger(__name__)


class ModelCache:
    def __init__(self, size_limit=3):
        """
        Cache to hold a limited number of loaded models. Thread-safe.
        :param size_limit: Maximum number of models to cache. Default is 3.
        """
        self.size_limit = size_limit
        self.cache = OrderedDict()
        self.lock = threading.Lock()  # Lock for thread safety

    def get(self, model_identifier):
        with self.lock:  # Acquire lock
            if model_identifier in self.cache:
                model = self.cache.pop(model_identifier)
                self.cache[model_identifier] = model
                return model
            else:
                raise KeyError(model_identifier)

    def put(self, model_identifier, model):
        with self.lock:  # Acquire lock
            if model_identifier in self.cache:
                self.cache.pop(model_identifier)
            elif len(self.cache) >= self.size_limit:
                self.cache.popitem(last=False)
            self.cache[model_identifier] = model

    def check_if_loaded(self, model_identifier):
        with self.lock:  # Acquire lock
            return model_identifier in self.cache

    def set_image(self, model_identifier, image):
        with self.lock:
            if not model_identifier in self.cache:
                logger.error(f"Trying to set an image for {model_identifier} in cache, but model is not loaded.")
            else:
                model = self.cache[model_identifier]
                try:
                    model.set_image(image)
                except NotImplementedError:
                    logger.warning(f"Trying to set an image for {model_identifier} in cache, "
                                   f"but model does not implement this function. Skipping.")
