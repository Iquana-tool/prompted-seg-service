import threading
from collections import OrderedDict
from models.model_registry import MODEL_REGISTRY

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
                model = MODEL_REGISTRY.load_model(model_identifier)
                if len(self.cache) >= self.size_limit:
                    self.cache.popitem(last=False)
                self.cache[model_identifier] = model
                return model

