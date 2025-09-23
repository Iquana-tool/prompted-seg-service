from models.model_registry import MODEL_REGISTRY


class ModelCache:
    """
    A simple in-memory cache for storing and retrieving models.
    """
    def __init__(self):
        self.cache = {}

    def get(self, model_identifier):
        """
        Retrieve a model from the cache by its name.

        Args:
            model_identifier (str): The name of the model to retrieve.
        Returns:
            The model if found, else loads the model into the cache and returns it.
        """
        if model_identifier in self.cache:
            return self.cache[model_identifier]
        else:
            model = MODEL_REGISTRY.load_model(model_identifier)
            self.cache[model_identifier] = model
            return model
