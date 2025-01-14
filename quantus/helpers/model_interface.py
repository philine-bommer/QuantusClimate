from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np


class ModelInterface(ABC):
    """Interface for torch and tensorflow models."""

    def __init__(
        self,
        model,
        channel_first: bool = True,
        softmax: bool = False,
        predict_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.channel_first = channel_first
        self.softmax = softmax

        if predict_kwargs is None:
            self.predict_kwargs = {}
        else:
            self.predict_kwargs = predict_kwargs

    @abstractmethod
    def predict(self, x, **kwargs):
        """Predict on the given input."""
        raise NotImplementedError

    @abstractmethod
    def shape_input(
        self, x: np.array, shape: Tuple[int, ...], channel_first: Optional[bool] = None
    ):
        """
        Reshape input into model expected input.
        channel_first: Explicitely state if x is formatted channel first (optional).
        """
        raise NotImplementedError

    @abstractmethod
    def get_model(self):
        """Get the original torch/tf model."""
        raise NotImplementedError

    @abstractmethod
    def state_dict(self):
        """Get a dictionary of the model's learnable parameters."""
        raise NotImplementedError

    @abstractmethod
    def get_random_layer_generator(self):
        """
        In every iteration yields a copy of the model with one additional layer's parameters randomized.
        Set order to top_down for cascading randomization.
        Set order to independent for independent randomization.
        """
        raise NotImplementedError
