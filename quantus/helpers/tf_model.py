"""This model creates the ModelInterface for Tensorflow."""
from typing import Any, Dict, Optional, Tuple

from keras.activations import linear, softmax
from keras.layers import Dense, Activation
from keras import Model
from keras import backend as K
from keras.models import clone_model
import numpy as np

from ..helpers.model_interface import ModelInterface
from ..helpers import utils


class TensorFlowModel(ModelInterface):
    """Interface for tensorflow models."""

    def __init__(
        self,
        model,
        channel_first: bool = True,
        softmax: bool = False,
        predict_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model=model,
            channel_first=channel_first,
            softmax=softmax,
            predict_kwargs=predict_kwargs,
        )

    def predict(self, x, **kwargs):
        """Predict on the given input."""

        # Use kwargs of predict call if specified, but don't overwrite object attribute
        predict_kwargs = {**self.predict_kwargs, **kwargs}

        output_act = self.model.layers[-1].activation
        target_act = softmax if self.softmax else linear

        if output_act == target_act:
            return self.model.predict(x)

        config = self.model.layers[-1].get_config()
        config["activation"] = target_act

        weights = self.model.layers[-1].get_weights()

        if isinstance(self.model.layers[-1], Activation):
            output_layer = self.model.layers[-2].output
            new_model = Model(inputs=[self.model.input], outputs=[output_layer])
        else:

            output_layer = Dense(**config)(self.model.layers[-2].output)
            new_model = Model(inputs=[self.model.input], outputs=[output_layer])
            new_model.layers[-1].set_weights(weights)

        return new_model.predict(x)

    def shape_input(
        self,
        x: np.ndarray,
        shape: Tuple[int, ...],
        channel_first: Optional[bool] = None,
        batched: bool = False,
    ):
        """
        Reshape input into model expected input.
        channel_first: Explicitely state if x is formatted channel first (optional).
        """
        if channel_first is None:
            channel_first = utils.infer_channel_first
        x = x.reshape(-1, *self.model.input_shape[1:])
        # Expand first dimension if this is just a single instance.
        if not batched:
            x = x.reshape(1, *shape)

        # Set channel order according to expected input of model.
        if self.channel_first:
            return utils.make_channel_first(x, channel_first)
        return utils.make_channel_last(x, channel_first)

    def get_model(self):
        """Get the original torch/tf model."""
        return self.model

    def state_dict(self):
        """Get a dictionary of the model's learnable parameters."""
        return self.model.get_weights()

    def load_state_dict(self, original_parameters):
        """Set model's learnable parameters."""
        self.model.set_weights(original_parameters)

    def get_random_layer_generator(self, order: str = "top_down", seed: int = 42):
        """
        In every iteration yields a copy of the model with one additional layer's parameters randomized.
        Set order to top_down for cascading randomization.
        Set order to independent for independent randomization.
        """
        original_parameters = self.state_dict()
        random_layer_model = clone_model(self.model)

        layers = [l for l in random_layer_model.layers if len(l.get_weights()) > 0]

        if order == "top_down":
            layers = layers[::-1]

        for layer in layers:
            if order == "independent":
                random_layer_model.set_weights(original_parameters)
            weights = layer.get_weights()
            np.random.seed(seed=seed + 1)
            layer.set_weights([np.random.permutation(w) for w in weights])
            yield layer.name, random_layer_model
