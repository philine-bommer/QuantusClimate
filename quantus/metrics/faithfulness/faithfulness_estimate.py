"""This module contains the implementation of the Faithfulness Estimate metric."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ..base import PerturbationMetric
from ...helpers import warn_func
from ...helpers import asserts
from ...helpers import utils
from ...helpers.model_interface import ModelInterface
from ...helpers.normalise_func import normalise_by_negative
from ...helpers.perturb_func import baseline_replacement_by_indices
from ...helpers.similarity_func import correlation_pearson


class FaithfulnessEstimate(PerturbationMetric):
    """
    Implementation of Faithfulness Estimate by Alvares-Melis at el., 2018a and 2018b.

    Computes the correlations of probability drops and the relevance scores on various points,
    showing the aggregate statistics.

    References:
        1) Alvarez-Melis, David, and Tommi S. Jaakkola. "Towards robust interpretability with self-explaining
        neural networks." arXiv preprint arXiv:1806.07538 (2018).
    """

    @asserts.attributes_check
    def __init__(
        self,
        similarity_func: Optional[Callable] = None,
        features_in_step: int = 1,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,
        perturb_baseline: str = "black",
        perturb_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable] = np.mean,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        similarity_func (callable): Similarity function applied to compare input and perturbed input.
            If None, the default value is used, default=correlation_spearman.
        features_in_step (integer): The size of the step, default=1.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
        perturb_func_kwargs (dict): Keyword arguments to be passed to perturb_func, default={}.
        return_aggregate (boolean): Indicates if an aggregated score should be computed over all instances.
        aggregate_func (callable): Callable that aggregates the scores given an evaluation call.
            default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative

        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices
        perturb_func = perturb_func

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs["perturb_baseline"] = perturb_baseline

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            normalise_func_kwargs=normalise_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        # Save metric-specific attributes.
        if similarity_func is None:
            similarity_func = correlation_pearson
        self.similarity_func = similarity_func
        self.features_in_step = features_in_step

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline' and similarity function "
                    "'similarity_func'"
                ),
                citation=(
                    "Alvarez-Melis, David, and Tommi S. Jaakkola. 'Towards robust interpretability"
                    " with self-explaining neural networks.' arXiv preprint arXiv:1806.07538 (2018)"
                ),
            )

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Optional[np.ndarray] = None,
        s_batch: Optional[np.ndarray] = None,
        custom_batch: Optional[np.ndarray] = None,
        channel_first: Optional[bool] = None,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict[str, Any]] = None,
        model_predict_kwargs: Optional[Dict[str, Any]] = None,
        softmax: bool = False,
        device: Optional[str] = None,
        **kwargs,
    ) -> List[float]:
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            custom_batch=custom_batch,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            softmax=softmax,
            device=device,
            **kwargs,
        )

    def evaluate_instance(
        self,
        i: int,
        model: ModelInterface,
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
        s: np.ndarray,
        c: Any,
        p: Any,
    ) -> float:

        # Flatten the attributions.
        a = a.flatten()

        # Get indices of sorted attributions (descending).
        a_indices = np.argsort(-a)

        # Predict on input.
        x_input = model.shape_input(x, x.shape, channel_first=True)
        y_pred = float(model.predict(x_input)[:, y])

        n_perturbations = len(range(0, len(a_indices), self.features_in_step))
        pred_deltas = [None for _ in range(n_perturbations)]
        att_sums = [None for _ in range(n_perturbations)]

        for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

            # Perturb input by indices of attributions.
            a_ix = a_indices[
                (self.features_in_step * i_ix) : (self.features_in_step * (i_ix + 1))
            ]
            x_perturbed = self.perturb_func(
                arr=x,
                indices=a_ix,
                indexed_axes=self.a_axes,
                **self.perturb_func_kwargs,
            )
            asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

            # Predict on perturbed input x.
            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            y_pred_perturb = float(model.predict(x_input)[:, y])
            pred_deltas[i_ix] = float(y_pred - y_pred_perturb)

            # Sum attributions.
            att_sums[i_ix] = np.sum(a[a_ix])

        similarity = self.similarity_func(a=att_sums, b=pred_deltas)
        return similarity

    def custom_preprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
        custom_batch: Optional[np.ndarray],
    ) -> Tuple[
        ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, Any
    ]:

        custom_preprocess_batch = [None for _ in x_batch]

        # Asserts.
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch.shape[2:],
        )

        return (
            model,
            x_batch,
            y_batch,
            a_batch,
            s_batch,
            custom_batch,
            custom_preprocess_batch,
        )
