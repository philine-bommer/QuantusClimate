"""This module contains the implementation of the Faithfulness Correlation metric."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import pdb

from ..base import PerturbationMetric
from ...helpers import warn_func
from ...helpers import asserts
from ...helpers import utils
from ...helpers.model_interface import ModelInterface
from ...helpers.normalise_func import normalise_by_negative
from ...helpers.perturb_func import baseline_replacement_by_indices
from ...helpers.similarity_func import correlation_pearson


class FaithfulnessCorrelation(PerturbationMetric):
    """
    Implementation of faithfulness correlation by Bhatt et al., 2020.

    The Faithfulness Correlation metric intend to capture an explanation's relative faithfulness
    (or 'fidelity') with respect to the model behaviour.

    Faithfulness correlation scores shows to what extent the predicted logits of each modified test point and
    the average explanation attribution for only the subset of features are (linearly) correlated, taking the
    average over multiple runs and test samples. The metric returns one float per input-attribution pair that
    ranges between -1 and 1, where higher scores are better.

    For each test sample, |S| features are randomly selected and replace them with baseline values (zero baseline
    or average of set). Thereafter, Pearson’s correlation coefficient between the predicted logits of each modified
    test point and the average explanation attribution for only the subset of features is calculated. Results is
    average over multiple runs and several test samples.

    References:
        1) Bhatt, Umang, Adrian Weller, and José MF Moura. "Evaluating and aggregating feature-based model
        explanations." arXiv preprint arXiv:2005.00631 (2020).

    """

    @asserts.attributes_check
    def __init__(
        self,
        similarity_func: Optional[Callable] = None,
        nr_runs: int = 100,
        subset_size: int = 224,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,
        perturb_baseline: str = "black",
        perturb_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = True,
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
            If None, the default value is used, default=correlation_pearson.
        nr_runs (integer): The number of runs (for each input and explanation pair), default=100.
        subset_size (integer): The size of subset, default=224.
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
            normalise_func_kwargs=normalise_func_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
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
        self.nr_runs = nr_runs
        self.subset_size = subset_size

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline', size of subset |S| 'subset_size'"
                    " and the number of runs (for each input and explanation pair) "
                    "'nr_runs'"
                ),
                citation=(
                    "Bhatt, Umang, Adrian Weller, and José MF Moura. 'Evaluating and aggregating "
                    "feature-based model explanations.' arXiv preprint arXiv:2005.00631 (2020)"
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
            softmax=softmax,
            device=device,
            model_predict_kwargs=model_predict_kwargs,
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

        # Predict on input.
        x_input = model.shape_input(x, x.shape, channel_first=True)
        try:
            y_pred = float(model.predict(x_input.flatten()[np.newaxis,:])[:, y])
        except:
            y_pred = float(model.predict(x_input[0,:,:,:,np.newaxis])[:, y])

        pred_deltas = []
        att_sums = []
        a_ix_set = p[0]
        y_pred_perturb_set= p[1]


        # For each test data point, execute a couple of runs.
        for i_ix in range(self.nr_runs):

            a_ix = a_ix_set[i_ix]
            y_pred_perturb = y_pred_perturb_set[i_ix][y]
            # y_pred_perturb = y_pred_perturb_set[i_ix][:, y]
            pred_deltas.append(float(y_pred - y_pred_perturb))

            # Sum attributions of the random subset.
            att_sums.append(np.sum(a[a_ix]))

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
        # Asserts.
        asserts.assert_value_smaller_than_input_size(
            x=x_batch, value=self.subset_size, value_name="subset_size"
        )
        iterator = tqdm(
            enumerate(
                zip(
                    x_batch,
                    y_batch,
                    a_batch,
                )
            ),
            total=len(x_batch),
            disable=not self.display_progressbar,
            desc=f"Evaluating {self.__class__.__name__}",
        )
        perturbed_samples = np.zeros((x_batch.shape[0], self.nr_runs,
                                      *model.shape_input(x_batch[0], x_batch[0].shape, channel_first=True).shape),
                                     dtype=float)
        a_ix_set = []
        for ix, (x, y, a) in iterator:
            for i_ix in range(self.nr_runs):
                a = a.flatten()
                # Randomly mask by subset size.
                a_ix = np.random.choice(a.shape[0], self.subset_size, replace=False)
                a_ix_set.append(a_ix)
                x_perturbed = self.perturb_func(
                    arr=x,
                    indices=a_ix,
                    indexed_axes=self.a_axes,
                    **self.perturb_func_kwargs,
                )
                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)

                # Append to samples.
                perturbed_samples[ix, i_ix, :] = x_input

        # Predict on perturbed input x.
        try:
            y_pert_samples = model.predict(perturbed_samples.reshape(x_batch.shape[0] * self.nr_runs,
                                             *x_batch[0].flatten().shape)).astype(float)
        except:
            y_pert_samples = model.predict(perturbed_samples.reshape(x_batch.shape[0] * self.nr_runs,
                                                                     *x_batch[0].shape[1:],1)).astype(float)
        # y_pert_samples = model.predict(perturbed_samples.reshape(x_batch.shape[0] * self.nr_runs,
        #                                                          *model.shape_input(x_batch[0], x_batch[0].shape,
        #                                                                             channel_first=True).shape)).astype(
        #     float)
        a_ix_set = np.array(a_ix_set).reshape(x_batch.shape[0],self.nr_runs,np.array(a_ix_set).shape[1])
        y_pert_samples = y_pert_samples.reshape(x_batch.shape[0],self.nr_runs,*y_pert_samples.shape[1:])
        custom_preprocess_batch = []
        for h in range(x_batch.shape[0]):
            custom_preprocess_batch.append([a_ix_set[h], y_pert_samples[h]])

        return (
            model,
            x_batch,
            y_batch,
            a_batch,
            s_batch,
            custom_batch,
            custom_preprocess_batch,
        )
