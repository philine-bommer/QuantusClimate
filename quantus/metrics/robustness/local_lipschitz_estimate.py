"""This module contains the implementation of the Local Lipschitz Estimate metric."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import pdb


from ..base import PerturbationMetric
from ...helpers import asserts
from ...helpers import warn_func
from ...helpers.model_interface import ModelInterface
from ...helpers.normalise_func import normalise_by_negative
from ...helpers.perturb_func import gaussian_noise
from ...helpers.similarity_func import lipschitz_constant, distance_euclidean


class LocalLipschitzEstimate(PerturbationMetric):
    """
    Implementation of the Local Lipschitz Estimate (or Stability) test by Alvarez-Melis et al., 2018a, 2018b.

    This tests asks how consistent are the explanations for similar/neighboring examples.
    The test denotes a (weaker) empirical notion of stability based on discrete,
    finite-sample neighborhoods i.e., argmax_(||f(x) - f(x')||_2 / ||x - x'||_2)
    where f(x) is the explanation for input x and x' is the perturbed input.

    References:
        1) Alvarez-Melis, David, and Tommi S. Jaakkola. "On the robustness of interpretability methods."
        arXiv preprint arXiv:1806.08049 (2018).

        2) Alvarez-Melis, David, and Tommi S. Jaakkola. "Towards robust interpretability with self-explaining
        neural networks." arXiv preprint arXiv:1806.07538 (2018).
    """

    @asserts.attributes_check
    def __init__(
        self,
        similarity_func: Optional[Callable] = None,
        norm_numerator: Optional[Callable] = None,
        norm_denominator: Optional[Callable] = None,
        nr_samples: int = 200,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,
        perturb_mean: float = 0.0,
        perturb_std: float = 0.1,
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
            If None, the default value is used, default=lipschitz_constant.
        norm_numerator (callable): Function for norm calculations on the numerator.
            If None, the default value is used, default=distance_euclidean.
        norm_denominator (callable): Function for norm calculations on the denominator.
            If None, the default value is used, default=distance_euclidean.
        nr_samples (integer): The number of samples iterated, default=200.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=gaussian_noise.
        perturb_std (float): The amount of noise added, default=0.1.
        perturb_mean (float): The mean of noise added, default=0.0.
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
            perturb_func = gaussian_noise

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs["perturb_mean"] = perturb_mean
        perturb_func_kwargs["perturb_std"] = perturb_std

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
            similarity_func = lipschitz_constant
        self.similarity_func = similarity_func

        if norm_numerator is None:
            norm_numerator = distance_euclidean
        self.norm_numerator = norm_numerator

        if norm_denominator is None:
            norm_denominator = distance_euclidean
        self.norm_denominator = norm_denominator

        self.nr_samples = nr_samples

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "amount of noise added 'perturb_std', the number of samples iterated "
                    "over 'nr_samples', the function to perturb the input 'perturb_func',"
                    " the similarity metric 'similarity_func' as well as norm "
                    "calculations on the numerator and denominator of the lipschitz "
                    "equation i.e., 'norm_numerator' and 'norm_denominator'"
                ),
                citation=(
                    "Alvarez-Melis, David, and Tommi S. Jaakkola. 'On the robustness of "
                    "interpretability methods.' arXiv preprint arXiv:1806.08049 (2018). and "
                    "Alvarez-Melis, David, and Tommi S. Jaakkola. 'Towards robust interpretability"
                    " with self-explaining neural networks.' arXiv preprint "
                    "arXiv:1806.07538 (2018)"
                ),
            )
            warn_func.warn_noise_zero(noise=perturb_std)

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
        softmax: bool = True,
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

        a_perturbed_batch = p[0]
        x_perturbed_batch = p[1]
        similarity_max = 0.0
        for j in range(self.nr_samples):

            # Perturb input.
            a_perturbed = a_perturbed_batch[j]
            x_perturbed = x_perturbed_batch[j]

            if self.normalise:
                a_perturbed = self.normalise_func(
                    a_perturbed,
                    **self.normalise_func_kwargs,
                )

            if self.abs:
                a_perturbed = np.abs(a_perturbed)

                # Measure similarity.
            similarity = self.similarity_func(
                a=a.flatten(),
                b=a_perturbed.flatten(),
                c=x.flatten(),
                d=x_perturbed.flatten()
            )
            similarity_max = max(similarity, similarity_max)

        return similarity_max

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

        # Create array to save intermediary results.
        perturbed_samples = np.zeros((x_batch.shape[0], self.nr_samples,
                                      *model.shape_input(x_batch[0], x_batch[0].shape, channel_first=True).shape),
                                     dtype=float)

        # Create progress bar if desired.
        iterator = tqdm(
            enumerate(
                zip(
                    x_batch,
                    y_batch,
                )
            ),
            total=len(x_batch),
            disable=not self.display_progressbar,
            desc=f"Preparing perturbations for {self.__class__.__name__}",
        )

        for ix, (x, y) in iterator:
            for j in range(self.nr_samples):

                # Perturb input.
                x_perturbed = self.perturb_func(
                    arr=x,
                    indices=np.arange(0, x.size),
                    indexed_axes=np.arange(0, x.ndim),
                    **self.perturb_func_kwargs,
                )
                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Append to samples.
                perturbed_samples[ix, j, :] = x_input

        # Generate explanation based on perturbed input x.
        a_perturbed_all = self.explain_func(
            model=model.get_model(),
            inputs=perturbed_samples.reshape(x_batch.shape[0] * self.nr_samples,
                                             *model.shape_input(x_batch[0], x_batch[0].shape,
                                                                channel_first=True).shape),
            targets=np.tile(y_batch, self.nr_samples),
            **self.explain_func_kwargs,
        )
        ap = a_perturbed_all.reshape((x_batch.shape[0], self.nr_samples,
                                                           *model.shape_input(x_batch[0], x_batch[0].shape,
                                                                              channel_first=True).shape))
        custom_preprocess_batch = []
        for h in range(x_batch.shape[0]):
            custom_preprocess_batch.append([ap[h], perturbed_samples[h]])

        # Additional explain_func assert, as the one in prepare() won't be
        # executed when a_batch != None.
        asserts.assert_explain_func(explain_func=self.explain_func)

        return (
            model,
            x_batch,
            y_batch,
            a_batch,
            s_batch,
            custom_batch,
            custom_preprocess_batch,
        )
