"""This modules holds a collection of perturbation functions i.e., ways to perturb an input or an explanation."""
import copy
import random
import warnings
from typing import Any, Sequence, Tuple, Union

import cv2
import numpy as np
import scipy

from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

from .utils import (
    get_baseline_value,
    blur_at_indices,
    expand_indices,
    get_leftover_shape,
    offset_coordinates,
)


def baseline_replacement_by_indices(
    arr: np.array,
    indices: Union[int, Sequence[int], Tuple[np.array]],
    indexed_axes: Sequence[int],
    perturb_baseline: Union[float, int, str, np.array],
    **kwargs,
) -> np.array:
    """
    Replace indices in an array by a given baseline.

    Parameters
    ----------
        arr: array to be perturbed
        indices: array-like, with a subset shape of arr
        indexed_axes: dimensions of arr that are indexed. These need to be consecutive,
                      and either include the first or last dimension of array.
        perturb_baseline: baseline value to replace arr at indices with
    """
    indices = expand_indices(arr, indices, indexed_axes)
    baseline_shape = get_leftover_shape(arr, indexed_axes)

    arr_perturbed = copy.copy(arr)

    # Get the baseline value.
    baseline_value = get_baseline_value(
        value=perturb_baseline, arr=arr, return_shape=tuple(baseline_shape), **kwargs
    )

    # Perturb the array.
    arr_perturbed[indices] = np.expand_dims(baseline_value, axis=tuple(indexed_axes))

    return arr_perturbed


def baseline_replacement_by_shift(
    arr: np.array,
    indices: Union[int, Sequence[int], Tuple[np.array]],
    indexed_axes: Sequence[int],
    input_shift: Union[float, int, str, np.array],
    **kwargs,
) -> np.array:
    """
    Shift values at indices in an image.

    Parameters
    ----------
        arr: array to be perturbed
        indices: array-like, with a subset shape of arr
        indexed_axes: axes of arr that are indexed. These need to be consecutive,
                      and either include the first or last dimension of array.
        input_shift: value to shift arr at indices with
    """
    indices = expand_indices(arr, indices, indexed_axes)
    baseline_shape = get_leftover_shape(arr, indexed_axes)

    arr_perturbed = copy.copy(arr)

    # Get the baseline value.
    baseline_value = get_baseline_value(
        value=input_shift, arr=arr, return_shape=tuple(baseline_shape), **kwargs
    )

    # Shift the input.
    arr_shifted = copy.copy(arr_perturbed)
    arr_shifted = np.add(
        arr_shifted,
        np.full(
            shape=arr_shifted.shape,
            fill_value=np.expand_dims(baseline_value, axis=tuple(indexed_axes)),
            dtype=float,
        ),
    )

    arr_perturbed[indices] = arr_shifted[indices]
    return arr_perturbed


def baseline_replacement_by_blur(
    arr: np.array,
    indices: Tuple[np.array],
    indexed_axes: Sequence[int],
    blur_kernel_size: Union[int, Sequence[int]] = 15,
    **kwargs,
) -> np.array:
    """
    Replace array at indices by a blurred version.

    Parameters
    ----------
        Blur is performed via convolution.
        arr: array to be perturbed
        indices: array-like, with a subset shape of arr
        indexed_axes: axes of arr that are indexed. These need to be consecutive,
                      and either include the first or last dimension of array.
        blur_kernel_size: controls the kernel-size of that convolution (Default is 15).
    """

    indices = expand_indices(arr, indices, indexed_axes)

    # Expand blur_kernel_size.
    if isinstance(blur_kernel_size, int):
        blur_kernel_size = [blur_kernel_size for _ in indexed_axes]

    assert len(blur_kernel_size) == len(indexed_axes)

    # Create a kernel and expand dimensions to arr.ndim.
    kernel = np.ones(blur_kernel_size, dtype=arr.dtype)
    kernel *= 1.0 / np.prod(blur_kernel_size)

    # Blur the array at indicies 8since otherwise n-d convolution can be quite computationally expensive),
    # else it is equal to arr.
    arr_perturbed = blur_at_indices(arr, kernel, indices, indexed_axes)

    return arr_perturbed


def gaussian_noise(
    arr: np.array,
    indices: Union[int, Sequence[int], Tuple[np.array]],
    indexed_axes: Sequence[int],
    perturb_mean: float = 0.0,
    perturb_std: float = 0.01,
    **kwargs,
) -> np.array:
    """
    Add gaussian noise to the input at indices.

    Parameters
    ----------
        arr: array to be perturbed
        indices: array-like, with a subset shape of arr
        indexed_axes: axes of arr that are indexed. These need to be consecutive,
                      and either include the first or last dimension of array.
        perturb_mean: Mean for gaussian
        perturb_std: Std for gaussian
    """

    indices = expand_indices(arr, indices, indexed_axes)
    noise = np.random.normal(loc=perturb_mean, scale=perturb_std, size=arr.shape)

    arr_perturbed = copy.copy(arr)
    arr_perturbed[indices] = (arr_perturbed + noise)[indices]

    return arr_perturbed


def uniform_noise(
    arr: np.array,
    indices: Union[int, Sequence[int], Tuple[np.array]],
    indexed_axes: Sequence[int],
    lower_bound: float = 0.02,
    upper_bound: Union[None, float] = None,
    **kwargs,
) -> np.array:
    """
    Add noise to the input at indices as sampled uniformly random from [-lower_bound, lower_bound].
    if upper_bound is None, and [lower_bound, upper_bound] otherwise.

    Parameters
    ----------
        arr: array to be perturbed
        indices: array-like, with a subset shape of arr
        indexed_axes: axes of arr that are indexed. These need to be consecutive,
                      and either include the first or last dimension of array.
        lower_bound: lower bound for uniform sampling
        upper_bound: upper bound for uniform sampling
    """

    indices = expand_indices(arr, indices, indexed_axes)

    if upper_bound is None:
        noise = np.random.uniform(low=-lower_bound, high=lower_bound, size=arr.shape)
    else:
        assert upper_bound > lower_bound, (
            "Parameter 'upper_bound' needs to be larger than 'lower_bound', "
            "but {} <= {}".format(upper_bound, lower_bound)
        )
        noise = np.random.uniform(low=lower_bound, high=upper_bound, size=arr.shape)

    arr_perturbed = copy.copy(arr)
    arr_perturbed[indices] = (arr_perturbed + noise)[indices]

    return arr_perturbed


def rotation(arr: np.array, perturb_angle: float = 10, **kwargs) -> np.array:
    """
    Rotate array by some given angle.
    Assumes image type data and channel first layout.

    Parameters
    ----------
        arr: array to be perturbed
        perturb_angle: rotation angle
    """
    if arr.ndim != 3:
        raise ValueError(
            "perturb func 'rotation' requires image-type data."
            "Check that this perturb_func receives a 3D array."
        )

    matrix = cv2.getRotationMatrix2D(
        center=(arr.shape[1] / 2, arr.shape[2] / 2),
        angle=perturb_angle,
        scale=1,
    )
    arr_perturbed = cv2.warpAffine(
        np.moveaxis(arr, 0, 2),
        matrix,
        (arr.shape[1], arr.shape[2]),
    )
    arr_perturbed = np.moveaxis(arr_perturbed, 2, 0)
    return arr_perturbed


def translation_x_direction(
    arr: np.array,
    perturb_baseline: Union[float, int, str, np.array],
    perturb_dx: int = 10,
    **kwargs,
) -> np.array:
    """
    Translate array by some given value in the x-direction.
    Assumes image type data and channel first layout.

    Parameters
    ----------
        arr: array to be perturbed
        perturb_baseline: value for pixels that are missing values after translation
        perturb_dy: translation length
    """
    if arr.ndim != 3:
        raise ValueError(
            "perturb func 'translation_x_direction' requires image-type data."
            "Check that this perturb_func receives a 3D array."
        )

    matrix = np.float32([[1, 0, perturb_dx], [0, 1, 0]])
    arr_perturbed = cv2.warpAffine(
        np.moveaxis(arr, 0, -1),
        matrix,
        (arr.shape[1], arr.shape[2]),
        borderValue=get_baseline_value(
            value=perturb_baseline,
            arr=arr,
            return_shape=(arr.shape[0]),
            **kwargs,
        ),
    )
    arr_perturbed = np.moveaxis(arr_perturbed, -1, 0)
    return arr_perturbed


def translation_y_direction(
    arr: np.array,
    perturb_baseline: Union[float, int, str, np.array],
    perturb_dx: int = 10,
    **kwargs,
) -> np.array:
    """
    Translate array by some given value in the x-direction.
    Assumes image type data and channel first layout.

    Parameters
    ----------
        arr: array to be perturbed
        perturb_baseline: value for pixels that are missing values after translation
        perturb_dy: translation length
    """
    if arr.ndim != 3:
        raise ValueError(
            "perturb func 'translation_y_direction' requires image-type data."
            "Check that this perturb_func receives a 3D array."
        )

    matrix = np.float32([[1, 0, 0], [0, 1, perturb_dx]])
    arr_perturbed = cv2.warpAffine(
        np.moveaxis(arr, 0, 2),
        matrix,
        (arr.shape[1], arr.shape[2]),
        borderValue=get_baseline_value(
            value=perturb_baseline,
            arr=arr,
            return_shape=(arr.shape[0]),
            **kwargs,
        ),
    )
    arr_perturbed = np.moveaxis(arr_perturbed, 2, 0)
    return arr_perturbed


def noisy_linear_imputation(
    arr: np.array,
    indices: Union[int, Sequence[int], Tuple[np.array]],
    noise: float = 0.01,
    **kwargs,
) -> np.array:
    """
    Based on https://github.com/tleemann/road_evaluation.
    Calculates noisy linear imputation for the given array and a list of indices indicating which elements are not
    included in the mask.

    Parameters
    ----------
        arr: array to be perturbed
        indices: array-like, subset of all indices in the array to not be included in the mask.
    """
    offset_weight = [
        ((1, 1), 1 / 12),
        ((0, 1), 1 / 6),
        ((-1, 1), 1 / 12),
        ((1, -1), 1 / 12),
        ((0, -1), 1 / 6),
        ((-1, -1), 1 / 12),
        ((1, 0), 1 / 6),
        ((-1, 0), 1 / 6),
    ]
    arr_flat = arr.reshape((arr.shape[0], -1))

    mask = np.ones(arr_flat.shape[1])
    mask[indices] = 0

    ind_to_var_ids = np.zeros(arr_flat.shape[1], dtype=int)
    ind_to_var_ids[indices] = np.arange(len(indices))

    # Equation system left-hand side.
    a = lil_matrix((len(indices), len(indices)))

    # Equation system right-hand side.
    b = np.zeros((len(indices), arr.shape[0]))

    sum_neighbors = np.ones(len(indices))

    for n in offset_weight:
        offset, weight = n[0], n[1]
        off_coords, valid = offset_coordinates(indices, offset, arr.shape)
        valid_ids = np.argwhere(valid == 1).flatten()

        # Add weighted values to vector b.
        in_off_coord = off_coords[mask[off_coords] == 1]
        in_off_coord_ids = valid_ids[mask[off_coords] == 1]
        b[in_off_coord_ids, :] -= weight * arr_flat[:, in_off_coord].T

        # Add weights to a.
        out_off_coord = off_coords[mask[off_coords] != 1]
        out_off_coord_ids = valid_ids[mask[off_coords] != 1]
        variable_ids = ind_to_var_ids[out_off_coord]
        a[out_off_coord_ids, variable_ids] = weight

        # Reduce weight for invalid coordinates.
        sum_neighbors[np.argwhere(valid == 0).flatten()] = (
            sum_neighbors[np.argwhere(valid == 0).flatten()] - weight
        )

    a[np.arange(len(indices)), np.arange(len(indices))] = -sum_neighbors

    # Solve the system of equations.
    res = np.transpose(spsolve(csc_matrix(a), b))

    # Fill the values with the solution of the system.
    arr_flat_copy = np.copy(arr.reshape((arr.shape[0], -1)))
    arr_flat_copy[:, indices] = res + noise * np.random.randn(*res.shape)

    return arr_flat_copy.reshape(*arr.shape)


def no_perturbation(arr: np.array, **kwargs) -> np.array:
    """Apply no perturbation to input."""
    return arr
