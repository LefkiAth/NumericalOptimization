"""
Two-stage SVD classification for Assignment 2 (optional task).

Stage 1: compare the unknown digit to only the *first* singular vector of
         each class.  If one residual is significantly smaller than the next,
         classify immediately.
Stage 2: fall back to full-basis classification.

Lefki Athanasopoulou
Numerical Optimization and Large Scale Linear Algebra
"""
from __future__ import annotations

import numpy as np

from src.assignment_2.svd_classifier import classify_single, extract_basis_per_class


def classify_digit_two_stage(
    test_image: np.ndarray,
    svd_train: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    basis_full: dict[int, np.ndarray],
    threshold: float = 1.3,
) -> tuple[int, float, bool]:
    """Classify a single digit with the two-stage algorithm.

    Parameters
    ----------
    test_image : (256,) vector
    svd_train  : dict from compute_svd()
    basis_full : dict from extract_basis_per_class() (per-class k)
    threshold  : ratio of 2nd-best / best residual to trigger Stage 2

    Returns
    -------
    (predicted_digit, residual, used_stage1)
    """
    # --- Stage 1: first singular vector only ---
    residuals_s1 = {}
    for digit, (U, S, Vt) in svd_train.items():
        B1 = U[:, :1]
        projection = B1 @ (B1.T @ test_image)
        residual = np.linalg.norm(test_image - projection) / np.linalg.norm(test_image)
        residuals_s1[digit] = residual

    sorted_digits = sorted(residuals_s1, key=lambda d: residuals_s1[d])
    min_digit = sorted_digits[0]
    min_residual = residuals_s1[min_digit]
    second_residual = residuals_s1[sorted_digits[1]]

    ratio = second_residual / min_residual if min_residual > 0 else np.inf

    if ratio >= threshold:
        return min_digit, min_residual, True

    # --- Stage 2: full basis ---
    best_digit, best_residual = classify_single(test_image, basis_full)
    return best_digit, best_residual, False


def classify_test_set_two_stage(
    images: np.ndarray,
    svd_train: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    basis_full: dict[int, np.ndarray],
    threshold: float = 1.3,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Classify all test images with the two-stage algorithm.

    Parameters
    ----------
    images     : (256, N_test)
    svd_train  : dict from compute_svd()
    basis_full : per-class basis dict
    threshold  : Stage 1 ratio threshold

    Returns
    -------
    (predictions, residuals, n_stage1_classified)
    """
    n_images = images.shape[1]
    predictions = np.empty(n_images, dtype=int)
    residuals = np.empty(n_images)
    stage1_count = 0

    for i in range(n_images):
        pred, res, used_s1 = classify_digit_two_stage(
            images[:, i], svd_train, basis_full, threshold,
        )
        predictions[i] = pred
        residuals[i] = res
        if used_s1:
            stage1_count += 1

    return predictions, residuals, stage1_count
