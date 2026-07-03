"""
SVD-based digit classifier for Assignment 2.

Implements:
  - SVD computation for each digit class
  - Basis extraction (top-k singular vectors)
  - Residual-based classification
  - Accuracy evaluation across different k values

Lefki Athanasopoulou
Numerical Optimization and Large Scale Linear Algebra
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score

from src.assignment_2.data_utils import dictionary_format


# ---------------------------------------------------------------------------
# SVD computation
# ---------------------------------------------------------------------------
def compute_svd(dataset_classes: dict[int, np.ndarray]) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute the (thin) SVD for every digit class.

    Parameters
    ----------
    dataset_classes : dict mapping digit -> (256, N_digit) array

    Returns
    -------
    dict mapping digit -> (U, S, Vt)
    """
    svd_dict: dict[int, tuple] = {}
    for digit, arrays in dataset_classes.items():
        U, S, Vt = np.linalg.svd(arrays, full_matrices=False)
        svd_dict[digit] = (U, S, Vt)
    return svd_dict


# ---------------------------------------------------------------------------
# Basis extraction
# ---------------------------------------------------------------------------
def extract_basis(
    svd_dataset: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    k: int,
) -> dict[int, np.ndarray]:
    """Return the first *k* left singular vectors for each digit class.

    Parameters
    ----------
    svd_dataset : dict from compute_svd()
    k           : number of singular vectors to keep

    Returns
    -------
    dict mapping digit -> (256, k) basis matrix
    """
    basis_dict: dict[int, np.ndarray] = {}
    for label, (U, S, Vt) in svd_dataset.items():
        basis_dict[label] = U[:, :k]
    return basis_dict


def extract_basis_per_class(
    svd_dataset: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    k_values: dict[int, int],
    default_k: int = 13,
) -> dict[int, np.ndarray]:
    """Return a per-class basis using different k for different digits.

    Parameters
    ----------
    svd_dataset : dict from compute_svd()
    k_values    : mapping digit -> custom k (digits not listed use default_k)
    default_k   : fallback k

    Returns
    -------
    dict mapping digit -> (256, k_i) basis matrix
    """
    basis_dict: dict[int, np.ndarray] = {}
    for label, (U, S, Vt) in svd_dataset.items():
        k = k_values.get(label, default_k)
        basis_dict[label] = U[:, :k]
    return basis_dict


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_single(
    image: np.ndarray,
    basis_dict: dict[int, np.ndarray],
) -> tuple[int, float]:
    """Classify one image by minimum relative residual.

    Parameters
    ----------
    image      : (256,) flattened test image
    basis_dict : digit -> (256, k) basis matrix

    Returns
    -------
    (predicted_digit, relative_residual)
    """
    min_residual = np.inf
    classified_digit = -1
    image_norm = np.linalg.norm(image)

    for digit, B in basis_dict.items():
        projection = B @ (B.T @ image)
        residual_norm = np.linalg.norm(image - projection)
        relative_residual = residual_norm / image_norm if image_norm > 0 else residual_norm

        if relative_residual < min_residual:
            min_residual = relative_residual
            classified_digit = digit

    return classified_digit, min_residual


def classify_test_set(
    images: np.ndarray,
    basis_dict: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Classify a batch of test images.

    Parameters
    ----------
    images     : (256, N_test) array
    basis_dict : digit -> (256, k) basis matrix

    Returns
    -------
    (predictions, residuals) each of length N_test
    """
    n_images = images.shape[1]
    predictions = np.empty(n_images, dtype=int)
    residuals = np.empty(n_images)

    for i in range(n_images):
        pred, res = classify_single(images[:, i], basis_dict)
        predictions[i] = pred
        residuals[i] = res

    return predictions, residuals


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------
def evaluate_accuracy_vs_basis(
    labels: np.ndarray,
    images: np.ndarray,
    svd_train: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    k_values: range,
) -> dict[int, float]:
    """Evaluate classification accuracy for each k in *k_values*.

    Parameters
    ----------
    labels    : (N,) true labels
    images    : (256, N) test / validation images
    svd_train : dict from compute_svd() on training data
    k_values  : iterable of k values to try

    Returns
    -------
    dict mapping k -> accuracy (percentage)
    """
    accuracy_dict: dict[int, float] = {}
    for k in k_values:
        basis_dict = extract_basis(svd_train, k)
        preds, _ = classify_test_set(images, basis_dict)
        acc = accuracy_score(labels, preds) * 100
        accuracy_dict[k] = acc
    return accuracy_dict


def build_train_val_split(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split training data into train / validation sets (stratified).

    Parameters
    ----------
    train_images : (N, 256) array of training images
    train_labels : (N,) array of labels
    test_size    : fraction for validation
    seed         : random seed

    Returns
    -------
    (train_images_split, val_images_split, train_labels_split, val_labels_split)
    """
    from sklearn.model_selection import train_test_split

    images_full = train_images  # (N, 256)
    labels_full = train_labels  # (N,)

    img_tr, img_val, lbl_tr, lbl_val = train_test_split(
        images_full, labels_full,
        test_size=test_size, random_state=seed, stratify=labels_full,
    )
    return img_tr, img_val, lbl_tr, lbl_val
