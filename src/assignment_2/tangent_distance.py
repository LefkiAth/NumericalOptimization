"""
Tangent distance classification for Assignment 2 (optional task).

Computes tangent distances that account for small transformations
(translation, rotation, scaling, etc.) of digit images.

Lefki Athanasopoulou
Numerical Optimization and Large Scale Linear Algebra
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve
from skimage.transform import resize


# ---------------------------------------------------------------------------
# Image resizing
# ---------------------------------------------------------------------------
def resize_image(A: np.ndarray, new_size: tuple[int, int] = (20, 20)) -> np.ndarray:
    """Resize a 256-dim vector (16x16 image) to *new_size*.

    Parameters
    ----------
    A        : (256,) flattened image
    new_size : target (height, width)

    Returns
    -------
    2-D array of shape new_size
    """
    a1 = np.squeeze(np.asarray(A, dtype=float))
    a1 = np.reshape(a1, (16, 16)).T
    a1 = a1 - a1.min()
    if a1.max() != 0:
        a1 = (20 / a1.max()) * a1
    a1_resized = resize(a1, new_size, mode="reflect", anti_aliasing=True)
    if a1_resized.max() != 0:
        a1_resized = (20 / a1_resized.max()) * a1_resized
    return a1_resized


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------
def compute_derivatives(image: np.ndarray, m: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference x- and y-derivatives of an m x m image.

    Parameters
    ----------
    image : (m, m) array
    m     : image dimension

    Returns
    -------
    (p_x, p_y) each of shape (m, m)
    """
    kernel_x = np.array([[-1, 1]])
    kernel_y = np.array([[-1], [1]])
    p_x = convolve(image, kernel_x, mode="nearest")
    p_y = convolve(image, kernel_y, mode="nearest")
    return p_x, p_y


# ---------------------------------------------------------------------------
# Tangent matrix
# ---------------------------------------------------------------------------
def compute_tangent_matrix(image: np.ndarray, m: int = 20) -> np.ndarray:
    """Build the (m*m, 7) tangent matrix for an image.

    The seven columns correspond to derivatives w.r.t.:
      x-translation, y-translation, rotation, scaling,
      parallel hyperbolic, diagonal hyperbolic, thickening.

    Parameters
    ----------
    image : (256,) flattened image vector
    m     : target image dimension

    Returns
    -------
    (m*m, 7) array
    """
    img_resized = resize_image(image, new_size=(m, m))
    p_x, p_y = compute_derivatives(img_resized, m)

    x_coords, y_coords = np.meshgrid(np.arange(m), np.arange(m))
    x_centered = x_coords - (m - 1) / 2.0
    y_centered = y_coords - (m - 1) / 2.0

    T_x = p_x.flatten()
    T_y = p_y.flatten()
    T_rot = (y_centered * p_x - x_centered * p_y).flatten()
    T_scaling = (x_centered * p_x + y_centered * p_y).flatten()
    T_par = (x_centered * p_x - y_centered * p_y).flatten()
    T_diag = (y_centered * p_x + x_centered * p_y).flatten()
    T_thick = (p_x ** 2 + p_y ** 2).flatten()

    return np.column_stack([T_x, T_y, T_rot, T_scaling, T_par, T_diag, T_thick])


# ---------------------------------------------------------------------------
# Tangent distance between two images
# ---------------------------------------------------------------------------
def tangent_distance(p: np.ndarray, e: np.ndarray, m: int = 20) -> float:
    """Compute the tangent distance between images *p* and *e*.

    Parameters
    ----------
    p, e : (256,) flattened image vectors
    m    : target image dimension

    Returns
    -------
    float – the tangent distance
    """
    p_resized = resize_image(p, new_size=(m, m)).flatten()
    e_resized = resize_image(e, new_size=(m, m)).flatten()

    T_p = compute_tangent_matrix(p, m)
    T_e = compute_tangent_matrix(e, m)

    A_mat = np.hstack([T_p, -T_e])
    b = p_resized - e_resized

    alpha, residuals, rank, s = np.linalg.lstsq(A_mat, b, rcond=None)

    if residuals.size > 0:
        return float(np.sqrt(residuals[0]))
    return float(np.linalg.norm(b - A_mat @ alpha))


# ---------------------------------------------------------------------------
# Classification with tangent distance
# ---------------------------------------------------------------------------
def classify_with_tangent_distance(
    test_images: np.ndarray,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    m: int = 20,
) -> tuple[np.ndarray, list[float]]:
    """Classify each test image by nearest tangent distance.

    Parameters
    ----------
    test_images  : (256, N_test)
    train_images : (256, N_train)
    train_labels : (N_train,) labels
    m            : target image dimension

    Returns
    -------
    (predictions, distances)
    """
    n_test = test_images.shape[1]
    predictions = np.empty(n_test, dtype=int)
    distances: list[float] = []

    for i in range(n_test):
        test_img = test_images[:, i]
        min_dist = np.inf
        pred_label = -1

        for j in range(train_images.shape[1]):
            d = tangent_distance(train_images[:, j], test_img, m)
            if d < min_dist:
                min_dist = d
                pred_label = int(train_labels[j])

        predictions[i] = pred_label
        distances.append(min_dist)

    return predictions, distances
