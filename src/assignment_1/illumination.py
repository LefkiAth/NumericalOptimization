"""
Core illumination matrix computations for Assignment 1.

Computes the illumination matrix A from inverse-square law distances
between lamp positions and a pixel grid on the floor.

Lefki Athanasopoulou
Numerical Optimization and Large Scale Linear Algebra
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from src.assignment_1.config import (
    GRID_SIZE,
    L_DESIRED_VALUE,
    N_LAMPS,
    N_PIXELS,
    PIXEL_RANGE,
)


def generate_pixel_grid(grid_size: int = GRID_SIZE) -> np.ndarray:
    """Create a (m, 3) array of pixel (x, y, z) coordinates on the floor.

    The grid spans *grid_size* x *grid_size* with integer coordinates
    starting at 1.  z = 0 for every pixel (floor level).

    Returns
    -------
    np.ndarray of shape (grid_size**2, 3)
    """
    x, y = np.meshgrid(
        np.arange(PIXEL_RANGE[0], PIXEL_RANGE[0] + grid_size),
        np.arange(PIXEL_RANGE[0], PIXEL_RANGE[0] + grid_size),
        indexing="ij",
    )
    pixels = np.stack([x.ravel(), y.ravel(), np.zeros(grid_size ** 2)], axis=1)
    return pixels


def compute_squared_distances(pixels: np.ndarray, lamps: np.ndarray) -> np.ndarray:
    """Return the squared Euclidean distances between every pixel and lamp.

    Parameters
    ----------
    pixels : (m, 3) array
    lamps  : (n, 3) array

    Returns
    -------
    np.ndarray of shape (m, n)
    """
    return cdist(pixels, lamps, metric="sqeuclidean")


def compute_illumination_matrix(distances: np.ndarray) -> np.ndarray:
    """Build the illumination matrix A from squared distances.

    A[i, j] = 1 / d_ij^2  (inverse-square law).

    Parameters
    ----------
    distances : (m, n) array of squared Euclidean distances

    Returns
    -------
    np.ndarray of shape (m, n)
    """
    return 1.0 / distances


def scale_illumination_matrix(A: np.ndarray, m: int = N_PIXELS) -> np.ndarray:
    """Scale *A* so that the average illumination equals 1 when all lamp
    powers are set to 1.

    A_scaled = A / (sum(A) / m)

    Parameters
    ----------
    A : (m, n) illumination matrix
    m : number of pixels (default 625)

    Returns
    -------
    np.ndarray of shape (m, n)
    """
    scale = np.sum(A) / m
    return A / scale


def build_scaled_illumination_matrix(
    lamps: np.ndarray,
    pixels: np.ndarray,
    m: int = N_PIXELS,
) -> np.ndarray:
    """End-to-end pipeline: distances -> A -> A_scaled.

    Convenience wrapper that chains the individual steps.

    Parameters
    ----------
    lamps  : (n, 3) lamp positions
    pixels : (m, 3) pixel coordinates
    m      : number of pixels

    Returns
    -------
    np.ndarray of shape (m, n)
    """
    distances = compute_squared_distances(pixels, lamps)
    A = compute_illumination_matrix(distances)
    return scale_illumination_matrix(A, m)


def desired_illumination(m: int = N_PIXELS, value: float = L_DESIRED_VALUE) -> np.ndarray:
    """Return the desired illumination vector (uniform illumination).

    Parameters
    ----------
    m     : number of pixels
    value : target illumination value

    Returns
    -------
    np.ndarray of shape (m,)
    """
    return np.full(m, value)


def compute_rms_error(predicted: np.ndarray, desired: np.ndarray) -> float:
    """Root mean squared error between predicted and desired illumination.

    Parameters
    ----------
    predicted : (m,) array of predicted illumination
    desired   : (m,) array of desired illumination

    Returns
    -------
    float
    """
    return float(np.sqrt(np.mean((predicted - desired) ** 2)))
