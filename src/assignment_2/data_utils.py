"""
Data loading and pre-processing utilities for Assignment 2.

Lefki Athanasopoulou
Numerical Optimization and Large Scale Linear Algebra
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

# Default data path (adjust as needed)
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "assignment_2"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(file_path: Optional[str | Path] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four Excel sheets (azip, dzip, testzip, dtest).

    Parameters
    ----------
    file_path : path to the .xlsx file.
                If *None*, looks for ``data.xlsx`` in data/assignment_2/.

    Returns
    -------
    (azip_df, dzip_df, testzip_df, dtest_df)
    """
    if file_path is None:
        file_path = _DEFAULT_DATA_DIR / "data.xlsx"
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {file_path}. "
            "Please place data.xlsx in data/assignment_2/."
        )

    azip_df = pd.read_excel(file_path, sheet_name="azip", header=None)
    dzip_df = pd.read_excel(file_path, sheet_name="dzip", header=None)
    testzip_df = pd.read_excel(file_path, sheet_name="testzip", header=None)
    dtest_df = pd.read_excel(file_path, sheet_name="dtest", header=None)
    return azip_df, dzip_df, testzip_df, dtest_df


# ---------------------------------------------------------------------------
# Organise into per-digit dictionary
# ---------------------------------------------------------------------------
def dictionary_format(df_digits: pd.Series, df_arrays: pd.DataFrame) -> dict[int, np.ndarray]:
    """Convert flat arrays into a digit-keyed dictionary.

    Parameters
    ----------
    df_digits : series of digit labels (length N_images)
    df_arrays : dataframe of shape (256, N_images)

    Returns
    -------
    dict mapping digit (0-9) -> np.ndarray of shape (256, N_digit)
    """
    digits = df_digits.to_numpy().flatten()
    arrays = df_arrays.to_numpy()

    dic: dict[int, list] = {i: [] for i in range(10)}
    for i in range(len(digits)):
        label = int(digits[i])
        dic[label].append(arrays[:, i])

    for digit in dic:
        dic[digit] = np.array(dic[digit]).T  # (256, N_digit)

    return dic


# ---------------------------------------------------------------------------
# Image display (MATLAB-style)
# ---------------------------------------------------------------------------
def ima2(A, ax=None):
    """Display a 256-dim vector as a 16x16 grayscale image.

    Parameters
    ----------
    A  : 1-D array-like of length 256
    ax : optional matplotlib Axes to draw on
    """
    a1 = np.squeeze(np.asarray(A))
    a1 = np.reshape(a1, (16, 16)).T
    a1 = a1 - a1.min()
    if a1.max() != 0:
        a1 = (20 / a1.max()) * a1

    mymap1 = np.array([
        1.0000, 1.0000, 1.0000, 0.8715, 0.9028, 0.9028, 0.7431, 0.8056, 0.8056,
        0.6146, 0.7083, 0.7083, 0.4861, 0.6111, 0.6111, 0.3889, 0.4722, 0.5139,
        0.2917, 0.3333, 0.4167, 0.1944, 0.1944, 0.3194, 0.0972, 0.0972, 0.1806,
        0, 0, 0.0417,
    ])
    cmap = ListedColormap(mymap1.reshape(-1, 1).repeat(3, axis=1))

    if ax is None:
        fig, ax_new = plt.subplots(figsize=(6, 4))
        ax_new.imshow(a1, cmap=cmap, vmin=0, vmax=20)
        ax_new.axis("on")
        plt.show()
    else:
        ax.imshow(a1, cmap=cmap, vmin=0, vmax=20)
        ax.axis("on")
