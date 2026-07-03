"""
Visualization utilities for Assignment 1.

Produces colormaps, histograms, and comparison plots for illumination data.
All figures are saved to the outputs/assignment_1 directory.

Lefki Athanasopoulou
Numerical Optimization and Large Scale Linear Algebra
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "assignment_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: create a custom colormap
# ---------------------------------------------------------------------------
def make_colormap(name: str, colors: list[str]) -> LinearSegmentedColormap:
    """Build a LinearSegmentedColormap from a list of hex / named colours."""
    return LinearSegmentedColormap.from_list(name, colors)


# Pre-defined colour schemes (match the notebook originals)
COLORMAP_PURPLE = make_colormap(
    "custom_purple",
    ["#43076e", "#43076e", "#9623ad", "#ad23a0", "#ad2376",
     "#c24690", "#e378b8", "#e8cfde", "#fcfcfc"],
)
COLORMAP_GOLD = make_colormap(
    "custom_gold",
    ["#202130", "#2c2f4f", "#2d3169", "#2c328a", "#8a5e2c",
     "#ab7232", "#cc822d", "#e0861f", "#f7a94f", "#e3ba8a",
     "#fae4ca", "#f0efed"],
)
COLORMAP_CONTRAST = make_colormap(
    "custom_contrast",
    ["#614791", "#b33232", "#614791"],
)
COLORMAP_ORANGE_BLUE = make_colormap(
    "custom_ob",
    ["#242d7d", "#d17206", "#242d7d"],
)
COLORMAP_HEAT = make_colormap(
    "custom_heat",
    ["#2a2942", "#342d5e", "#3a34eb", "#ebeb34",
     "#e0890d", "#910a0f", "#6e0713"],
)
COLORMAP_PURPLE_PINK = make_colormap(
    "custom_pp",
    ["#232736", "#292c42", "#32324f", "#33198a", "#33198a",
     "#6e1d85", "#851d7e", "#cc3798", "#e35fb5", "#e0abce", "#f5f2f5"],
)
COLORMAP_VIBRANT = make_colormap(
    "custom_vibrant",
    ["#3a34eb", "#ebeb34", "#e0890d", "#910a0f",
     "#e0890d", "#ebeb34", "#3a34eb"],
)
COLORMAP_PINK = make_colormap(
    "custom_pink",
    ["#1b1fe3", "#5d2185", "#f280dd", "#5d2185", "#1b1fe3"],
)


# ---------------------------------------------------------------------------
# 1. Illumination colormap
# ---------------------------------------------------------------------------
def plot_illumination_colormap(
    illumination: np.ndarray,
    title: str,
    filename: str,
    grid_size: int = 25,
    cmap: Optional[LinearSegmentedColormap] = None,
    save: bool = True,
) -> None:
    """Plot and optionally save a 2D illumination heatmap.

    Parameters
    ----------
    illumination : (m,) array  –  flattened illumination values
    title        : figure title
    filename     : output file name (saved under OUTPUT_DIR)
    grid_size    : side length of the square grid
    cmap         : matplotlib colormap (default:gist_heat)
    save         : whether to save the figure to disk
    """
    if cmap is None:
        cmap = plt.cm.gist_heat

    grid = illumination.reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap=cmap, origin="lower")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Illumination")

    if save:
        fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Histogram with colour-mapped bars
# ---------------------------------------------------------------------------
def plot_histogram(
    illumination: np.ndarray,
    title: str,
    filename: str,
    bins: int = 30,
    cmap: Optional[LinearSegmentedColormap] = None,
    center: float = 1.0,
    save: bool = True,
) -> None:
    """Plot a colour-coded histogram of illumination values.

    Parameters
    ----------
    illumination : (m,) array
    title        : figure title
    filename     : output file name
    bins         : number of histogram bins
    cmap         : colour map (default: custom contrast)
    center       : illumination value around which the colour diverges
    save         : whether to save
    """
    if cmap is None:
        cmap = COLORMAP_CONTRAST

    fig, ax = plt.subplots(figsize=(8, 6))
    counts, bin_edges, patches = ax.hist(illumination, bins=bins, edgecolor="black")

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    norm = TwoSlopeNorm(vmin=bin_centers.min(), vcenter=center, vmax=bin_centers.max())

    for bcenter, patch in zip(bin_centers, patches):
        patch.set_facecolor(cmap(norm(bcenter)))

    ax.set_title(title)
    ax.set_xlabel("Illumination Intensity")
    ax.set_ylabel("Number of Pixels")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Illumination Value")

    if save:
        fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
