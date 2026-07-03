"""
Visualization utilities for Assignment 2.

Plots accuracy curves, confusion matrices, singular-value decay,
and misclassified digit images.

Lefki Athanasopoulou
Numerical Optimization and Large Scale Linear Algebra
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "assignment_2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Accuracy vs. number of singular vectors
# ---------------------------------------------------------------------------
def plot_accuracy_vs_k(
    accuracy_results: dict[int, float],
    filename: str = "accuracy_vs_k.png",
    save: bool = True,
) -> None:
    """Line plot of classification accuracy vs. k (number of basis vectors).

    Parameters
    ----------
    accuracy_results : dict mapping k -> accuracy (%)
    filename         : output file name
    save             : whether to save
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 10))

    ks = list(accuracy_results.keys())
    accs = list(accuracy_results.values())

    sns.lineplot(x=ks, y=accs, marker="o", linewidth=2.5, color="royalblue", ax=ax)

    # Annotate each point
    results_df = pd.DataFrame({"k": ks, "Accuracy (%)": accs})
    for _, row in results_df.iterrows():
        ax.text(
            row["k"], row["Accuracy (%)"] + 0.05,
            f"{row['Accuracy (%)']:.2f}%",
            fontsize=10, ha="center", va="bottom",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
        )

    ax.set_xlabel("Number of Singular Vectors (k)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Classification Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Accuracy vs. Number of Singular Vectors", fontsize=14, fontweight="bold")

    if save:
        fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    filename: str = "confusion_matrix.png",
    save: bool = True,
) -> None:
    """Heat-map of a confusion matrix with logarithmic normalisation.

    Parameters
    ----------
    cm       : (10, 10) confusion matrix
    title    : figure title
    filename : output file name
    save     : whether to save
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    log_norm = mcolors.LogNorm(vmin=1, vmax=cm.max())
    colors = ["#ffffff", "#dbbdcf", "#748aad", "#bf6399", "#4f2679"]
    cmap = mcolors.LinearSegmentedColormap.from_list("distinct_cmap", colors)

    sns.heatmap(cm, annot=True, cmap=cmap, norm=log_norm, fmt="d",
                linewidths=0.5, linecolor="gray", ax=ax)

    ax.set_xlabel("Predicted Digit", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Digit", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")

    if save:
        fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Singular value decay
# ---------------------------------------------------------------------------
def plot_singular_values(
    svd_train: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    k: int,
    filename: str = "singular_values.png",
    save: bool = True,
) -> None:
    """Plot the first *k* singular values for each digit class.

    Parameters
    ----------
    svd_train : dict from compute_svd()
    k         : number of singular values to display
    filename  : output file name
    save      : whether to save
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    for digit, (U, S, Vt) in svd_train.items():
        ax.plot(S[:k], label=f"Digit {digit}")

    ax.set_xticks(range(0, k + 1, max(1, k // 10)))
    ax.set_xlabel("Singular Value Index", fontsize=12)
    ax.set_ylabel("Singular Value Magnitude", fontsize=12)
    ax.set_title(f"Singular Values for Different Digit Classes (Top {k})", fontsize=14, fontweight="bold")
    ax.legend(title="Digit Class", fontsize=10)
    ax.grid(True)

    if save:
        fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Display misclassified digits
# ---------------------------------------------------------------------------
def display_misclassified(
    test_images_df: pd.DataFrame,
    test_labels: np.ndarray,
    predictions: np.ndarray,
    ima2_fn,
    max_samples: int = 10,
) -> int:
    """Show the first *max_samples* misclassified digits.

    Parameters
    ----------
    test_images_df : DataFrame whose columns are test images
    test_labels    : (N,) true labels
    predictions    : (N,) predicted labels
    ima2_fn        : the ima2 display function
    max_samples    : how many to show

    Returns
    -------
    int – total number of misclassified digits
    """
    misclassified = np.where(predictions != test_labels)[0]
    print(f"Total misclassified digits: {len(misclassified)}")

    for idx in misclassified[:max_samples]:
        print(f"  True Digit: {test_labels[idx]}, Predicted: {predictions[idx]}")
        ima2_fn(test_images_df.iloc[:, idx])

    return len(misclassified)
