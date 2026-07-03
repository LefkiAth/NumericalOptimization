# Numerical Optimization – Large Scale Linear Algebra Assignments

> **Lefki Athanasopoulou** (f3352404)

---

## Overview

Two assignments covering numerical optimisation and large-scale linear algebra techniques applied to real problems:

| Assignment | Topic | Core Techniques |
|------------|-------|----------------|
| **1** | Lamp Illumination Optimisation | Least squares, constrained optimisation, differential evolution |
| **2** | Handwritten Digit Classification | SVD, basis projection, tangent distance |

---

## Project Structure

```
NumericalOptimization/
├── README.md
├── requirements.txt
├── data/
│   └── assignment_2/
│       └── data.xlsx              # digit dataset (not in repo)
├── outputs/
│   ├── assignment_1/              # 10 generated figures
│   └── assignment_2/              # 7 generated figures
├── src/
│   ├── __init__.py
│   ├── assignment_1/
│   │   ├── config.py              # constants, lamp positions
│   │   ├── illumination.py        # pixel grid, distance matrix, A_scaled
│   │   ├── optimization.py        # LS, constrained LS, random search, DE
│   │   ├── visualization.py       # colormaps, histograms
│   │   └── run.py                 # main entry point
│   └── assignment_2/
│       ├── data_utils.py          # data loading, ima2 digit display
│       ├── svd_classifier.py      # SVD, basis extraction, classification
│       ├── two_stage.py           # two-stage fast classification
│       ├── tangent_distance.py    # tangent distance computation
│       ├── visualization.py       # accuracy plots, confusion matrices
│       └── run.py                 # main entry point
├── Assignment_1/
│   └── Assignment1.ipynb          # original notebook (reference)
└── Assignment_2/
    └── assignment_2.ipynb         # original notebook (reference)
```

---

## Setup

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data for Assignment 2

Place `data.xlsx` inside `data/assignment_2/`. The file contains four Excel sheets:

| Sheet      | Shape        | Description                       |
|------------|-------------|-----------------------------------|
| `azip`     | 256 × 1707  | Training images (flattened 16×16) |
| `dzip`     | 1 × 1707    | Training labels (digits 0–9)      |
| `testzip`  | 256 × 2007  | Test images (flattened 16×16)     |
| `dtest`    | 1 × 2007    | Test labels                       |

---

## Assignment 1: Lamp Illumination Optimisation

### Problem Setup

We have **10 lamps** illuminating a **25 × 25 metre area** (625 pixels, each 1 m²). The illumination at each pixel follows an **inverse-square law**:

$$
l_i = \sum_{j=1}^{n} \frac{p_j}{d_{ij}^2}
$$

where $p_j$ is the power of lamp $j$ and $d_{ij}$ is the 3D distance between pixel $i$ and lamp $j$. This gives us a linear system:

$$
\mathbf{l} = A \mathbf{p}
$$

The matrix $A$ (625 × 10) is scaled so that average illumination equals 1 when all lamp powers are 1. The desired illumination is **uniform**: $\mathbf{l}_{\text{des}} = \mathbf{1}$.

The 10 lamp positions (x, y, height) are fixed at:

| Lamp | x | y | z (height) |
|------|-----|-----|------|
| 1 | 4.1 | 20.4 | 4.0 |
| 2 | 14.1 | 21.3 | 3.5 |
| 3 | 22.6 | 17.1 | 6.0 |
| 4 | 5.5 | 12.3 | 4.0 |
| 5 | 12.2 | 9.7 | 4.0 |
| 6 | 15.3 | 13.8 | 6.0 |
| 7 | 21.3 | 10.5 | 5.5 |
| 8 | 3.9 | 3.3 | 5.0 |
| 9 | 13.1 | 4.3 | 5.0 |
| 10 | 20.3 | 4.2 | 4.5 |

### Questions

#### Q1 – Baseline & Unconstrained Least Squares

**Task:** Using colormaps, show the illumination pattern for:
- All lamps set to power 1
- Lamp powers found by least squares (minimise $\|A\mathbf{p} - \mathbf{l}_{\text{des}}\|_2^2$)

**Report:** The RMS error in both cases.

**Results:**
| Method | RMS Error |
|--------|-----------|
| All lamps = 1 | 0.2459 |
| Least squares | 0.1458 |

---

#### Q2 – Histograms

**Task:** Create histograms of the pixel illumination values for:
- All lamps set to power 1
- Least-squares solution

**Insight:** The LS histogram is more concentrated around 1.0, showing improved uniformity.

---

#### Q3 – Constrained Least Squares

**Task:** Find lamp powers subject to:
- $p_i \geq 0$ (non-negative powers)
- $\sum_{i=1}^{10} p_i = 10$ (total energy = 10)

**Method:** Convex optimisation with CVXPY.

**Result:** RMS error = 0.1598

---

#### Q4 – Optimising Lamp Positions

**Task:** Find new (x, y, z) positions for the lamps that beat the RMS error from Q1. Constraints:
- x, y ∈ [1, 25] metres
- z (height) ∈ [4, 6] metres
- Total power = 10, all powers ≥ 0

**Two approaches:**

##### Q4.1 – Random Search
Randomly sample lamp positions until one beats the baseline RMS. Stop at first improvement.

**Result:** RMS error = 0.1389 (after 8 trials)

##### Q4.2 – Differential Evolution
Use scipy's `differential_evolution` global optimiser with:
- Strategy: `randtobest1bin`
- Population size: 15
- Max iterations: 100
- Early stopping patience: 20

**Result:** RMS error = 0.0303

**Conclusion:** Differential Evolution dramatically outperforms random search and fixed-position LS, achieving the most uniform illumination.

---

## Assignment 2: SVD-Based Handwritten Digit Classification

### Problem Setup

Given a training set of 1707 handwritten digit images (16×16 = 256 pixels each) and a test set of 2007 images, classify each test image into digits 0–9 using **SVD-based subspace classification**.

**Algorithm:**
1. **Training:** For each digit class (0–9), compute the SVD of its training image matrix to get left singular vectors $U$.
2. **Classification:** For a test image $z$, project it onto each class's basis $B_k$ (first $k$ columns of $U$) and compute the relative residual:
$$
\text{residual}_k = \frac{\|z - B_k B_k^T z\|_2}{\|z\|_2}
$$
3. Classify as the digit with the **smallest residual**.

### Questions

#### Q1 – Tune the Number of Basis Vectors

**Task:** Find the optimal $k$ (number of singular vectors) in the range [5, 20] using a validation set (80/20 split of training data). Plot accuracy vs. $k$.

**Results:**

| k | Validation Accuracy (%) |
|---|------------------------|
| 5 | 96.20 |
| 8 | 97.37 |
| 10 | 97.37 |
| **13** | **97.95** |
| 16 | 96.20 |
| 20 | 96.49 |

**Best k = 13** → test accuracy = 93.42%

---

#### Q2 – Per-Digit Classification Analysis

**Task:** Check if all digits are equally easy or difficult to classify. Examine difficult cases.

**Findings:**
- **Easy digits:** 0 (98.61% recall), 1 (98.48%), 6 (94.12%)
- **Hard digits:** 5 (86.25% recall), 3 (87.95%), 4 (92.00%)
- Digit 5 is often confused with 3 and 9
- Digit 3 is confused with 4 and 5
- Digit 9 is confused with 4

Many misclassified digits are genuinely poorly written.

---

#### Q3 – Per-Class k Tuning

**Task:** Plot singular value decay for each digit class. Is it motivated to use different $k$ for different classes? Experiment with fewer basis vectors for classes with fast-decaying singular values.

**Findings:**
- Digits 1 and 7 have the fastest singular value decay → candidates for reduced $k$
- Optimal: $k=8$ for digit 1, $k=10$ for digit 7 (rest stay at 13)
- Test accuracy improved slightly: 93.42% → 93.47%

---

#### Optional 1 – Two-Stage Classification

**Task:** Implement a two-stage algorithm to save computation:
- **Stage 1:** Compare the test digit to only the **first singular vector** of each class. If one residual is significantly smaller (ratio ≥ 1.3), classify immediately.
- **Stage 2:** Otherwise, use the full basis vectors.

**Results:**
- Same accuracy as full method: **93.47%**
- Stage 1 sufficient for **616 / 2007 images (30.7%)**
- Stage 2 needed for 69.3%

**Conclusion:** ~30% of images can be classified with minimal computation, making this a good efficiency–accuracy trade-off.

---

#### Optional 2 – Tangent Distance Classification

**Task:** Instead of SVD projection, classify by **tangent distance** — a distance metric that accounts for small geometric transformations (translation, rotation, scaling, thickening) of digit images.

**Method:**
1. Resize images to 20×20
2. Compute tangent matrices (7 transformation derivatives per image)
3. For each test image, find the training image with smallest tangent distance
4. Classify using that training image's label

**Results:**
- Tangent distance accuracy: **95.17%** (vs 93.47% for SVD)
- Better across most digits, especially digit 5 (91.25% → 94.71% recall)

---

## Running

```bash
# Assignment 1
python -m src.assignment_1.run

# Assignment 2 (requires data.xlsx)
python -m src.assignment_2.run
```

All generated figures are saved to `outputs/assignment_1/` and `outputs/assignment_2/`.

---

## Outputs Summary

### Assignment 1 (10 figures)

| File | Question | Description |
|------|----------|-------------|
| `q1_all_ones_colormap.png` | Q1 | Illumination heatmap, all lamps at 1 |
| `q1_ls_colormap.png` | Q1 | Illumination heatmap, least squares |
| `q2_all_ones_hist.png` | Q2 | Histogram, all lamps at 1 |
| `q2_ls_hist.png` | Q2 | Histogram, least squares |
| `q3_constrained_ls_colormap.png` | Q3 | Constrained LS heatmap |
| `q3_constrained_ls_hist.png` | Q3 | Constrained LS histogram |
| `q4_rms_colormap.png` | Q4.1 | Random search heatmap |
| `q4_rms_hist.png` | Q4.1 | Random search histogram |
| `q5_optimal_lamps_position_colormap.png` | Q4.2 | Differential Evolution heatmap |
| `q5_optimal_lamp_position_hist.png` | Q4.2 | Differential Evolution histogram |

### Assignment 2 (7 figures)

| File | Question | Description |
|------|----------|-------------|
| `q1_data_display.png` | Setup | Sample digit (0) display |
| `q1_accuracy_vs_number_of_singular_vectors.png` | Q1 | Accuracy vs. k plot |
| `q2_true_vs_predicted_digits_confusion_matrix.png` | Q2 | Baseline confusion matrix |
| `q3_singular_values_for_different_classes.png` | Q3 | Singular value decay per digit |
| `q3_confusion_matrix.png` | Q3 | Custom per-class k confusion matrix |
| `q4_confusion_matrix.png` | Optional | Two-stage classification confusion matrix |
| `q5_confusion_matrix.png` | Optional | Tangent distance confusion matrix |

---

## Original Notebooks

The original Jupyter notebooks are preserved under `Assignment_1/` and `Assignment_2/` for reference.
