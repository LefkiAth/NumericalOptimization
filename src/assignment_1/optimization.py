"""
Optimization routines for Assignment 1: Illumination Optimization.

Implements:
  - Unconstrained least-squares solution
  - Non-negative constrained least-squares (CVXPY)
  - Random search over lamp positions
  - Differential Evolution over lamp positions

Lefki Athanasopoulou
Numerical Optimization and Large Scale Linear Algebra
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution

from src.assignment_1.config import (
    HEIGHT_RANGE,
    LAMP_POSITIONS,
    N_LAMPS,
    N_PIXELS,
    PIXEL_RANGE,
    TOTAL_POWER,
)
from src.assignment_1.illumination import (
    compute_rms_error,
    desired_illumination,
)


# ---------------------------------------------------------------------------
# Data class for storing optimization results
# ---------------------------------------------------------------------------
@dataclass
class OptimizationResult:
    """Container for a single optimization run."""
    rms_error: float
    lamp_powers: np.ndarray
    illumination: np.ndarray
    lamp_positions: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Helper: evaluate a given set of lamp positions
# ---------------------------------------------------------------------------
def _evaluate_positions(
    lamp_positions: np.ndarray,
    pixels: np.ndarray,
    l_des: np.ndarray,
    m: int,
    n: int,
    total_power: float,
):
    """Evaluate RMS error for a given set of lamp positions.

    Returns (rms, illumination, optimal_powers).
    """
    distances = cdist(pixels, lamp_positions, "sqeuclidean")
    A = 1.0 / distances
    scale = np.sum(A) / m
    A_scaled = A / scale

    p = cp.Variable(n, nonneg=True)
    objective = cp.Minimize(cp.sum_squares(A_scaled @ p - l_des))
    constraints = [cp.sum(p) == total_power]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    l_pred = A_scaled @ p.value
    rms = float(np.sqrt(np.mean((l_pred - l_des) ** 2)))
    return rms, l_pred, p.value


# ---------------------------------------------------------------------------
# 1. Unconstrained least-squares
# ---------------------------------------------------------------------------
def unconstrained_least_squares(
    A_scaled: np.ndarray,
    l_des: np.ndarray,
) -> OptimizationResult:
    """Solve  min_p || A_scaled p - l_des ||_2^2  with no constraints."""
    p, *_ = np.linalg.lstsq(A_scaled, l_des, rcond=None)
    l_pred = A_scaled @ p
    rms = compute_rms_error(l_pred, l_des)
    return OptimizationResult(rms_error=rms, lamp_powers=p, illumination=l_pred)


# ---------------------------------------------------------------------------
# 2. Constrained least-squares (non-negative powers, sum = total_power)
# ---------------------------------------------------------------------------
def constrained_least_squares(
    A_scaled: np.ndarray,
    l_des: np.ndarray,
    total_power: float = TOTAL_POWER,
    n_lamps: int = N_LAMPS,
) -> OptimizationResult:
    """Solve:
        min_p  || A_scaled p - l_des ||_2^2
        s.t.   p >= 0,  sum(p) = total_power
    """
    p = cp.Variable(n_lamps, nonneg=True)
    objective = cp.Minimize(cp.sum_squares(A_scaled @ p - l_des))
    constraints = [cp.sum(p) == total_power]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    l_pred = A_scaled @ p.value
    rms = compute_rms_error(l_pred, l_des)
    return OptimizationResult(
        rms_error=rms,
        lamp_powers=p.value,
        illumination=l_pred,
    )


# ---------------------------------------------------------------------------
# 3. Random search over lamp positions
# ---------------------------------------------------------------------------
def random_search(
    pixels: np.ndarray,
    n_lamps: int = N_LAMPS,
    m: int = N_PIXELS,
    max_trials: int = 1000,
    total_power: float = TOTAL_POWER,
    baseline_rms: Optional[float] = None,
    seed: int = 42,
) -> OptimizationResult:
    """Randomly sample lamp positions and stop when one beats *baseline_rms*."""
    rng = np.random.RandomState(seed)
    l_des = desired_illumination(m)

    if baseline_rms is None:
        distances = cdist(pixels, LAMP_POSITIONS, "sqeuclidean")
        A = 1.0 / distances
        scale = np.sum(A) / m
        A_scaled = A / scale
        baseline_rms = compute_rms_error(A_scaled @ np.ones(n_lamps), l_des)

    best_rms = float("inf")
    best_pos = None
    best_illum = None
    best_powers = None

    for trial in range(1, max_trials + 1):
        random_positions = np.column_stack((
            rng.uniform(PIXEL_RANGE[0], PIXEL_RANGE[1], n_lamps),
            rng.uniform(PIXEL_RANGE[0], PIXEL_RANGE[1], n_lamps),
            rng.uniform(HEIGHT_RANGE[0], HEIGHT_RANGE[1], n_lamps),
        ))
        rms, illum, powers = _evaluate_positions(
            random_positions, pixels, l_des, m, n_lamps, total_power,
        )
        if rms < best_rms:
            best_rms = rms
            best_pos = random_positions
            best_illum = illum
            best_powers = powers
            if rms < baseline_rms:
                print(f"  Trial {trial}: RMS {rms:.4f} < baseline {baseline_rms:.4f} -> accepted")
                break
        else:
            print(f"  Trial {trial}: RMS {rms:.4f} (no improvement)")

    return OptimizationResult(
        rms_error=best_rms,
        lamp_powers=best_powers,
        illumination=best_illum,
        lamp_positions=best_pos,
    )


# ---------------------------------------------------------------------------
# 4. Differential Evolution
# ---------------------------------------------------------------------------
def _de_objective(x, pixels, l_des, m, n, total_power):
    """Objective function for scipy's differential_evolution."""
    lamp_positions = np.array(x).reshape((n, 3))
    rms, _, _ = _evaluate_positions(lamp_positions, pixels, l_des, m, n, total_power)
    return rms


def differential_evolution_optim(
    pixels: np.ndarray,
    n_lamps: int = N_LAMPS,
    m: int = N_PIXELS,
    total_power: float = TOTAL_POWER,
    seed: int = 42,
    patience: int = 20,
    maxiter: int = 100,
    popsize: int = 15,
) -> OptimizationResult:
    """Run scipy Differential Evolution to optimise lamp positions."""
    l_des = desired_illumination(m)

    bounds = []
    for _ in range(n_lamps):
        bounds.append((PIXEL_RANGE[0], PIXEL_RANGE[1]))
        bounds.append((PIXEL_RANGE[0], PIXEL_RANGE[1]))
        bounds.append((HEIGHT_RANGE[0], HEIGHT_RANGE[1]))

    best_val = [float("inf")]
    stale = [0]

    def _cb(x, convergence):
        current = _de_objective(x, pixels, l_des, m, n_lamps, total_power)
        if current < best_val[0]:
            best_val[0] = current
            stale[0] = 0
        else:
            stale[0] += 1
            print(f"    No improvement for {stale[0]} iterations.")
        if stale[0] >= patience:
            print("    Early stopping triggered.")
            return True
        return False

    result = differential_evolution(
        _de_objective,
        bounds,
        args=(pixels, l_des, m, n_lamps, total_power),
        seed=seed,
        strategy="randtobest1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=0.01,
        disp=True,
        callback=_cb,
        vectorized=False,
    )

    optimal_positions = result.x.reshape((n_lamps, 3))
    best_rms = result.fun

    # Re-compute the final illumination for reporting
    _, l_final, powers_final = _evaluate_positions(
        optimal_positions, pixels, l_des, m, n_lamps, total_power,
    )

    return OptimizationResult(
        rms_error=best_rms,
        lamp_powers=powers_final,
        illumination=l_final,
        lamp_positions=optimal_positions,
    )
