#!/usr/bin/env python3
"""
Main entry point for Assignment 1: Illumination Optimization.

Run:
    python -m src.assignment_1.run

Lefki Athanasopoulou
Numerical Optimization and Large Scale Linear Algebra
"""
from __future__ import annotations

import numpy as np

from src.assignment_1.config import LAMP_POSITIONS, N_LAMPS, N_PIXELS, TOTAL_POWER
from src.assignment_1.illumination import (
    build_scaled_illumination_matrix,
    compute_rms_error,
    desired_illumination,
    generate_pixel_grid,
)
from src.assignment_1.optimization import (
    constrained_least_squares,
    differential_evolution_optim,
    random_search,
    unconstrained_least_squares,
)
from src.assignment_1.visualization import (
    COLORMAP_GOLD,
    COLORMAP_HEAT,
    COLORMAP_PINK,
    COLORMAP_PURPLE,
    COLORMAP_VIBRANT,
    plot_histogram,
    plot_illumination_colormap,
)


def main() -> None:
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    pixels = generate_pixel_grid()
    l_des = desired_illumination()
    A_scaled = build_scaled_illumination_matrix(LAMP_POSITIONS, pixels)
    n, m = N_LAMPS, N_PIXELS

    # ==================================================================
    # Q1 – All lamps at power 1
    # ==================================================================
    print("=" * 60)
    print("Q1: All lamps set to power 1")
    print("=" * 60)
    l_ones = A_scaled @ np.ones(n)
    rms_ones = compute_rms_error(l_ones, l_des)
    print(f"  Mean illumination : {np.mean(l_ones):.4f}")
    print(f"  RMS error         : {rms_ones:.4f}")

    plot_illumination_colormap(
        l_ones, "Illumination - All lamps set to 1",
        "q1_all_ones_colormap.png", cmap=COLORMAP_PURPLE,
    )

    # ==================================================================
    # Q1 – Unconstrained least-squares
    # ==================================================================
    print("\n" + "=" * 60)
    print("Q1: Unconstrained Least-Squares solution")
    print("=" * 60)
    res_ls = unconstrained_least_squares(A_scaled, l_des)
    print(f"  Mean illumination : {np.mean(res_ls.illumination):.4f}")
    print(f"  RMS error         : {res_ls.rms_error:.4f}")
    print(f"  Lamp powers       : {res_ls.lamp_powers}")

    plot_illumination_colormap(
        res_ls.illumination, "Illumination - LS Solution",
        "q1_ls_colormap.png", cmap=COLORMAP_GOLD,
    )

    # ==================================================================
    # Q2 – Histograms
    # ==================================================================
    print("\n" + "=" * 60)
    print("Q2: Histograms of illumination distributions")
    print("=" * 60)

    plot_histogram(
        l_ones, "Histogram of Illumination (lamps set to 1)",
        "q2_hist_all_ones.png",
    )
    plot_histogram(
        res_ls.illumination, "Histogram of Illumination (LS solution)",
        "q2_hist_ls.png", cmap=COLORMAP_GOLD,
    )

    # ==================================================================
    # Q3 – Constrained least-squares
    # ==================================================================
    print("\n" + "=" * 60)
    print("Q3: Constrained Least-Squares (non-negative, sum = 10)")
    print("=" * 60)
    res_con = constrained_least_squares(A_scaled, l_des, total_power=TOTAL_POWER)
    print(f"  Optimal powers : {res_con.lamp_powers}")
    print(f"  RMS error      : {res_con.rms_error:.4f}")

    plot_illumination_colormap(
        res_con.illumination, "Illumination Pattern (Constrained LS)",
        "q3_constrained_colormap.png",
    )
    plot_histogram(
        res_con.illumination, "Histogram of Illumination (Constrained LS)",
        "q3_constrained_hist.png",
    )

    # ==================================================================
    # Q4.1 – Random search for better lamp positions
    # ==================================================================
    print("\n" + "=" * 60)
    print("Q4.1: Random Search for Optimal Lamp Positions")
    print("=" * 60)
    res_random = random_search(
        pixels, n_lamps=n, m=m,
        max_trials=1000, total_power=TOTAL_POWER,
        baseline_rms=res_ls.rms_error, seed=42,
    )
    print(f"  Best RMS error  : {res_random.rms_error:.4f}")
    print(f"  Lamp positions  :\n{res_random.lamp_positions}")
    print(f"  Lamp powers     : {res_random.lamp_powers}")

    plot_illumination_colormap(
        res_random.illumination,
        f"Illumination Pattern (Random Search, RMS: {res_random.rms_error:.4f})",
        "q4_1_random_colormap.png", cmap=COLORMAP_HEAT,
    )
    plot_histogram(
        res_random.illumination,
        "Histogram of Illumination (Random Search Best)",
        "q4_1_random_hist.png", cmap=COLORMAP_VIBRANT,
    )

    # ==================================================================
    # Q4.2 – Differential Evolution
    # ==================================================================
    print("\n" + "=" * 60)
    print("Q4.2: Differential Evolution for Optimal Lamp Positions")
    print("=" * 60)
    res_de = differential_evolution_optim(
        pixels, n_lamps=n, m=m,
        total_power=TOTAL_POWER, seed=42, patience=20,
        maxiter=100, popsize=15,
    )
    print(f"  Optimal RMS error    : {res_de.rms_error:.4f}")
    print(f"  Optimal lamp positions:\n{res_de.lamp_positions}")
    print(f"  Optimal lamp powers  : {res_de.lamp_powers}")

    plot_illumination_colormap(
        res_de.illumination,
        "Illumination Pattern for Best Lamp Positions (DE)",
        "q4_2_de_colormap.png", cmap=COLORMAP_PINK,
    )
    plot_histogram(
        res_de.illumination,
        "Histogram of Illumination (DE Best)",
        "q4_2_de_hist.png", cmap=COLORMAP_PINK,
    )

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  All ones      RMS: {rms_ones:.4f}")
    print(f"  LS            RMS: {res_ls.rms_error:.4f}")
    print(f"  Constrained   RMS: {res_con.rms_error:.4f}")
    print(f"  Random Search RMS: {res_random.rms_error:.4f}")
    print(f"  Diff. Evol.   RMS: {res_de.rms_error:.4f}")
    print(f"\nAll figures saved to: {__import__('pathlib').Path('outputs/assignment_1').resolve()}")


if __name__ == "__main__":
    main()
