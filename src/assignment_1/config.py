"""
Configuration and constants for Assignment 1: Illumination Optimization.

Lefki Athanasopoulou
Numerical Optimization and Large Scale Linear Algebra
"""
import numpy as np


# ---------- Problem dimensions ----------
N_LAMPS: int = 10
N_PIXELS: int = 625          # 25x25 grid
GRID_SIZE: int = 25
PIXEL_RANGE: tuple = (1, 25)  # x, y in [1, 25] meters
HEIGHT_RANGE: tuple = (4, 6)  # z in [4, 6] meters

# ---------- Lamp positions (x, y, height) ----------
LAMP_POSITIONS = np.array([
    [4.1,  20.4, 4.0],
    [14.1, 21.3, 3.5],
    [22.6, 17.1, 6.0],
    [5.5,  12.3, 4.0],
    [12.2,  9.7, 4.0],
    [15.3, 13.8, 6.0],
    [21.3, 10.5, 5.5],
    [3.9,   3.3, 5.0],
    [13.1,  4.3, 5.0],
    [20.3,  4.2, 4.5],
])

# ---------- Desired illumination ----------
L_DESIRED_VALUE: float = 1.0  # Uniform illumination target

# ---------- Optimization constraints ----------
TOTAL_POWER: int = 10
