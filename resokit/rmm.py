#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2024, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""
This module provides tools for identifying, labeling, and computing
distances to MMRs in the phase space.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import warnings
from itertools import product
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# ============================================================================
# FUNCTIONS
# ============================================================================


def three_body_mmr_curve(
    x: Union[float, np.ndarray], resonance: tuple[int, int, int]
) -> np.ndarray:
    """
    Computes the 3-body mean-motion resonance (MMR) curve.

    Parameters:
    ----------
    x : float or np.ndarray
        The independent variable for the curve.
    resonance : tuple[int, int, int]
        Coefficients (a, b, c) defining the resonance.

    Returns:
    -------
    np.ndarray
        The corresponding values of the 3-body resonance curve.
        Singularities are replaced with NaN.
    """
    a, b, c = resonance
    if np.ndim(x) == 0:
        if a * x + b == 0:
            return np.nan
        return -c / (a * x + b)
    curve = np.divide(
        -c, (a * x + b), out=np.full_like(x, np.nan), where=(a * x + b) != 0
    )

    # Handle singularities
    singularity = -b / a
    if (
        np.ndim(x) > 0
        and singularity >= np.min(x)
        and singularity <= np.max(x)
    ):
        closest_idx = np.argmin(np.abs(x - singularity))
        curve[closest_idx] = np.nan
    return curve


def find_mmrs_in_area(
    bounds: tuple[float, float, float, float],
    r3p_order: int = 0,
    r3p_maxint: int = 10,
    compute_r2p: bool = True,
    r2p_order: int = 2,
    r2p_maxint: int = 10,
) -> list:
    """
    Identifies 3-body (3P-MMR) and 2-body (2P-MMR) resonances in a
    specified phase-space region.

    Parameters:
    ----------
    bounds : tuple[float, float, float, float]
        The limits of the region (x_min, x_max, y_min, y_max).
    r3p_order : int
        Maximum allowable order for 3P-MMRs (default: 0).
    r3p_maxint : int
        Maximum integer coefficients for 3P-MMRs (default: 10).
    compute_r2p : bool
        Whether to compute 2P-MMRs (default: True).
    r2p_order : int
        Maximum allowable order for 2P-MMRs (default: 2).
    r2p_maxint : int
        Maximum integer coefficients for 2P-MMRs (default: 10).

    Returns:
    -------
    list
        A list containing detected 3P-MMRs and optionally
        2P-MMRs along the x and y axes.
    """
    x_min, x_max, y_min, y_max = bounds
    r3p_resonances = []
    coeff_range = np.flip(np.arange(-r3p_maxint, r3p_maxint + 1))

    # Identify 3P-MMRs
    for i in range(1, r3p_maxint + 1):
        for j, k in product(coeff_range, repeat=2):
            if (
                abs(i + j + k) > r3p_order  # Check order
                # or i == 0  # Adjacent 2P-MMR
                or k == 0  # Adjacent 2P-MMR
                or (
                    j == 0 and not ((i > 0) and (k < 0))
                )  # Take (i,0,-k) over (-i,0,k)
            ):
                continue
            # if i < 0:  # Use (|i|,...,...)
            #     i *= -1
            #     j *= -1
            #     k *= -1
            # Normalize coefficients
            gcd = np.gcd.reduce([i, j, k])
            i_r, j_r, k_r = i // gcd, j // gcd, k // gcd
            if [i_r, j_r, k_r] in r3p_resonances:
                continue
            # Check bounds for the resonance curve
            if not is_curve_within_bounds([i_r, j_r, k_r], bounds):
                continue
            r3p_resonances.append([i_r, j_r, k_r])

    # Identify 2P-MMRs if required
    if compute_r2p:
        r2p_x, r2p_y = [], []
        for i in range(2, r2p_maxint + 1):
            for j in range(1, i):
                if (i - j) > r2p_order:
                    continue
                gcd = np.gcd(i, j)
                i_r, j_r = i // gcd, j // gcd
                if x_min <= i_r / j_r <= x_max and [i_r, j_r] not in r2p_x:
                    r2p_x.append([i_r, j_r])
                if y_min <= i_r / j_r <= y_max and [i_r, j_r] not in r2p_y:
                    r2p_y.append([i_r, j_r])
        return [r3p_resonances, r2p_x, r2p_y]

    return r3p_resonances


def is_curve_within_bounds(
    resonance: list[int, int, int], bounds: tuple[float, float, float, float]
) -> bool:
    """
    Determines if a resonance curve intersects a bounded region.

    Parameters:
    ----------
    resonance : list[int, int, int]
        Coefficients defining the resonance.
    bounds : tuple[float, float, float, float]
        The bounding region as (x_min, x_max, y_min, y_max).

    Returns:
    -------
    bool
        True if the curve intersects the region, False otherwise.
    """
    x_min, x_max, y_min, y_max = bounds
    i, j, k = resonance

    # No singularity handling needed
    if (-j / i < x_min) or (-j / i > x_max):
        # Si cruza el eje izquierdo
        if (-k / (i * x_min + j) >= y_min) and (-k / (i * x_min + j) <= y_max):
            return True

        # Si cruza el eje derecho
        elif (-k / (i * x_max + j) >= y_min) and (
            -k / (i * x_max + j) <= y_max
        ):
            return True

        # Si cruza el eje de abajo
        elif (-(j * y_min + k) / i / y_min >= x_min) and (
            -(j * y_min + k) / i / y_min <= x_max
        ):
            return True

        return False

    # Handle singularities
    else:
        # Si cruza el eje izquierdo
        if (
            not np.isclose(-j / i, x_min)
            and (-k / (i * x_min + j) >= y_min)
            and (-k / (i * x_min + j) <= y_max)
        ):
            return True

        # si cruza el eje derecho
        elif (
            not np.isclose(-j / i, x_max)
            and (-k / (i * x_max + j) >= y_min)
            and (-k / (i * x_max + j) <= y_max)
        ):
            return True

        # si cruza el eje de abajo
        elif (
            (-(j * y_min + k) / i / y_min >= x_min)
            and (-(j * y_min + k) / i / y_min < -j / i)
        ) or (
            (-(j * y_min + k) / i / y_min > -j / i)
            and (-(j * y_min + k) / i / y_min <= x_max)
        ):
            return True

        return False


def mindist_r3p(
    a: float,
    b: float,
    resonance: tuple[int, int, int],
    bounds_x: tuple[float, float] = (1, np.inf),
    x0: float = None,
    **minimize_kwargs,
) -> tuple[float, float]:
    """
    Calculate the minimum distance to a 3-body resonance curve.

    Parameters:
    ----------
    a : float
        The x-coordinate of the point.
    b : float
        The y-coordinate of the point.
    resonance : tuple[int, int, int]
        Coefficients defining the resonance.
    bounds_x : tuple[float, float], optional. Default: (1, inf)
        The bounds for the x-coordinate.
    x0 : float, optional. Default: None
        Initial guess for the optimization.
    minimize_kwargs : dict, optional
        Additional arguments for the optimization function.

    Returns:
    -------
    tuple[float, float]
        The x-coordinate of the minimum distance and the distance value.
    """

    # Function to calculate the r3p curve value at x
    def dist2_to_curve(x):
        y = three_body_mmr_curve(x, resonance)
        if np.isnan(y) or y < 1:
            return np.inf
        return (x - a) ** 2 + (y - b) ** 2

    # Redefine x0 if necessary
    if x0 is None:
        x0 = a
    elif x0 < bounds_x[0] or x0 > bounds_x[1]:
        raise ValueError("Initial guess is out of bounds!")
    if np.isinf(dist2_to_curve(x0)) or x0 is None:
        x0 = None
        bounds_x1 = a if np.isinf(bounds_x[1]) else max(bounds_x[1], 1)
        for i, n_steps in enumerate([10, 100, 100]):
            x0_arr = np.linspace(bounds_x[0], i * bounds_x1, n_steps)
            y0 = three_body_mmr_curve(x0_arr, resonance)
            dists = (x0_arr - a) ** 2 + (y0 - b) ** 2
            # Get the x0_arr value closest to 'a' with y0 != inf
            if np.any(np.isfinite(dists)):
                x0 = x0_arr[np.nanargmin(dists)]
                break
        if x0 is None:
            raise ValueError(
                "Not a valid x0!. Please provide a (better) valid x0."
            )

    # Function to calculate the distance between a point and the curve
    # Avoid runtime warnings in subtraction
    with np.errstate(invalid="ignore"):
        result = minimize(dist2_to_curve, x0, **minimize_kwargs)
    if result.success:
        x_min = result.x[0]
        distance_min = np.sqrt(result.fun)
        return x_min, distance_min
    else:
        raise ValueError("Optimization failed!")


def r3p_label(res: tuple, ax: plt.Axes, lims: tuple = None):
    """
    Annotates a plot with the label of a resonance line
    based on its coefficients.

    The label is placed where the resonance line crosses
    either the right or top axis of the plot.
    If the line does not cross these axes,  a warning is printed.

    Parameters:
    ----------
    res : tuple
        Coefficients of the resonance line (a, b, c) in
        the form a*x + b*y + c = 0.
    ax : matplotlib.axes.Axes
        The axis object on which the label will be placed.
    lims : tuple, optional
        Custom axis limits in the format (x_min, x_max, y_min, y_max).
        If not provided, the function will use the current axis limits.

    Returns:
    -------
    None
    """
    a, b, c = res

    # Define the resonance line and its inverse
    def r(x):
        """Calculate y for a given x using the line equation."""
        return -c / (a * x + b)

    def rinv(y):
        """Calculate x for a given y using the line equation."""
        return -(b * y + c) / (a * y)

    # Get axis limits
    if lims:
        x_min, x_max, y_min, y_max = lims
    else:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

    # Check crossing on the right axis
    y = three_body_mmr_curve(x_max, res)
    if y_min <= y <= y_max:
        label = f"{a} {b} {c}"  # Compact label format
        y_ax = (y - y_min) / (
            y_max - y_min
        )  # Normalize y-coordinate for axis transform
        ax.text(1.01, y_ax, label, transform=ax.transAxes, va="center")

    # Check crossing on the top axis
    elif x_min <= rinv(y_max) <= x_max:
        label = f"{a}\n{b}\n{c}"  # Multi-line label format
        x = rinv(y_max)
        x_ax = (x - x_min) / (
            x_max - x_min
        )  # Normalize x-coordinate for axis transform
        ax.text(x_ax, 1.02, label, transform=ax.transAxes, ha="center")

    # If the line does not cross the right or top axis
    else:
        warnings.warn(f"{res} does not cross the right or top axis.")

    return


# # Test the code
# if __name__ == "__main__":
#     bounds: list = (1.05, 2, 1.05, 2)
#     r3p_order: int = 0
#     r3p_maxint: int = 4
#     compute_r2p: bool = True
#     r2p_order: int = 2
#     r2p_maxint: int = 5

#     # MMR identification
#     mmrs3p, mmrs2px, mmrs2py = find_mmrs_in_area(
#         bounds, r3p_order, r3p_maxint, compute_r2p, r2p_order, r2p_maxint
#     )

#     # # Curve computation
#     x = np.linspace(bounds[0], bounds[1], 100)
#     curve = three_body_mmr_curve(x, (1, -3, 2))

#     # Compute and Plot the curves
#     plt.figure()
#     ax = plt.gca()
#     for resonance in mmrs2px:
#         plt.axvline(resonance[0] / resonance[1], color="k", linestyle="--")
#     for resonance in mmrs2py:
#         plt.axhline(resonance[0] / resonance[1], color="k", linestyle="--")
#     for resonance in mmrs3p:
#         curve = three_body_mmr_curve(x, resonance)
#         plt.plot(x, curve, ".-", label=f"3P-MMR {resonance}")
#         r3p_label(resonance, ax, lims=bounds)
#     # For the last curve, compute the distance to a point
#     a, b = 2.2, 1.5
#     resonance = [3, -4, 1]
#     x0 = 1.5
#     x_min, distance_min = mindist_r3p(
#         a,
#         b,
#         resonance,
#         # bounds_x=bounds[:2],
#         # x0=x0
#     )
#     plt.scatter(a, b, color="r", label="Point")
#     plt.plot([a, x_min], [b, three_body_mmr_curve(x_min, resonance)], "k-")
#     plt.xlim(bounds[0], bounds[1])
#     plt.ylim(bounds[2], bounds[3])
#     # Make the plot look square
#     plt.gca().set_aspect("equal", adjustable="box")
#     # plt.legend()
#     plt.show()

#     print("All tests passed!")
