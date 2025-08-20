#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2025, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Module to manage 3-body mean-motion resonances (MMRs).

This module provides tools for identifying, labeling, and computing distances
to MMRs in the phase space.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import warnings
from itertools import product
from typing import List, Tuple, Union

import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import minimize

# ============================================================================
# FUNCTIONS
# ============================================================================


def mmr3b(
    x: Union[float, np.ndarray], resonance: Tuple[int, int, int]
) -> Union[float, np.ndarray]:
    """Compute the 3-body mean-motion resonance (MMR) curve.

    Equation
        :math:`a * x + b + c / y = 0`

    Parameters
    ----------
    x : float or np.ndarray
        The independent variable for the curve.
    resonance : tuple[int, int, int]
        Coefficients (a, b, c) defining the resonance.

    Returns
    -------
    y : float or np.ndarray
        The corresponding values of the 3-body resonance curve.
        Singularities are replaced with NaN.
    """
    a, b, c = resonance  # Coefficients of the resonance curve

    if np.ndim(x) == 0:  # Single value
        if a * x + b == 0:
            return np.nan
        return -c / (a * x + b)

    # Avoid division by zero
    curve = np.divide(
        -c, (a * x + b), out=np.full_like(x, np.nan), where=(a * x + b) != 0
    )

    # Handle singularities
    singularity = -b / a
    if (
        np.ndim(x) > 0  # x is an array
        and singularity >= np.min(x)  # Singularity is within bounds
        and singularity <= np.max(x)  # Singularity is within bounds
    ):
        closest_idx = np.argmin(np.abs(x - singularity))
        curve[closest_idx] = np.nan

    return curve


def mmrs_in_area(
    bounds: Tuple[float, float, float, float],
    order3: int = 0,
    max_coeff3: int = 10,
    max_order3: int = 0,
    mmr2b: bool = True,
    order2: int = 2,
    max_coeff2: int = 10,
    max_order2: int = 2,
    verbose: bool = False,
) -> list:
    """Identify 3-body and 2-body MMRs in a specified phase-space region.

    This function identifies 3-body mean-motion resonances (3P-MMRs) and
    optionally 2-body mean-motion resonances (2P-MMRs) in a specified
    region of the phase space.

    Parameters
    ----------
    bounds : tuple[float, float, float, float]
        The limits of the region (x_min, x_max, y_min, y_max) in the phase
        space. Remeber that x_min and y_min must be > 1.
    order3 : int, optional. Default: 0
        Exact order for 3P-MMRs.
    max_coeff3 : int, optional. Default: 10
        Calculate 3P-MMRs up to this maximum integer coefficient.
    max_order3 : int, optional. Default: 0
        Calculate 3P-MMRs up to this maximum order.
        If set to 0, only the exact order is considered.
    mmr2b : bool, optional. Default: True
        Whether to compute 2P-MMRs.
    order2 : int, optional. Default: 2
        Exact order for 2P-MMRs.
    max_coeff2 : int, optional. Default: 10
        Calculate 2P-MMRs up to this maximum integer coefficient.
    max_order2 : int, optional. Default: 2
        Calculate 2P-MMRs up to this maximum order.
        If set to 0, only the exact order is considered.
    verbose : bool, optional. Default: False
        Whether to print the resonances found.

    Returns
    -------
    resonances: list
        A list containing detected 3P-MMRs and optionally
        2P-MMRs along the x and y axes. This last case is
        returned as [r3, r2x, r2y], where r3, r2x, and r2y
        are lists of 3P-MMRs, 2P-MMRs along the x-axis, and
        2P-MMRs along the y-axis, respectively.
    """
    # Get the bounds
    x_min, x_max, y_min, y_max = bounds

    # Check bounds
    if x_min < 1 or y_min < 1:
        raise ValueError("Bounds must be greater or equal than 1.")
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Invalid bounds. Must be ascending.")

    # Initialize the list of resonances
    r3p_resonances = []

    # Define the range of coefficients
    coeff_range = np.flip(np.arange(-max_coeff3, max_coeff3 + 1))

    # Define good_order function
    def good_order3(i, j, k):
        if max_order3 == 0:
            return abs(i + j + k) == order3
        return (i + j + k) <= max_order3

    # Identify 3P-MMRs
    for i in range(1, max_coeff3 + 1):
        for j, k in product(coeff_range, repeat=2):

            if (
                not good_order3(i, j, k)  # Check order
                # or i == 0  # Adjacent 2P-MMR
                or k == 0  # Adjacent 2P-MMR
                or (
                    j == 0 and not (i > 0 and k < 0)
                )  # Take (i,0,-k) over (-i,0,k)
            ):
                continue

            # Normalize coefficients
            gcd = np.gcd.reduce([i, j, k])
            i_r, j_r, k_r = i // gcd, j // gcd, k // gcd

            # Skip if the resonance is already in the list
            if [i_r, j_r, k_r] in r3p_resonances:
                continue

            # Check bounds for the resonance curve
            if not _is_curve_within_bounds([i_r, j_r, k_r], bounds):
                continue

            r3p_resonances.append([i_r, j_r, k_r])

    if verbose:
        print(f"Found {len(r3p_resonances)} 3-body mean-motion resonances.")

    # Check if 2P-MMRs are not required
    if not mmr2b:  # Return 3P-MMRs only
        return r3p_resonances

    # 2P-MMRs ideification  required

    # Define good_order function
    def good_order2(i, j):
        if max_order2 == 0:
            return (i - j) == order2
        return (i - j) <= max_order2

    r2p_x, r2p_y = [], []
    for i in range(2, max_coeff2 + 1):
        for j in range(1, i):

            # Check order
            if not good_order2(i, j):
                continue

            # Normalize coefficients
            gcd = np.gcd(i, j)
            i_r, j_r = i // gcd, j // gcd

            if x_min <= i_r / j_r <= x_max and [i_r, j_r] not in r2p_x:
                r2p_x.append([i_r, j_r])

            if y_min <= i_r / j_r <= y_max and [i_r, j_r] not in r2p_y:
                r2p_y.append([i_r, j_r])

    if verbose:
        print(
            f"Found {len(r2p_x)} 2-body mean-motion resonances "
            + "along the x-axis."
        )
        print(
            f"Found {len(r2p_y)} 2-body mean-motion resonances "
            + "along the y-axis."
        )

    return [r3p_resonances, r2p_x, r2p_y]


def _is_curve_within_bounds(
    resonance: List[int], bounds: Tuple[float, float, float, float]
) -> bool:
    """Determine if a resonance curve intersects a bounded region.

    Parameters
    ----------
    resonance : list[int, int, int]
        Coefficients defining the resonance.
    bounds : tuple[float, float, float, float]
        The bounding region as (x_min, x_max, y_min, y_max).

    Returns
    -------
    bool
        `True` if the curve intersects the region, `False` otherwise.
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


def mindist_mmr3b(
    a: float,
    b: float,
    resonance: Tuple[int, int, int],
    x0: Union[float, None] = None,
    unphysical: bool = False,
    **minimize_kwargs,
) -> Tuple[float, float, float]:
    """Calculate the minimum distance to a 3-body resonance curve.

    Parameters
    ----------
    a : float
        The x-coordinate of the point.
    b : float
        The y-coordinate of the point.
    resonance : Tuple[int, int, int]
        Coefficients defining the resonance.
    x0 : float, optional. Default: None
        Initial guess for the optimization.
        If None, the function will use the middle point of the curve.
    unphysical : bool, optional. Default: False
        Whether to allow unphysical solutions. (y below 1)
    minimize_kwargs : dict, optional
        Additional arguments for :py:func:scipy.optimize.minimize
        function.

    Returns
    -------
    x_min : float
        The x coordinate of the minimum distance.
    y_min : float
        The y coordinate of the minimum distance.
    distance_min : float
        The minimum distance to the resonance curve.
    """
    # Singularity handling
    singularity = -resonance[1] / resonance[0]

    # Use the right or left side of the curve, in relation to singularity
    use_right = resonance[2] < 0
    # if "use_right", then the curve behaves like "1/x", else like "- 1/x"
    x_y1 = -(resonance[1] + resonance[2]) / resonance[0]

    # Fast check for (1,1) solutions
    if use_right and mmr3b(1, resonance) == 1 and not unphysical:
        # Here, the only physical solution is (1,1)
        return 1, 1, np.sqrt((1 - a) ** 2 + (1 - b) ** 2)

    # Avoid crossing the singularity
    if x_y1 < max(1, singularity):
        x_y1 = np.inf

    # Define the bounds for the optimization
    if use_right:
        bounds_x = [singularity, x_y1]
    elif not unphysical:
        bounds_x = [1, singularity]
    else:
        bounds_x = [-np.inf, singularity]

    # Function to calculate the r3p curve value at x
    def dist2_to_curve(x, unphysical=unphysical):
        y = mmr3b(x, resonance)  # Calculate the curve value
        if np.isnan(y) or (y < 1 and not unphysical):
            return np.inf
        return (x - a) ** 2 + (y - b) ** 2

    # Redefine x0 if necessary
    if x0 is None:
        # Use the middle point if the singularity is infinite
        if use_right:
            # Use the middle point if finite, else use the singularity + 1e-3
            x0 = (
                0.5 * (singularity + x_y1)
                if np.isfinite(x_y1)
                else singularity + 1e-3
            )
        else:
            x0 = 0.5 * (1 + singularity)  # Use the middle point

    elif x0 < bounds_x[0] or x0 > bounds_x[1]:  # Check if x0 is within bounds
        raise ValueError(f"Initial guess {x0} is out of bounds: {bounds_x}")

    # Function to calculate the distance between a point and the curve
    # Avoid runtime warnings in subtraction
    with np.errstate(invalid="ignore"):
        result = minimize(dist2_to_curve, x0, **minimize_kwargs)

    if result.success:
        x_min = result.x[0]
        distance_min = np.sqrt(result.fun)
        return x_min, mmr3b(x_min, resonance), distance_min
    else:
        raise ValueError("Optimization failed!")


def closest_mmr3b(
    a: float,
    b: float,
    order3: int = 0,
    max_coeff3: int = 10,
    max_order3: int = 0,
    bounds: Tuple[float, float, float, float] = None,
    radius: float = 1e-3,
    verbose: bool = True,
    **minimize_kwargs,
) -> Tuple[List[int], float]:
    """Find the closest 3-body mean-motion resonance to a point.

    Parameters
    ----------
    a : float
        The x-coordinate of the point.
    b : float
        The y-coordinate of the point.
    order3 : int, optional. Default: 0
        Exact order for 3P-MMRs.
    max_coeff3 : int, optional. Default: 10
        Calculate 3P-MMRs up to this maximum integer coefficient.
    max_order3 : int, optional. Default: 0
        Calculate 3P-MMRs up to this maximum order.
        If set to 0, only the exact order is considered.
    bounds : tuple[float, float, float, float], optional. Default: None
        The limits of the region (x_min, x_max, y_min, y_max).
        If not provided and radius is not None, the function will
        use the search radius around the point. If
    radius : float, optional. Default: 1e-3
        The radius of the search area around the point.
        If bounds are provided, the radius will be ignored.
    ret_point : bool, optional. Default: False
    verbose : bool, optional. Default: True
        Whether to print the optimization results.
    minimize_kwargs : dict, optional
        Additional arguments for the :py:func:scipy.optimize.minimize
        function.

    Returns
    -------
    resonance : list[int, int, int]
        The coefficients of the closest 3-body resonance.
    distance : float
        The distance to the closest 3-body resonance.
    """
    if bounds is None and radius is None:
        raise ValueError("Either bounds or radius must be provided.")

    if bounds is None:
        bounds = (
            max(a - radius, 1),
            a + radius,
            max(b - radius, 1),
            b + radius,
        )

    # Get the resonances in the specified area
    mmrs = mmrs_in_area(
        bounds,
        order3,
        max_coeff3,
        max_order3,
        mmr2b=False,
        verbose=verbose,
    )

    if not mmrs:
        raise ValueError("No 3-body mean-motion resonances found in the area.")

    # Initialize the minimum distance
    min_distance = np.inf
    closest_resonance = None

    # Find the closest 3-body resonance
    for resonance in mmrs:
        _, _, distance = mindist_mmr3b(a, b, resonance, **minimize_kwargs)
        if distance < min_distance:
            min_distance = distance
            closest_resonance = resonance

    if verbose:
        print(f"Closest 3-body mean-motion resonance: {closest_resonance}")
        print(f"Distance: {min_distance}")

    return closest_resonance, min_distance


def label_mmr2b(
    resonance: tuple,
    ax: plt.Axes = None,
    xaxis: bool = True,
    lims: tuple = None,
    warn: bool = True,
) -> plt.Axes:
    """Annotate a plot with the label of a 2-body resonance line.

    The label is placed where the resonance line crosses either the
    right or top axis of the plot. The resonance coefficients are
    displayed in a compact format. If the line does not cross these
    axes, a warning is printed.

    Parameters
    ----------
    resonance : tuple
        Coefficients of the resonance line (a, b) in the form a*x + b = 0.
    ax : matplotlib.axes.Axes, optional. Default: None
        The axis object on which the label will be placed.
        If not provided, the function will use the current axis.
    xaxis : bool, optional. Default: True
        Whether the resonance is along the x-axis (True) or y-axis (False).
    lims : tuple, optional. Default: None
        Custom axis limits in the format (x_min, x_max, y_min, y_max).
        If not provided, the function will use the current axis limits.
    warn : bool, optional. Default: True
        Whether to print a warning if the resonance does not cross
        the right or top axis.

    Returns
    -------
    ax : Matplotlib Axes
        The axis object with the annotations.
    """
    a, b = resonance  # Coefficients of the resonance line
    res = a / b

    # Get the current axis if not provided
    if ax is None:
        ax = plt.gca()

    # Get axis limits
    if lims:
        x_min, x_max, y_min, y_max = lims
    else:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

    # X of y axis
    if xaxis and x_min <= res <= x_max:
        # Define label
        label = f"{a}\n{b}"  # Compact label format
        # Normalize x-coordinate for axis transform
        res_ax = (res - x_min) / (x_max - x_min)
        ax.text(res_ax, 1.02, label, transform=ax.transAxes, ha="center")
    elif not xaxis and y_min <= res <= y_max:
        # Define label
        label = f"{a} {b}"  # Multi-line label format
        # Normalize y-coordinate for axis transform
        res_ax = (res - y_min) / (y_max - y_min)
        ax.text(1.02, res_ax, label, transform=ax.transAxes, va="center")
    elif warn:
        warnings.warn(
            f"{resonance} does not cross the right or top axis.",
            stacklevel=2,
        )

    return ax


def label_mmr3b(
    resonance: tuple,
    ax: plt.Axes = None,
    lims: tuple = None,
    warn: bool = True,
) -> plt.Axes:
    """Annotate a plot with the label of a resonance line.

    The label is placed where the resonance line crosses either the
    right or top axis of the plot. The resonance coefficients are
    displayed in a compact format. If the line does not cross these
    axes, a warning is printed.

    Parameters
    ----------
    resonance : tuple
        Coefficients of the resonance line (a, b, c) in
        the form a * x + b + c / y = 0.
    ax : matplotlib.axes.Axes, optional. Default: None
        The axis object on which the label will be placed.
        If not provided, the function will use the current axis.
    lims : tuple, optional. Default: None
        Custom axis limits in the format (x_min, x_max, y_min, y_max).
        If not provided, the function will use the current axis limits.
    warn : bool, optional. Default: True
        Whether to print a warning if the resonance does not cross
        the right or top axis.

    Returns
    -------
    ax : Matplotlib Axes
        The axis object with the annotations.
    """
    a, b, c = resonance  # Coefficients of the resonance line

    # Define the inverse of the line equation
    def rinv(y):
        """Calculate x for a given y using the line equation."""
        return -(b * y + c) / (a * y)

    # Get the current axis if not provided
    if ax is None:
        ax = plt.gca()

    # Get axis limits
    if lims:
        x_min, x_max, y_min, y_max = lims
    else:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

    # Calculate y-coordinate at x_max
    y = mmr3b(x_max, resonance)

    # Check crossing on the right axis
    if y_min <= y <= y_max:
        label = f"{a} {b} {c}"  # Compact label format
        # Normalize y-coordinate for axis transform
        y_ax = (y - y_min) / (y_max - y_min)
        ax.text(1.01, y_ax, label, transform=ax.transAxes, va="center")

    # Check crossing on the top axis
    elif x_min <= rinv(y_max) <= x_max:
        label = f"{a}\n{b}\n{c}"  # Multi-line label format
        x = rinv(y_max)
        # Normalize x-coordinate for axis transform
        x_ax = (x - x_min) / (x_max - x_min)
        ax.text(x_ax, 1.02, label, transform=ax.transAxes, ha="center")

    # If the line does not cross the right or top axis
    elif warn:
        warnings.warn(
            f"{resonance} does not cross the right or top axis.",
            stacklevel=2,
        )

    return ax


def plot_mmrs(
    bounds: Tuple[float, float, float, float] = None,
    order3: int = 0,
    max_coeff3: int = 10,
    max_order3: int = 0,
    mmr2b: bool = True,
    order2: int = 2,
    max_coeff2: int = 10,
    max_order2: int = 2,
    n_points: int = 1000,
    ax: plt.Axes = None,
    label_mmrs: bool = False,
    label_2mmrs: bool = False,
    **plot_kwargs,
):
    """Plot 3-body and 2-body mean-motion resonances in a phase-space region.

    This function plots 3-body mean-motion resonances (3P-MMRs) and optionally
    2-body mean-motion resonances (2P-MMRs) in a specified region of the phase
    space.

    Note
    ----
    The function will adjust the bounds to the axis limits if the axis object
    is provided. If the `label_mmrs` option is used, it is recommended to set
    the `xlim` and `ylim` before calling this function, or the labels may be
    placed outside the plot.

    Parameters
    ----------
    bounds : Tuple[float, float, float, float], optional. Default: None
        The limits of the region (x_min, x_max, y_min, y_max).
        If ax is provided, the bounds will be adjusted to the axis limits.
    order3 : int. Default: 0
        Exact order for 3P-MMRs.
    max_coeff3 : int, optional. Default: 10
        Calculate 3P-MMRs up to this maximum integer coefficient.
    max_order3 : int, optional. Default: 0
        Calculate 3P-MMRs up to this maximum order.
        If set to 0, only the exact order is considered (default: 10).
    mmr2b : bool, optional. Default: True
        Whether to compute 2P-MMRs.
    order2 : int, optional. Default: 2
        Exact order for 2P-MMRs.
    max_coeff2 : int, optional. Default: 10
        Calculate 2P-MMRs up to this maximum integer coefficient.
    max_order2 : int, optional. Default: 2
        Calculate 2P-MMRs up to this maximum order.
        If set to 0, only the exact order is considered.
    n_points : int, optional. Default: 500
        Number of points for the curve.
    ax : plt.Axes, optional. Default: None
        The axis object on which the plot will be drawn.
        If not provided, a new figure and axis will be created.
    label_mmrs : bool, optional. Default: False
        Whether to label the resonances on the plot.
        Recommended to set xlim and ylim before using this option,
        or the labels may be placed outside the plot. (default: False).
    label_2mmrs : bool, optional. Default: False
        Whether to label the 2-body resonances on the plot.
        Recommended to set xlim and ylim before using this option,
        or the labels may be placed outside the plot. (default: False).
    plot_kwargs : dict, optional
        Additional arguments for the plot function.

    Returns
    -------
    ax : Matplotlib Axes
        The axis object on which the plot was drawn.
    """
    # Get the current axis if not provided
    if ax is None:
        ax = plt.gca()

    if bounds is None:
        # Get the axis limits
        bounds = [
            ax.get_xlim()[0],
            ax.get_xlim()[1],
            ax.get_ylim()[0],
            ax.get_ylim()[1],
        ]

    # Get the resonances in the specified area
    mmrs = mmrs_in_area(
        bounds,
        order3,
        max_coeff3,
        max_order3,
        mmr2b,
        order2,
        max_coeff2,
        max_order2,
    )

    # Plot the resonances
    if mmr2b:
        mmr3 = mmrs[0]
        mmr2x = mmrs[1]
        mmr2y = mmrs[2]
    else:
        mmr3 = mmrs
        mmr2x = mmr2y = []

    # Plot the 2P-MMRs
    for r2x in mmr2x:
        ax.axvline(r2x[0] / r2x[1], color="k", linestyle="--")
        if label_2mmrs:
            label_mmr2b(r2x, ax, xaxis=True, warn=True)

    # Plot the 2P-MMRs
    for r2y in mmr2y:
        ax.axhline(r2y[0] / r2y[1], color="k", linestyle="--")
        if label_2mmrs:
            label_mmr2b(r2y, ax, xaxis=False, warn=True)

    # Plot the 3P-MMRs
    for r3 in mmr3:
        x = np.linspace(bounds[0], bounds[1], n_points)
        curve = mmr3b(x, r3)
        ax.plot(x, curve, **plot_kwargs)
        if label_mmrs:
            label_mmr3b(r3, ax, warn=True)

    return ax
