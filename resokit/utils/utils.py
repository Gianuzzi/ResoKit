#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2025, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Module with additional functions and tools for medley tasks."""

# =============================================================================
# IMPORTS
# =============================================================================

from fractions import Fraction
from typing import Callable, Tuple, Union

from numpy import pi, sqrt

from resokit.units import MKS

# =============================================================================
# CONSTANTS
# =============================================================================

# Maximum error tolerance for the float-to-fraction conversion
MIN_F2F_ERROR = 1e-10  # It is a very small number
MAX_F2F_ITER = 12  # Maximum number of iterations

# =============================================================================
# FUNCTIONS
# =============================================================================


def float_to_fraction(
    value: float,
    max_iter: int = None,
    max_error: float = None,
    as_fraction: bool = False,
    stop_func: Callable = None,
    verbose: bool = True,
) -> Union[Fraction, Tuple[int, int]]:
    """Calculate the continued fraction approximation of a value.

    Parameters
    ----------
    value : float
        Value to approximate.
    max_iter : int, optional.
        Maximum number of terms to use in the continued fraction expansion.
    max_error : float, optional.
        Maximum relative error tolerance for the approximation.
    as_fraction : bool, optional. Default: False
        Whether to return the result as a Fraction object.
    stop_func : callable, optional
        Function to use as a stopping criterion for the approximation.
        Takes the numerator and denominator of the current approximation as
        arguments and returns a boolean indicating whether to stop.
        If STOP is reached, the function will return the
        previous approximation.
    verbose : bool, optional. Default: True
        Whether to print the intermediate results of the calculation.

    Returns
    -------
    Union[Fraction, Tuple[int, int]]
        Tuple with the numerator and denominator of the best approximation,
        or a Fraction object if `as_fraction` is True.
    """
    # Check input values
    if max_iter is None and max_error is None and stop_func is None:
        raise ValueError(
            "At least one of max_iter or max_error or stop_func must be set."
        )

    if not isinstance(value, (int, float)):
        raise TypeError("value must be a number.")

    if max_iter is not None and not isinstance(max_iter, int):
        raise TypeError("max_iter must be an integer.")

    if max_error is not None and not isinstance(max_error, (int, float)):
        raise TypeError("max_error must be a number.")

    if stop_func is not None:  # Check stop_func

        has_stop = True  # Stop function is set

        if callable(stop_func):  # Check if it is callable
            try:  # Check if it returns a boolean
                if not isinstance(stop_func(1, 1), bool):
                    raise TypeError("stop_func must return a boolean.")
            except Exception as e:  # Any error
                print(e)
                raise TypeError(
                    "stop_func must be able to return a boolean "
                    + "from the numerator and denominator."
                )
        else:  # Not callable
            raise TypeError("stop_func must be a callable.")
    else:

        has_stop = False  # Stop function is not set

        #  Default stop function: Keep going
        def stop_func(n, d):
            return False  # Keep going

    # Define max error
    max_error = abs(max_error) if max_error is not None else None

    # Initialize variables
    z = value
    a = []
    numer = []
    denom = []
    i = 0

    # Print the initial value
    if verbose:
        print(f"Approximating float {value:.6f} as a continued fraction:")

    while True:
        a_i = int(z)
        a.append(a_i)
        z = 1 / (z - a_i) if z != a_i else 0

        if i == 0:
            numer.append(a_i)
            denom.append(1)
        elif i == 1:
            numer.append(a_i * numer[i - 1] + 1)
            denom.append(a_i)
        else:
            numer.append(a_i * numer[i - 1] + numer[i - 2])
            denom.append(a_i * denom[i - 1] + denom[i - 2])

        # Calculate the approximation
        approx_value = numer[i] / denom[i]

        # Calculate the relative error
        error = abs((approx_value - value) / value)

        # Check if the stop function is reached
        is_stop = stop_func(numer[i], denom[i])

        # Print the intermediate results
        if verbose:
            print(
                f" Iter {i + 1:>2d}: {numer[i]:>3d}/{denom[i]:<3d} "
                + f"-> {approx_value:.6f} "
                + f"(error: {error:.2e})"
                + (f" -> STOP: {is_stop}" if has_stop else "")
            )

        # Check stopping criteria
        if (max_error is not None and error < max_error) or (  # Relative error
            max_iter is not None and i + 1 >= max_iter  # Max iterations
        ):
            break

        # Check if the stop function is reached
        if is_stop:
            i -= 1  # Go back one step before
            break

        # Check if the error is below the minimum
        if error < MIN_F2F_ERROR:  # Minimum error (close enough)
            if verbose:  # Print a warning
                print(f"Minimum function error reached: {MIN_F2F_ERROR}")
            break

        # Check if the maximum number of iterations is reached
        if i >= MAX_F2F_ITER:
            if verbose:  # Print a warning
                print(f"Maximum number of iterations reached: {MAX_F2F_ITER}")
            break

        i += 1

    # Return the best approximation as a Fraction object
    if as_fraction:
        return Fraction(numer[i], denom[i])

    return numer[i], denom[i]


# Below are the functions used in ResokitPlanet (and ResokitSystem) class, but
# could also be used by the user. They are not all part of the public API, but
# they are still useful for the user.


def calc_period(a: float, m_star: float, m_planet) -> float:
    r"""Calculate the orbital period of a planet.

    Equation:
        :math:`P = 2 \pi \sqrt{\dfrac{a^3}{G (m_\star + m_p)}}`

    Parameters
    ----------
    a : float
        Semi-major axis of the planet, in AU.
    m_star : float
        Mass of the star, in solar masses.
    m_planet : float
        Mass of the planet, in Jupiter masses.

    Returns
    -------
    float
        Orbital period of the planet, in days.
    """
    ene = sqrt(
        MKS["G"]
        * (m_star * MKS["ms"] + m_planet * MKS["mj"])
        / (a * MKS["au"]) ** 3
    )

    return 2 * pi / ene / MKS["day"]


def calc_period_with_errors(
    a: float,
    a_err_min: float,
    a_err_max: float,
    m_star: float,
    m_star_err_min: float,
    m_star_err_max: float,
    m_planet: float,
    m_planet_err_min: float,
    m_planet_err_max: float,
    err_method: int = -1,
) -> Tuple[float, float, float]:
    """Calculate the orbital period and its error using error propagation.

    Parameters
    ----------
    a : float
        Semi-major axis of the planet, in AU.
    a_err_min : float
        Minimum error in the semi-major axis, in AU.
    a_err_max : float
        Maximum error in the semi-major axis, in AU.
    m_star : float
        Mass of the star, in solar masses.
    m_star_err_min : float
        Minimum error in the star's mass, in solar masses.
    m_star_err_max : float
        Maximum error in the star's mass, in solar masses.
    m_planet : float
        Mass of the planet, in Jupiter masses.
    m_planet_err_min : float
        Minimum error in the planet's mass, in Jupiter masses.
    m_planet_err_max : float
        Maximum error in the planet's mass, in Jupiter masses.
    err_method : int, optional. Default: -1
        Error method to use:
            - <=0: No error. Return the period and 0 error.
            - 1: Extremes. Estimate the period at the extreme values of
                    each parameter and retrieve the errors from the difference.
            - 2: Max propagation. Assume each parameters follows a normal
                    distribution with sigma = sigma = max(err_min, err_max).
            - 3: Centred propagation. Assume each parameters follows a normal
                    distribution with sigma = (err_min + err_max) / 2.

    Returns
    -------
    Tuple[float, float, float]
        Orbital period and its minimum and maximum errors, in days.
    """
    # Switch for the error propagation method
    if err_method in [-1, 0]:
        return calc_period(a, m_star, m_planet), 0, 0
    elif err_method == 1:
        period = calc_period(a, m_star, m_planet)
        period_min = calc_period(
            a - a_err_min, m_star + m_star_err_max, m_planet + m_planet_err_max
        )
        period_max = calc_period(
            a + a_err_max, m_star - m_star_err_min, m_planet - m_planet_err_min
        )
        period_err_min = abs(period - period_min)
        period_err_max = abs(period - period_max)
        return period, period_err_min, period_err_max
    elif err_method == 2:
        a_err = max(a_err_min, a_err_max) * MKS["au"]
        m_star_err = max(m_star_err_min, m_star_err_max) * MKS["ms"]
        m_planet_err = max(m_planet_err_min, m_planet_err_max) * MKS["mj"]
    elif err_method == 3:
        a_err = (a_err_min + a_err_max) * 0.5 * MKS["au"]
        m_star_err = (m_star_err_min + m_star_err_max) * 0.5 * MKS["ms"]
        m_planet_err = (m_planet_err_min + m_planet_err_max) * 0.5 * MKS["mj"]
    else:
        raise ValueError("Invalid error propagation method.")

    # Calculate the period (in days)
    period = calc_period(a, m_star, m_planet)

    # Partial derivatives for error propagation
    dperiod_dm_star = (
        -period
        * MKS["day"]
        / (2 * ((m_star * MKS["ms"]) + (m_planet * MKS["mj"])))
    )
    dperiod_dm_planet = dperiod_dm_star  # Same derivative as the star
    dperiod_da = -6 * pi**2 / ((period * MKS["day"]) * (a * MKS["au"]))

    # Errors
    period_err = (
        sqrt(
            (dperiod_da * a_err) ** 2
            + (dperiod_dm_star * m_star_err) ** 2
            + (dperiod_dm_planet * m_planet_err) ** 2
        )
        / MKS["day"]
    )  # In days

    return period, period_err, period_err  # Same error for min and max


def calc_a(period: float, m_star: float, m_planet: float) -> float:
    r"""Calculate the semi-major axis of a planet.

    Equation:
        :math:`a = \left(\dfrac{G (m_\star + m_p)}{4 \pi^2 P^2}\right)^{1/3}`

    Parameters
    ----------
    period : float
        Orbital period of the planet, in days.
    m_star : float
        Mass of the star, in solar masses.
    m_planet : float
        Mass of the planet, in Jupiter masses.

    Returns
    -------
    float
        Semi-major axis of the planet, in AU.
    """
    ene = 2 * pi / period / MKS["day"]

    return (
        MKS["G"] * (m_star * MKS["ms"] + m_planet * MKS["mj"]) / ene**2
    ) ** (1 / 3) / MKS["au"]


def calc_a_with_errors(
    period: float,
    period_err_min: float,
    period_err_max: float,
    m_star: float,
    m_star_err_min: float,
    m_star_err_max: float,
    m_planet: float,
    m_planet_err_min: float,
    m_planet_err_max: float,
    err_method: int = -1,
) -> Tuple[float, float, float]:
    """Calculate the semi-major axis and its error using error propagation.

    Parameters
    ----------
    period : float
        Orbital period of the planet, in days.
    period_err_min : float
        Minimum error in the orbital period, in days.
    period_err_max : float
        Maximum error in the orbital period, in days.
    m_star : float
        Mass of the star, in solar masses.
    m_star_err_min : float
        Minimum error in the star's mass, in solar masses.
    m_star : float
        Maximum error in the star's mass, in solar masses.
    m_planet : float
        Mass of the planet, in Jupiter masses.
    m_planet_err_min : float
        Minimum error in the planet's mass, in Jupiter masses.
    m_planet_err_max : float
        Maximum error in the planet's mass, in Jupiter masses.
    err_method : int, optional. Default: -1
        Error method to use:
            - <=0: No error. Return the period and 0 error.
            - 1: Extremes. Estimate the period at the extreme values of
                    each parameter and retrieve the errors from the difference.
            - 2: Max propagation. Assume each parameters follows a normal
                    distribution with sigma = max(err_min, err_max).
            - 3: Centred propagation. Assume each parameters follows a normal
                    distribution with sigma = (err_min + err_max) / 2.

    Returns
    -------
    Tuple[float, float, float]
        Semi-major axis and its minimum and maximum errors, in AU.
    """
    # Switch for the error propagation method
    if err_method in [-1, 0]:
        return calc_a(period, m_star, m_planet), 0, 0
    elif err_method == 1:
        a = calc_a(period, m_star, m_planet)
        a_min = calc_a(
            period - period_err_min,
            m_star - m_star_err_min,
            m_planet - m_planet_err_min,
        )
        a_max = calc_a(
            period + period_err_max,
            m_star + m_star_err_max,
            m_planet + m_planet_err_max,
        )
        a_err_min = abs(a - a_min)
        a_err_max = abs(a - a_max)
        return a, a_err_min, a_err_max
    elif err_method == 2:
        period_err = max(period_err_min, period_err_max) * MKS["day"]
        m_star_err = max(m_star_err_min, m_star_err_max) * MKS["ms"]
        m_planet_err = max(m_planet_err_min, m_planet_err_max) * MKS["mj"]
    elif err_method == 3:
        period_err = (period_err_min + period_err_max) * 0.5 * MKS["day"]
        m_star_err = (m_star_err_min + m_star_err_max) * 0.5 * MKS["ms"]
        m_planet_err = (m_planet_err_min + m_planet_err_max) * 0.5 * MKS["mj"]
    else:
        raise ValueError("Invalid error propagation method.")

    # Calculate the semi-major axis (in AU)
    a = calc_a(period, m_star, m_planet)

    # Partial derivatives for error propagation
    da_dm_star = (
        MKS["G"]
        * (period * MKS["day"]) ** 2
        / (12 * pi**2 * (a * MKS["au"]) ** 2)
    )
    da_dm_planet = da_dm_star  # Same derivative as the star
    da_dperiod = 2 / 3 * (a * MKS["au"]) / (period * MKS["day"])

    # Errors
    a_err = (
        sqrt(
            (da_dperiod * period_err) ** 2
            + (da_dm_star * m_star_err) ** 2
            + (da_dm_planet * m_planet_err) ** 2
        )
        / MKS["au"]
    )  # In AU

    return a, a_err, a_err  # Same error for min and max


def calc_hill_radius(
    a: float, e: float, m_star: float, m_planet: float
) -> float:
    r"""Calculate the Hill radius of a planet.

    Equation:
        :math:`r_H = a (1 - e) \left(\dfrac{m_p}
        {3 (m_\star + m_p)}\right)^{1/3}`

    Parameters
    ----------
    a : float
        Semi-major axis of the planet, in AU.
    e : float
        Eccentricity of the planet.
    m_star : float
        Mass of the star, in solar masses.
    m_planet : float
        Mass of the planet, in Jupiter masses.

    Returns
    -------
    float
        Hill radius of the planet, in AU.
    """
    return (
        a
        * (1 - e)
        * (
            m_planet
            * MKS["mj"]
            / (3 * (m_star * MKS["ms"] + m_planet * MKS["mj"]))
        )
        ** (1 / 3.0)
    )


def calc_hill_radius_with_errors(
    a: float,
    a_err_min: float,
    a_err_max: float,
    e: float,
    e_err_min: float,
    e_err_max: float,
    m_star: float,
    m_star_err_min: float,
    m_star_err_max: float,
    m_planet: float,
    m_planet_err_min: float,
    m_planet_err_max: float,
    err_method: int = -1,
) -> Tuple[float, float, float]:
    """Calculate the Hill radius and its error using error propagation.

    Parameters
    ----------
    a : float
        Semi-major axis of the planet, in AU.
    a_err_min : float
        Minimum error in the semi-major axis, in AU.
    a_err_max : float
        Maximum error in the semi-major axis, in AU.
    e : float
        Eccentricity of the planet.
    e_err_min : float
        Minimum error in the eccentricity.
    e_err_max : float
        Maximum error in the eccentricity.
    m_star : float
        Mass of the star, in solar masses.
    m_star_err_min : float
        Minimum error in the star's mass, in solar masses.
    m_star_err_max : float
        Maximum error in the star's mass, in solar masses.
    m_planet : float
        Mass of the planet, in Jupiter masses.
    m_planet_err_min : float
        Minimum error in the planet's mass, in Jupiter masses.
    m_planet_err_max : float
        Maximum error in the planet's mass, in Jupiter masses.
    err_method : int, optional. Default: -1
        Error method to use:
            - <=0: No error. Return the period and 0 error.
            - 1: Extremes. Estimate the period at the extreme values of
                    each parameter and retrieve the errors from the difference.
            - 2: Max propagation. Assume each parameters follows a normal
                    distribution with sigma = sigma = max(err_min, err_max).
            - 3: Centred propagation. Assume each parameters follows a normal
                    distribution with sigma = (err_min + err_max) / 2.


    Returns
    -------
    Tuple[float, float, float]
        Hill radius and its minimum and maximum errors, in AU.
    """
    # Switch for the error propagation method
    if err_method in [-1, 0]:
        return calc_hill_radius(a, e, m_star, m_planet), 0, 0
    elif err_method == 1:
        hill = calc_hill_radius(a, e, m_star, m_planet)
        hill_min = calc_hill_radius(
            a - a_err_min,
            e - e_err_min,
            m_star + m_star_err_max,
            m_planet + m_planet_err_max,
        )
        hill_max = calc_hill_radius(
            a + a_err_max,
            e + e_err_max,
            m_star - m_star_err_min,
            m_planet - m_planet_err_min,
        )
        hill_err_min = abs(hill - hill_min)
        hill_err_max = abs(hill - hill_max)
        return hill, hill_err_min, hill_err_max
    elif err_method == 2:
        a_err = max(a_err_min, a_err_max) * MKS["au"]
        e_err = max(e_err_min, e_err_max)
        m_star_err = max(m_star_err_min, m_star_err_max) * MKS["ms"]
        m_planet_err = max(m_planet_err_min, m_planet_err_max) * MKS["mj"]
    elif err_method == 3:
        a_err = (a_err_min + a_err_max) * 0.5 * MKS["au"]
        e_err = (e_err_min + e_err_max) * 0.5
        m_star_err = (m_star_err_min + m_star_err_max) * 0.5 * MKS["ms"]
        m_planet_err = (m_planet_err_min + m_planet_err_max) * 0.5 * MKS["mj"]
    else:
        raise ValueError("Invalid error propagation method.")

    # Calculate the Hill radius (in AU)
    hill = calc_hill_radius(a, e, m_star, m_planet)

    # Auxiliary total mass
    total_mass = m_star * MKS["ms"] + m_planet * MKS["mj"]

    # Partial derivatives for error propagation
    dhill_da = hill / (a * MKS["au"])
    dhill_de = -hill / (1 - e)
    dhill_dm_star = -hill / (3 * total_mass)
    dhill_dm_planet = (
        -dhill_dm_star * (m_star * MKS["ms"]) / (m_planet * MKS["mj"])
    )

    # Errors
    hill_err = sqrt(
        (dhill_da * a_err) ** 2
        + (dhill_de * e_err) ** 2
        + (dhill_dm_star * m_star_err) ** 2
        + (dhill_dm_planet * m_planet_err) ** 2
    )

    return hill, hill_err, hill_err  # Same error for min and max


def calc_sum_with_errors(
    *vals: Tuple[float, float, float], err_method: int = -1
) -> Tuple[float, float, float]:
    r"""Calculate the sum of values with errors using error propagation.

    Equation:
        :math:`\Sigma = \sum_{i=1}^{N} x_i`

    Parameters
    ----------
    vals : Tuple[Tuple[float, float, float], ...]
        Tuple with the values and their minimum and maximum errors.
    err_method : int, optional. Default: -1
        Error method to use:
            - <=0: No error. Return the sum and 0 error.
            - 1: Extremes. Estimate the sum at the extreme values of
                    each parameter and retrieve the errors from the difference.
            - 2: Max propagation. Assume each parameters follows a normal
                    distribution with sigma = sigma = max(err_min, err_max).
            - 3: Centred propagation. Assume each parameters follows a normal
                    distribution with sigma = (err_min + err_max) / 2.

    Returns
    -------
    Tuple[float, float]
        Sum of the values and its error.
    """
    # Switch for the error propagation method
    if err_method in [-1, 0]:
        return sum(val[0] for val in vals), 0.0, 0.0
    elif err_method == 1:
        suma = sum(val[0] for val in vals)
        suma_min = sum(val[0] - val[1] for val in vals)
        suma_max = sum(val[0] + val[2] for val in vals)
        suma_err_min = abs(suma - suma_min)
        suma_err_max = abs(suma - suma_max)
        return suma, suma_err_min, suma_err_max
    elif err_method == 2:
        vals_err = [max(val[1], val[2]) for val in vals]
    elif err_method == 3:
        vals_err = [(val[1] + val[2]) * 0.5 for val in vals]
    else:
        raise ValueError("Invalid error propagation method.")

    # Calculate the sum
    suma = sum(val[0] for val in vals)

    # Partial derivatives for error propagation
    dsuma_dvals = [1.0 for _ in vals]

    # Errors
    suma_err = sqrt(
        sum((dsuma_dvals[i] * vals_err[i]) ** 2 for i in range(len(vals)))
    )

    return suma, suma_err, suma_err  # Same error for min and max
