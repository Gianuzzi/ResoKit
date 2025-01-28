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

"""Module with internal utility functions for the ResoKit package."""

# =============================================================================
# IMPORTS
# =============================================================================

import platform
import sys
from fractions import Fraction
from types import MappingProxyType
from typing import Iterable, Tuple, Union

from numpy import pi, sqrt

from resokit import __version__ as version
from resokit.units import AU, DAY, G, M_JUP, M_SUN

# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_METADATA = MappingProxyType(
    {
        "ResoKit": version,
        "author": "Emmanuel Gianuzzi",
        "author_email": "egianuzzi@unc.edu.ar",
        "affiliation": "FAMAF-IATE-OAC-CONICET",
        "platform": platform.platform(),
        "system_encoding": sys.getfilesystemencoding(),
        "python": sys.version,
        "license": "MIT",
    }
)

# =============================================================================
# CONSTANTS
# =============================================================================


# EU column to resokit
_EU_MAPPING = MappingProxyType(
    {
        "name": "name",
        "mass": "mass",
        "mass_error_min": "mass_err_min",
        "mass_error_max": "mass_err_max",
        "mass_sini": "mass_sin_i",
        "mass_sini_error_min": "mass_sin_i_err_min",
        "mass_sini_error_max": "mass_sin_i_err_max",
        "radius": "radius",
        "radius_error_min": "radius_err_min",
        "radius_error_max": "radius_err_max",
        "orbital_period": "P",
        "orbital_period_error_min": "P_err_min",
        "orbital_period_error_max": "P_err_max",
        "semi_major_axis": "a",
        "semi_major_axis_error_min": "a_err_min",
        "semi_major_axis_error_max": "a_err_max",
        "eccentricity": "e",
        "eccentricity_error_min": "e_err_min",
        "eccentricity_error_max": "e_err_max",
        "inclination": "inc",
        "inclination_error_min": "inc_err_min",
        "inclination_error_max": "inc_err_max",
        "omega": "w",
        "omega_error_min": "w_err_min",
        "omega_error_max": "w_err_max",
        "tperi": "tperi",
        "tperi_error_min": "tperi_err_min",
        "tperi_error_max": "tperi_err_max",
        # Star columns
        "star_name": "star_name",
        "star_mass": "star_mass",
        "star_mass_error_min": "star_mass_err_min",
        "star_mass_error_max": "star_mass_err_max",
        "star_radius": "star_radius",
        "star_radius_error_min": "star_radius_err_min",
        "star_radius_error_max": "star_radius_err_max",
        # System/Star columns
        "star_distance": "star_dist",
        "star_distance_error_min": "star_dist_err_min",
        "star_distance_error_max": "star_dist_err_max",
        # Metadata columns
        "publication": "reference",
        "updated": "rowupdate",
        "discovered": "disc_year",
        "detection_type": "disc_method",
    }
)

# Nasa columns to resokit
_NASA_MAPPING = MappingProxyType(
    {
        "pl_name": "name",
        "pl_massj": "mass",
        "pl_massjerr1": "mass_err_min",
        "pl_massjerr2": "mass_err_max",
        "pl_msinij": "mass_sin_i",
        "pl_msinijerr1": "mass_sin_i_err_min",
        "pl_msinijerr2": "mass_sin_i_err_max",
        "pl_radj": "radius",
        "pl_radjerr1": "radius_err_min",
        "pl_radjerr2": "radius_err_max",
        "pl_orbper": "P",
        "pl_orbpererr1": "P_err_min",
        "pl_orbpererr2": "P_err_max",
        "pl_orbsmax": "a",
        "pl_orbsmaxerr1": "a_err_min",
        "pl_orbsmaxerr2": "a_err_max",
        "pl_orbeccen": "e",
        "pl_orbeccenerr1": "e_err_min",
        "pl_orbeccenerr2": "e_err_max",
        "pl_orbincl": "inc",
        "pl_orbinclerr1": "inc_err_min",
        "pl_orbinclerr2": "inc_err_max",
        "pl_orblper": "w",
        "pl_orblpererr1": "w_err_min",
        "pl_orblpererr2": "w_err_max",
        "pl_orbtper": "tperi",
        "pl_orbtpererr1": "tperi_err_min",
        "pl_orbtpererr2": "tperi_err_max",
        # Star columns
        "hostname": "star_name",
        "st_mass": "star_mass",
        "st_masserr1": "star_mass_err_min",
        "st_masserr2": "star_mass_err_max",
        "st_rad": "star_radius",
        "st_raderr1": "star_radius_err_min",
        "st_raderr2": "star_radius_err_max",
        # System/Star columns
        "sy_dist": "star_dist",
        "sy_disterr1": "star_dist_err_min",
        "sy_disterr2": "star_dist_err_max",
        # Metadata columns
        "pl_refname": "reference",
        "rowupdate": "rowupdate",
        "disc_year": "disc_year",
        "discoverymethod": "disc_method",
        # Other columns
        "sy_snum": "n_stars",
        "sy_pnum": "n_planets",
        "pl_controv_flag": "controversial",
        "default_flag": "default_set",
        "circumbinary_flag": "circumbinary",
    }
)

# Column mappings for the different data sources
MAPPINGS = MappingProxyType({"eu": _EU_MAPPING, "nasa": _NASA_MAPPING})


# Default attributes for resokit planet
RESO_PL_TYPES = MappingProxyType(
    {
        "name": "object",
        "mass": "float64",
        "mass_err_min": "float64",
        "mass_err_max": "float64",
        "mass_sin_i": "float64",
        "mass_sin_i_err_min": "float64",
        "mass_sin_i_err_max": "float64",
        "radius": "float64",
        "radius_err_min": "float64",
        "radius_err_max": "float64",
        "P": "float64",
        "P_err_min": "float64",
        "P_err_max": "float64",
        "a": "float64",
        "a_err_min": "float64",
        "a_err_max": "float64",
        "e": "float64",
        "e_err_min": "float64",
        "e_err_max": "float64",
        "inc": "float64",
        "inc_err_min": "float64",
        "inc_err_max": "float64",
        "w": "float64",
        "w_err_min": "float64",
        "w_err_max": "float64",
        "tperi": "float64",
        "tperi_err_min": "float64",
        "tperi_err_max": "float64",
    }
)

# Default attributes for resokit star
RESO_SR_TYPES = MappingProxyType(
    {
        "star_name": "object",
        "star_mass": "float64",
        "star_mass_err_min": "float64",
        "star_mass_err_max": "float64",
        "star_radius": "float64",
        "star_radius_err_min": "float64",
        "star_radius_err_max": "float64",
        "star_dist": "float64",
        "star_dist_err_min": "float64",
        "star_dist_err_max": "float64",
    }
)

# Default attributes for resokit object (star and/or planet)
# Note: 'default_set' and 'circumbinary' are not present in the EU dataset
# Note: 'controversial' is not present in the NASA dataset
RESO_OB_TYPES = MappingProxyType(
    {
        "reference": "object",
        "rowupdate": "object",
        "disc_year": "int64",
        "disc_method": "object",
        "n_stars": "int64",
        "n_planets": "int64",
        "controversial": "int64",
        "default_set": "int64",
        "circumbinary": "int64",
    }
)

# Default attributes for resokit dataset
RESO_DTYPES = MappingProxyType(
    {**RESO_PL_TYPES, **RESO_SR_TYPES, **RESO_OB_TYPES}
)

# Maximum error tolerance for the float-to-fraction conversion
MIN_F2F_ERROR = 1e-10  # It is a very small number
MAX_F2F_ITER = 12  # Maximum number of iterations

# =============================================================================
# FUNCTIONS
# =============================================================================


def assert_module_imported(
    imported: bool, module_name: str, message: str = ""
):
    """Assert that the specified module is imported.

    Parameters
    ----------
    imported : bool
        Boolean indicating whether the module is imported.
    module_name : str
        Name of the module to check.
    message : str, optional. Default: ""
        Error message to display if the module is not imported.
    """
    if not imported:
        raise ImportError(
            f"{module_name} is required for this function. {message}"
        )

    return True  # Module is imported


def float_to_fraction(
    value: float,
    max_iter: int = None,
    max_error: float = None,
    as_fraction: bool = False,
    stop_func: callable = None,
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


def parse_to_iter(value: any, to: type = list) -> Iterable:
    """Parse a value to an iterable if it is not already.

    Parameters
    ----------
    value : Any
        Value to parse.
    to : type, optional. Default: list
        Type of iterable to return.
        If not None, to(value) will be called.

    Returns
    -------
    Iterable
        Parsed value as an iterable.
    """
    # If it is a string (already iterable) or not an iterable, return a list
    if isinstance(value, str) or not isinstance(value, Iterable):
        return [value]
    elif to is not None:
        return to(value)

    return value


def parse_name(name: str, force: bool = False) -> str:
    """Parse a name to a more versatile format.

    Steps:
    1) The trailing whitespaces are removed.
    2) The trailing " A" or " B" or " AB" or " (AB)" or "(AB)"
    are removed.
    2.5) If force is `True`, removes (AB) from the middle of the name.
    3) The name is converted to lowercase.
    4) All whitespaces and hyphens are removed.

    Parameters
    ----------
    name : str
        Object name.

    Returns
    -------
    str
        Name in a more versatile format.
    """
    # Remove the trailing whitespaces
    name = name.strip()

    # Remove the trailing " A" or " B" or " AB" or " (AB)" or "(AB)"
    # Only if it is at the end of the name
    if name.endswith(" A") or name.endswith(" B") or name.endswith(" AB"):
        name = name[:-2]
    elif name.endswith("(AB)") or name.endswith(" (AB)"):
        name = name[:-4]

    # Remove (AB) from the middle of the name
    if force:
        name = name.replace("(AB)", "")

    # Convert the name to lowercase
    name = name.lower()

    # Remove all whitespaces and hyphens
    name = name.replace(" ", "").replace("-", "")

    return name


# Below are the functions used in ResokitPlanet (and ResokitSystem) class, but
# could also be used by the user. They are not part of the public API, but they
# are still useful for the user.


def calc_period(a: float, m_star: float, m_planet) -> float:
    """Calculate the orbital period of a planet.

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
    ene = sqrt(G * (m_star * M_SUN + m_planet * M_JUP) / (a * AU) ** 3)

    return 2 * pi / ene / DAY


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
    err_method: int = 0,
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
    err_method : int, optional. Default: 0
        Error method to use:
            - <=0: No error. Return the period and 0 error.
            - 1: Extremes. Estimate the period at the extreme values of
                    each parameter and retrieve the errors from the difference.
            - 2: Max propagation. Assume each parameters follows a normal
                    distribution with sigma = err_max.
            - 3: Centred propagation. Assume each parameters follows a normal
                    distribution with sigma = (err_min + err_max) / 2.
            - 4: Deviated propagation. Assume each parameters follows a normal
                    distribution with sigma = (err_max + err_min) / 2, but the
                    mean is at ((val + err_min) + (val + err_max)) / 2.

    Returns
    -------
    Tuple[float, float, float]
        Orbital period and its minimum and maximum errors, in days.
    """
    # Switch for the error propagation method
    if err_method <= 0:
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
        a_err = max(a_err_min, a_err_max) * AU
        m_star_err = max(m_star_err_min, m_star_err_max) * M_SUN
        m_planet_err = max(m_planet_err_min, m_planet_err_max) * M_JUP
    elif err_method == 3 or err_method == 4:
        a_err = (a_err_min + a_err_max) * 0.5 * AU
        m_star_err = (m_star_err_min + m_star_err_max) * 0.5 * M_SUN
        m_planet_err = (m_planet_err_min + m_planet_err_max) * 0.5 * M_JUP
        if err_method == 4:
            a = (a - a_err_min + a + a_err_max) * 0.5
            m_star = (m_star - m_star_err_min + m_star + m_star_err_max) * 0.5
            m_planet = (
                m_planet - m_planet_err_min + m_planet + m_planet_err_max
            ) * 0.5
    else:
        raise ValueError("Invalid error propagation method.")

    # Calculate the period (in days)
    period = calc_period(a, m_star, m_planet)

    # Partial derivatives for error propagation
    dperiod_dm_star = (
        -period * DAY / (2 * ((m_star * M_SUN) + (m_planet * M_JUP)))
    )
    dperiod_dm_planet = dperiod_dm_star  # Same derivative as the star
    dperiod_da = -6 * pi**2 / ((period * DAY) * (a * AU))

    # Errors
    period_err = (
        sqrt(
            (dperiod_da * a_err) ** 2
            + (dperiod_dm_star * m_star_err) ** 2
            + (dperiod_dm_planet * m_planet_err) ** 2
        )
        / DAY
    )  # In days

    return period, period_err, period_err  # Same error for min and max


def calc_a(period: float, m_star: float, m_planet: float) -> float:
    """Calculate the semi-major axis of a planet.

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
    ene = 2 * pi / period / DAY

    return (G * (m_star * M_SUN + m_planet * M_JUP) / ene**2) ** (1 / 3) / AU


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
    err_method: int = 0,
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
    err_method : int, optional. Default: 0
        Error method to use:
            - <=0: No error. Return the period and 0 error.
            - 1: Extremes. Estimate the period at the extreme values of
                    each parameter and retrieve the errors from the difference.
            - 2: Max propagation. Assume each parameters follows a normal
                    distribution with sigma = err_max.
            - 3: Centred propagation. Assume each parameters follows a normal
                    distribution with sigma = (err_min + err_max) / 2.
            - 4: Deviated propagation. Assume each parameters follows a normal
                    distribution with sigma = (err_max + err_min) / 2, but the
                    mean is at ((val + err_min) + (val + err_max)) / 2.

    Returns
    -------
    Tuple[float, float, float]
        Semi-major axis and its minimum and maximum errors, in AU.
    """
    # Switch for the error propagation method
    if err_method <= 0:
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
        period_err = max(period_err_min, period_err_max) * DAY
        m_star_err = max(m_star_err_min, m_star_err_max) * M_SUN
        m_planet_err = max(m_planet_err_min, m_planet_err_max) * M_JUP
    elif err_method == 23 or err_method == 4:
        period_err = (period_err_min + period_err_max) * 0.5 * DAY
        m_star_err = (m_star_err_min + m_star_err_max) * 0.5 * M_SUN
        m_planet_err = (m_planet_err_min + m_planet_err_max) * 0.5 * M_JUP
        if err_method == 3:
            period = (period - period_err_min + period + period_err_max) * 0.5
            m_star = (m_star - m_star_err_min + m_star + m_star_err_max) * 0.5
            m_planet = (
                m_planet - m_planet_err_min + m_planet + m_planet_err_max
            ) * 0.5
    else:
        raise ValueError("Invalid error propagation method.")

    # Calculate the semi-major axis (in AU)
    a = calc_a(period, m_star, m_planet)

    # Partial derivatives for error propagation
    da_dm_star = G * (period * DAY) ** 2 / (12 * pi**2 * (a * AU) ** 2)
    da_dm_planet = da_dm_star  # Same derivative as the star
    da_dperiod = 2 / 3 * (a * AU) / (period * DAY)

    # Errors
    a_err = (
        sqrt(
            (da_dperiod * period_err) ** 2
            + (da_dm_star * m_star_err) ** 2
            + (da_dm_planet * m_planet_err) ** 2
        )
        / AU
    )  # In AU

    return a, a_err, a_err  # Same error for min and max


def hill_radius(a: float, e: float, m_star: float, m_planet: float) -> float:
    """Calculate the Hill radius of a planet.

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
        * (m_planet * M_JUP / (3 * (m_star * M_SUN + m_planet * M_JUP)))
        ** (1 / 3.0)
    )


def hill_radius_with_errors(
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
    err_method: int = 0,
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
    err_method : int, optional. Default: 0
        Error method to use:
            - <=0: No error. Return the period and 0 error.
            - 1: Extremes. Estimate the period at the extreme values of
                    each parameter and retrieve the errors from the difference.
            - 2: Max propagation. Assume each parameters follows a normal
                    distribution with sigma = err_max.
            - 3: Centred propagation. Assume each parameters follows a normal
                    distribution with sigma = (err_min + err_max) / 2.
            - 4: Deviated propagation. Assume each parameters follows a normal
                    distribution with sigma = (err_max + err_min) / 2, but the
                    mean is at ((val + err_min) + (val + err_max)) / 2.


    Returns
    -------
    Tuple[float, float, float]
        Hill radius and its minimum and maximum errors, in AU.
    """
    # Switch for the error propagation method
    if err_method <= 0:
        return hill_radius(a, e, m_star, m_planet), 0, 0
    elif err_method == 1:
        hill = hill_radius(a, e, m_star, m_planet)
        hill_min = hill_radius(
            a - a_err_min,
            e - e_err_min,
            m_star + m_star_err_max,
            m_planet + m_planet_err_max,
        )
        hill_max = hill_radius(
            a + a_err_max,
            e + e_err_max,
            m_star - m_star_err_min,
            m_planet - m_planet_err_min,
        )
        hill_err_min = abs(hill - hill_min)
        hill_err_max = abs(hill - hill_max)
        return hill, hill_err_min, hill_err_max
    elif err_method == 2:
        a_err = max(a_err_min, a_err_max) * AU
        e_err = max(e_err_min, e_err_max)
        m_star_err = max(m_star_err_min, m_star_err_max) * M_SUN
        m_planet_err = max(m_planet_err_min, m_planet_err_max) * M_JUP
    elif err_method == 3 or err_method == 4:
        a_err = (a_err_min + a_err_max) * 0.5 * AU
        e_err = (e_err_min + e_err_max) * 0.5
        m_star_err = (m_star_err_min + m_star_err_max) * 0.5 * M_SUN
        m_planet_err = (m_planet_err_min + m_planet_err_max) * 0.5 * M_JUP
        if err_method == 4:
            a = (a - a_err_min + a + a_err_max) * 0.5
            e = (e - e_err_min + e + e_err_max) * 0.5
            m_star = (m_star - m_star_err_min + m_star + m_star_err_max) * 0.5
            m_planet = (
                m_planet - m_planet_err_min + m_planet + m_planet_err_max
            ) * 0.5
    else:
        raise ValueError("Invalid error propagation method.")

    # Calculate the Hill radius (in AU)
    hill = hill_radius(a, e, m_star, m_planet)

    # Auxiliary total mass
    total_mass = m_star * M_SUN + m_planet * M_JUP

    # Partial derivatives for error propagation
    dhill_da = hill / a
    dhill_de = -hill / (1 - e)
    dhill_dm_star = -hill / (3 * total_mass)
    dhill_dm_planet = -dhill_dm_star * (m_star * M_SUN) / (m_planet * M_JUP)

    # Errors
    hill_err = sqrt(
        (dhill_da * a_err) ** 2
        + (dhill_de * e_err) ** 2
        + (dhill_dm_star * m_star_err) ** 2
        + (dhill_dm_planet * m_planet_err) ** 2
    )

    return hill, hill_err, hill_err  # Same error for min and max
