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
import warnings
from fractions import Fraction
from types import MappingProxyType
from typing import Iterable, Union

from numpy import log, pi, random, sqrt

from resokit import __version__ as version

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

# Gravitational constant in SI units
G = 6.67430e-11  # m^3 kg^-1 s^-2

# Astronomical unit in meters
AU = 1.496e11  # m
# Parsec in meters
PC = 3.086e16  # m
# Solar radius in m
R_SUN = 6.957e8  # m
# Jupiter radius in m
R_JUP = 6.9911e7  # m
# Earth radius in m
R_EAR = 6.371e6  # m

# Solar mass in kg
M_SUN = 1.989e30  # kg
# Jupiter mass in kg
M_JUP = 1.898e27  # kg
# Earth mass in kg
M_EAR = 5.972e24  # kg

# Hour in seconds
HOUR = 3600  # s
# Day in seconds
DAY = 86400  # s
# Year in seconds
YEAR = 3.154e7  # s


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
) -> Union[Fraction, tuple[int, int]]:
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
    Union[Fraction, tuple[int, int]]
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
                f"Iter {i + 1:>2d}: {numer[i]:>3d}/{denom[i]:<3d} "
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


def calc_period(a: float, m_star: float, m_planet) -> float:
    """Calculate the orbital period of a planet.

    Parameters
    ----------
    a : float
        Semi-major axis of the planet.
    m_star : float
        Mass of the star.
    m_planet : float
        Mass of the planet.

    Returns
    -------
    float
        Orbital period of the planet, in days.
    """

    ene = sqrt(G * (m_star * M_SUN + m_planet * M_JUP) / (a * AU) ** 3)

    return 2 * pi / ene / DAY


def calc_a(period: float, m_star: float, m_planet: float) -> float:
    """Calculate the semi-major axis of a planet.

    Parameters
    ----------
    period : float
        Orbital period of the planet, in days.
    m_star : float
        Mass of the star.
    m_planet : float
        Mass of the planet.

    Returns
    -------
    float
        Semi-major axis of the planet, in AU.
    """

    ene = 2 * pi / period / DAY

    return (G * (m_star * M_SUN + m_planet * M_JUP) / ene**2) ** (1 / 3) / AU


def chen_kipping_mass(
    radius: float,
    radius_err_min: float = 0.0,
    radius_err_max: float = 0.0,
    trivariate: tuple = (0.1, 0.8),
    method: int = 1,
    silent: bool = False,
) -> tuple[float, float]:
    """Calculate the mass of a planet using the Chen & Kipping (2017).

    We follow a naive approximation of the mass-radius relation for exoplanets
    proposed by Chen, J., & Kipping, D. (2017, ApJ, 834, 17). Instead of using
    markov chain monte carlo (MCMC) methods, we use a simple power-law
    approximation with naive error propagation.
    Equation (1) defines: r / R_EAR = C x (m / M_EAR)**S, which can be rewritten
    as Equation(2): R = C_10 + M x S, where
     > R is log_10(r / R_EAR), where r is the radius of the planet
     > M is log_10(m / M_EAR), where m is the mass of the planet
     > S is the slope of the power-law relation
     > C_10 is log_10(C), where C is the constant of the power-law relation

    Parameters
    ----------
    radius : float
        Radius of the planet, in Earth radii.
    radius_err_min : float
        Lower error of the radius, in Earth radii.
    radius_err_max : float
        Upper error of the radius, in Earth radii.
    trivariate : float, optional. Default: 0.8
        Probabilities (from 0 to 1) that the returned radius that falls in the
        trivariate region is calculated with the second (left), or third
        (center) branch of the power-law approximation. The probability of
        using the fourth (right) branch is equal to 1 - sum(bivariate), so
        the sum of them must be lower equal than 1.
    method : int, optional. Default: 1
        Which method implement for error calculation.
        Method 1: (Naive) Error propagation with the power-law approximation,
         using the radius error as the maximum of the two extremes.
        Method 2: Evalaute the radius extremes and calculate each mass extreme
         with the power-law approximation.
    silent : bool, optional. Default: False
        Whether to be silence warning.

    Returns
    -------
    tuple[float, float]
        Mass of the planet and its errors, in Earth masses.
    """

    # Error propagation function
    def calc_err(m, r, r_err, c, c_err, s, s_err):
        return sqrt(
            (-m * log(r / c) / s**2 * s_err) ** 2  # s_err
            + (m / r / s * r_err) ** 2  # r_err
            + (-m / r / s / c * c_err) ** 2  # c_err
        )

    # Define coefficients of the power-law approximation

    # ------------------------------------------------------------------------
    # c1 = (1.008, -0.0045, 0.0046)
    # s1 = (0.279, -0.0094, 0.0092)
    # s2 = (0.589, -0.031, 0.044)
    # s3 = (-0.044, -0.019, 0.017)
    # s4 = (0.881, -0.024, 0.025)
    # m1_tr = (2.04, -0.59, 0.66)
    # m2_tr = tuple(x * M_JUP / M_EAR for x in [0.414, -0.065, 0.057])
    # m3_tr = tuple(x * M_SUN / M_EAR for x in [0.08, -0.0072, 0.0081])
    # ------------------------------------------------------------------------

    # Constant 1 of the power-law approximation
    c1 = (1.008, 0.0046)

    # Slope of the power-law approximation
    s1 = (0.279, 0.0094)
    s2 = (0.589, 0.044)
    s3 = (-0.044, 0.019)
    s4 = (0.881, 0.025)

    # ------------------------------------------------------------------------
    # # Transition mass
    # m1_tr = (2.04, 0.66)
    # m2_tr = tuple(x * M_JUP / M_EAR for x in [0.414, 0.065])
    # m3_tr = tuple(x * M_SUN / M_EAR for x in [0.08, 0.0081])

    # # Auxiliary calculations
    # aux_1 = m1_tr[0] ** s1[0]
    # aux_1 = (
    #     aux_1,
    #     sqrt(
    #         (aux_1 * s1[0] / m1_tr[0] * m1_tr[1]) ** 2
    #         + (aux_1 * log(m1_tr[0]) * s1[1]) ** 2
    #     ),
    # )

    # # Calculate the transition radius 1
    # r1_tr = c1[0] * aux_1[0]  # Transition Radius 1
    # r1_tr = (r1_tr, sqrt((aux_1[0] * c1[1]) ** 2
    #                         + (c1[0] * aux_1[1]) ** 2))

    # # Auxiliary calculations
    # aux_2 = m1_tr[0] ** s2[0]
    # aux_2 = (
    #     aux_2,
    #     sqrt(
    #         (aux_2 * s2[0] / m1_tr[0] * m1_tr[1]) ** 2
    #         + (aux_2 * log(m1_tr[0]) * s2[1]) ** 2
    #     ),
    # )

    # # Calculate the Constant 2
    # c2 = r1_tr[0] / aux_2[0]  # Constant 2
    # c2 = (
    #     c2,
    #     sqrt(
    #         (1.0 / aux_2[0] * r1_tr[1]) ** 2 +
    #         (c2 / aux_2[0] * aux_2[1]) ** 2
    #     ),
    # )

    # # Calculate the transition radius 2
    # r2_tr = c2[0] * m2_tr[0] ** s2[0]  # Transition Radius 2
    # r2_tr = (
    #     r2_tr,
    #     sqrt(
    #         (r2_tr / c2[0] * c2[1]) ** 2
    #         + (r2_tr * s2[0] / m2_tr[0] * m2_tr[1]) ** 2
    #         + (r2_tr * log(m2_tr[0]) * s2[1]) ** 2
    #     ),
    # )

    # # Auxiliary calculations
    # aux_3 = m2_tr[0] ** s3[0]
    # aux_3 = (
    #     aux_3,
    #     sqrt(
    #         (aux_3 * s3[0] / m2_tr[0] * m2_tr[1]) ** 2
    #         + (aux_3 * log(m2_tr[0]) * s3[1]) ** 2
    #     ),
    # )

    # # Calculate the Constant 3
    # c3 = r2_tr[0] / aux_3[0]  # Constant 3
    # c3 = (
    #     c3,
    #     sqrt(
    #         (1.0 / aux_3[0] * r2_tr[1]) ** 2 +
    #         (c3 / aux_3[0] * aux_3[1]) ** 2
    #     ),
    # )

    # # Calculate the transition radius 3
    # r3_tr = c3[0] * m3_tr[0] ** s3[0]  # Transition Radius 3
    # r3_tr = (
    #     r3_tr,
    #     sqrt(
    #         (r3_tr / c3[0] * c3[1]) ** 2
    #         + (r3_tr * s3[0] / m3_tr[0] * m3_tr[1]) ** 2
    #         + (r3_tr * log(m3_tr[0]) * s3[1]) ** 2
    #     ),
    # )

    # # Auxiliary calculations
    # aux_4 = m3_tr[0] ** s4[0]
    # aux_4 = (
    #     aux_4,
    #     sqrt(
    #         (aux_4 * s4[0] / m3_tr[0] * m3_tr[1]) ** 2
    #         + (aux_4 * log(m3_tr[0]) * s4[1]) ** 2
    #     ),
    # )

    # # Calculate the Constant 4
    # c4 = r3_tr[0] / aux_4[0]  # Constant 4
    # c4 = (
    #     c4,
    #     sqrt(
    #         (1.0 / aux_4[0] * r3_tr[1]) ** 2 +
    #         (c4 / aux_4[0] * aux_4[1]) ** 2
    #     ),
    # )

    # ------------------------------------------------------------------------

    # print(f"Transition M: {m1_tr:.6f}, {m2_tr:.6f}, {m3_tr:.6f} M_EAR")
    # print(f" Errors: {m1_tr[1]:.6f}, {m2_tr[1]:.6f}, {m3_tr[1]:.6f} M_EAR")
    # print(
    #     f"Transition R: {r1_tr[0]:.6f}, {r2_tr[0]:.6f}, {r3_tr[0]:.6f} R_EAR"
    # )
    # print(f" Errors: {r1_tr[1]:.6f}, {r2_tr[1]:.6f}, {r3_tr[1]:.6f} R_EAR")
    # print(f"Constants: {c1[0]:.6f}, {c2[0]:.6f}, {c3[0]:.6f}, {c4[0]:.6f}")
    # print(f" Errors: {c1[1]:.6f}, {c2[1]:.6f}, {c3[1]:.6f}, {c4[1]:.6f}")
    # print(f"Slopes: {s1[0]:.6f}, {s2[0]:.6f}, {s3[0]:.6f}, {s4[0]:.6f}")
    # print(f" Errors: {s1[1]:.6f}, {s2[1]:.6f}, {s3[1]:.6f}, {s4[1]:.6f}")

    # ------------------------------------------------------------------------

    # Transition radii
    r1_tr = (1.229836, 0.111458)
    r2_tr = (14.31101, 4.529131)
    r3_tr = (11.328892, 4.333345)

    # Constants of the power-law approximation
    c2 = (0.808119, 0.0172397)
    c3 = (17.738384, 5.851034)
    c4 = (0.00143, 0.000669)

    # return (
    #    s1, s2, s3, s4, c1, c2, c3, c4,
    #    r1_tr, r2_tr, r3_tr, m1_tr, m2_tr, m3_tr
    #    )

    prob = -1

    # Separate the power-law approximation
    if radius <= r1_tr[0]:  # First branch
        c = c1
        s = s1
    elif radius > r2_tr[0]:  # Fourth branch
        c = c4
        s = s4
    elif radius < r3_tr[0]:  # Second branch
        c = c2
        s = s2
    else:  # Trivariate region
        if not silent:
            warnings.warn(
                "Radius falls in the trivariate region: "
                + f"{r3_tr[0]} < R < {r2_tr[0]}"
                + "\n The mass-radius relation may not be accurate.",
                stacklevel=2,
            )
        if len(trivariate) != 2:
            raise ValueError("Bivariate must have lenght 2.")
        sumb = sum(trivariate)
        if sumb < 0 or sumb > 1:
            raise ValueError(
                "Sum of trivariate must be a number between 0 and 1."
            )
        prob = random.rand()
        if prob < trivariate[0]:  # Second branch
            c = c2
            s = s2
        elif prob < sumb:  # Third branch
            c = c3
            s = s3
        else:  # Fourth branch
            c = c4
            s = s4

    # Handle errors as absolute values
    radius_err_min = abs(radius_err_min)
    radius_err_max = abs(radius_err_max)

    # Calculate the mass
    mass = (radius / c[0]) ** (1.0 / s[0])

    # Calculate the error
    if method == 1:  # Error propagation
        radius_err = max(radius_err_min, radius_err_max)  # Use max R_err
        mass_err_max = calc_err(
            mass, radius, radius_err, c[0], c[1], s[0], s[1]
        )
        
        if s != s3:  # If not case 3...
            mass_err_min = -mass_err_max  # common error
        else:  # If case 3
            mass_if_s2 = (radius / c2[0]) ** (1.0 / s2[0])
            mass_err_min = mass_if_s2 - mass
    else:  # Using the extremes
        if not silent and any([radius_err_min, radius_err_max]) == 0.0:
            warnings.warn(
                "Calculating the mass error at extremes without a radius error "
                + " generanes no mass error",
                stacklevel=2,
            )
        # Calculate the mass extremes
        mass_min = ((radius - radius_err_min) / c[0]) ** (1.0 / s[0])
        mass_max = ((radius + radius_err_max) / c[0]) ** (1.0 / s[0])
        # Calculate the mass error. Safe sign
        mass_err_min = min(min(mass_min, mass_max), mass) - mass
        mass_err_max = max(max(mass_min, mass_max), mass) - mass

    return mass, mass_err_min, mass_err_max
