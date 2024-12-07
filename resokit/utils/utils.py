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


def power_law(x: float, c: float, s: float, x0: float = 1.0) -> float:
    """Calculate a power-law.

    Equation: y = c * x^s

    Parameters
    ----------
    x : float
        Value to calculate the power-law.
    c : float
        Constant of the power-law relation.
    s : float
        Slope of the power-law relation.
    x0 : float, optional. Default: 1
        Reference value of the power-law

    Returns
    -------
    float
        Result of the power-law.
    """
    return c * (x / x0) ** s


def power_law_error(
    x: float,
    x_err: float,
    c: float,
    c_err: float,
    s: float,
    s_err: float,
    x0: float = 1.0,
    x0_err: float = 0.0,
    y: float = 0,
) -> float:
    """Calculate the (naive propagation) error of a power-law.

    Equation: y_err = sqrt(
        (y / c  * c_err)^2
        + (y * log(x / x0) * s_err)^2
        + (y * s / x * x_err)^2
        + (y * s / x0 * x0_err)^2
    )

    Parameters
    ----------
    x : float
        Value to calculate the power-law.
    x_err : float
        Error of the value.
    c : float
        Constant of the power-law relation.
    c_err : float
        Error of the constant.
    s : float
        Slope of the power-law relation.
    s_err : float
        Error of the slope.
    x0 : float, optional. Default: 1
        Reference value of the power-law.
    x0_err : float, optional. Default: 0
        Error of the reference value.
    y : float, optional. Default: 0
        Value of the power-law.

    Returns
    -------
    float
        Error of the power-law.
    """
    if y == 0:
        y = power_law(x, c, s, x0)

    return sqrt(
        (y / c * c_err) ** 2  # dy/dc
        + (y * log(x / x0) * s_err) ** 2  # dy/ds
        + (y * s / x * x_err) ** 2  # dy/dx
        + (y * s / x0 * x0_err) ** 2  # dy/dx0
    )


def otegi_2020_mass(
    radius: float,
    density: float = 0.0,
    bivariate: float = 0.5,
    silent: bool = False,
) -> tuple[float, float, float]:
    """Calculate the mass of a planet using Otegi et al. (2020).

    Power law approximation:
        mass = C x radius^S.
    Citation:
        Otegi, J. F., Bouchy, F., & Helled, R. 2020, A&A, 634, A43

    Parameters
    ----------
    radius : float
        Mass of the planet, in Earth radii.
    density : float, optional. Default: 0.0
        Density of the planet, in kg m^-3.
    bivariate : float, optional. Default: 0.5
        Probability that the returned radius that falls in the bivariate
        region is calculated with lower (rho < 3300 kg m^-3) branch, instead
        of using the upper (rho > 3300 kg m^-3) branch of the power-law
        approximation. Must be a number between 0 and 1.
    silent : bool, optional. Default: False
        Whether to print a warning if the radius is greater than the maximum
        value used by Otegi et al. (2020), or if the estimation falls
        in a multivariate region.

    Returns
    -------
    tuple[float, tuple, tuple]
        Mass of the planet, in Earth masses, and the constant and slope used.
    """
    if radius > 14.3 and not silent:
        warnings.warn(
            "Radius is greater than the maximum value "
            + "used by Otegi et al. (2020): R = 14.3 R_earth.\n"
            + "The power-law approximation may not be accurate.",
            stacklevel=2,
        )

    # Otegi cuts at rho = 3300 kg m^-3 = 3.3 g cm^-3
    # Constants
    c1 = (1.74, 0.38)  # lower branch: <= 3300 kg m^-3
    c2 = (0.90, 0.06)  # upper branch: > 3300 kg m^-3
    # Slopes
    s1 = (1.58, 0.10)  # lower branch: <= 3300 kg m^-3
    s2 = (3.45, 0.12)  # upper branch: > 3300 kg m^-3

    dens_cut = 3300  # kg m^-3

    if density > 0.0:  # If density is set
        if density > dens_cut:
            c = c1
            s = s1
        else:
            c = c2
            s = s2
        mass = power_law(radius, c[0], s[0])
    else:  # If density is not set
        scaled_dens = dens_cut / M_EAR * R_EAR**3  # dens [M_ear R_ear^-3]
        # Try both branches
        mass1 = power_law(radius, c1[0], s1[0])
        mass2 = power_law(radius, c2[0], s2[0])
        # Calculate the density
        density1 = mass1 / (4 / 3 * pi * radius**3)
        density2 = mass2 / (4 / 3 * pi * radius**3)
        # Check if lower or upper branch
        if density1 <= scaled_dens and density2 <= scaled_dens:
            # First branch is valid
            c = c1
            s = s1
            mass = mass1
        elif density1 > scaled_dens and density2 > scaled_dens:
            # Second branch is valid
            c = c2
            s = s2
            mass = mass2
        elif density1 <= scaled_dens and density2 > scaled_dens:
            if not silent:
                warnings.warn(
                    "The estimation falls in a multivariate region. "
                    + "The power-law approximation may not be accurate.",
                    stacklevel=2,
                )
            # Both branches are valid.
            if not isinstance(bivariate, (int, float)):
                raise ValueError("Bivariate must be a number between 0 and 1.")
            if bivariate < 0 or bivariate > 1:
                raise ValueError("Bivariate must be a number between 0 and 1.")
            if random.rand() < bivariate:
                c = c1
                s = s1
                mass = mass1
            else:
                c = c2
                s = s2
                mass = mass2
        else:
            raise ValueError("Error in the density calculation.")

    return mass, c, s


def otegi_2020_radius(
    mass: float,
    density: float = 0.0,
    bivariate: float = 0.5,
    silent: bool = False,
) -> tuple[float, float, float]:
    """Calculate the radius of a planet using Otegi et al. (2020).

    Power law approximation:
        radius = C x mass^S
    Citation:
        Otegi, J. F., Bouchy, F., & Helled, R. 2020, A&A, 634, A43

    Parameters
    ----------
    mass : float
        Mass of the planet, in Earth radii.
    density : float, optional. Default: 0.0
        Density of the planet, in kg m^-3.
    bivariate : float, optional. Default: 0.5
        Probability that the returned mass that falls in the bivariate
        region is calculated with lower (rho < 3300 kg m^-3) branch, instead
        of using the upper (rho > 3300 kg m^-3) branch of the power-law
        approximation. Must be a number between 0 and 1.
    silent : bool, optional. Default: False
        Whether to silence the warning if the radius is greater than the maximum
        value used by Otegi et al. (2020), or if the estimation falls
        in a multivariate region.

    Returns
    -------
    tuple[float, tuple, tuple]
        Radius of the planet, in Earth radii, and the constant and slope used.
    """
    if mass > 120 and not silent:
        warnings.warn(
            "Radius is greater than the maximum value "
            + "used by Otegi et al. (2020): M = 120 M_earth.\n"
            + "The power-law approximation may not be accurate.",
            stacklevel=2,
        )

    # Otegi cuts at rho = 3300 kg m^-3 = 3.3 g cm^-3
    # Constants
    c1 = (0.70, 0.11)  # lower branch: <= 3300 kg m^-3
    c2 = (1.03, 0.02)  # upper branch: > 3300 kg m^-3
    # Slopes
    s1 = (0.63, 0.04)  # lower branch: <= 3300 kg m^-3
    s2 = (0.29, 0.01)  # upper branch: > 3300 kg m^-3

    dens_cut = 3300  # kg m^-3

    if density > 0.0:  # If density is set
        if density > dens_cut:
            c = c1
            s = s1
        else:
            c = c2
            s = s2
        radius = power_law(mass, c[0], s[0])
    else:  # If density is not set
        scaled_dens = dens_cut / M_EAR * R_EAR**3  # dens [M_ear R_ear^-3]
        # Try both branches
        radius1 = power_law(mass, c1[0], s1[0])
        radius2 = power_law(mass, c2[0], s2[0])
        # Calculate the density
        density1 = mass / (4 / 3 * pi * radius1**3)
        density2 = mass / (4 / 3 * pi * radius2**3)
        # Check if lower or upper branch
        if density1 <= scaled_dens and density2 <= scaled_dens:
            # First branch is valid
            c = c1
            s = s1
            radius = radius1
        elif density1 > scaled_dens and density2 > scaled_dens:
            # Second branch is valid
            c = c2
            s = s2
            radius = radius2
        elif density1 <= scaled_dens and density2 > scaled_dens:
            if not silent:
                warnings.warn(
                    "The estimation falls in a multivariate region. "
                    + "The power-law approximation may not be accurate.",
                    stacklevel=2,
                )
            # Both branches are valid.
            if not isinstance(bivariate, (int, float)):
                raise ValueError("Bivariate must be a number between 0 and 1.")
            if bivariate < 0 or bivariate > 1:
                raise ValueError("Bivariate must be a number between 0 and 1.")
            if random.rand() < bivariate:
                c = c1
                s = s1
                radius = radius1
            else:
                c = c2
                s = s2
                radius = radius2
        else:
            raise ValueError("Error in the density calculation.")

    return radius, c, s


def chen_kipp_2017_radius(mass: float) -> tuple[float, float, float]:
    """Calculate the radius of a planet using the Chen & Kipping (2017).

    Power law approximation:
        radius = C x mass^S
    Citation:
        Chen, J., & Kipping, D. 2017, ApJ, 834, 17
    The original code is available at:
        https://github.com/chenjj2/forecaster

    Parameters
    ----------
    mass : float
        Mass of the planet, in Earth masses.

    Returns
    -------
    tuple[float, tuple, tuple]
        Radius of the planet, in Earth radii, and the constant and slope used.
    """
    # Constants
    c1 = (1.008, 0.0046)
    c2 = (0.808119, 0.0172397)
    c3 = (17.738384, 5.851034)
    c4 = (0.00143, 0.000669)
    # Slopes
    s1 = (0.279, 0.0094)
    s2 = (0.589, 0.044)
    s3 = (-0.044, 0.019)
    s4 = (0.881, 0.025)
    # Transition mass
    m1_tr = 2.04
    m2_tr = 0.414 * M_JUP / M_EAR
    m3_tr = 0.08 * M_SUN / M_EAR

    if mass < m1_tr:
        c = c1
        s = s1
    elif mass < m2_tr:
        c = c2
        s = s2
    elif mass < m3_tr:
        c = c3
        s = s3
    else:
        c = c4
        s = s4

    return power_law(mass, c[0], s[0]), c, s


def chen_kipp_2017_mass(
    radius: float, trivariate: tuple = (0.15, 0.8), silent: bool = False
) -> tuple[float, float]:
    """Calculate the mass of a planet using the Chen & Kipping (2017).

    Power law approximation:
        mass = (1/C) x radius^(1/S)
    Citation:
        Chen, J., & Kipping, D. 2017, ApJ, 834, 17
    The original code is available at:
        https://github.com/chenjj2/forecaster

    Parameters
    ----------
    radius : float
        Radius of the planet, in Earth radii.
    trivariate : tuple, optional. Default: (0.15, 0.8)
        Probabilities (from 0 to 1) that the returned radius that falls in the
        trivariate region is calculated with the second (left), ant then third
        (center) branch of the power-law approximation. The probability of
        using the fourth (right) branch is equal to 1 - sum(bivariate), so
        the sum of them must be lower equal than 1.
    silent : bool, optional. Default: False
        Whether to silence the warning if the radius falls in a
        multivariate region.

    Returns
    -------
    tuple[float, tuple, tuple]
        Mass of the planet and the constant and slope used.
    """
    # Constants
    c1 = (1.008, 0.0046)
    c2 = (0.808119, 0.0172397)
    c3 = (17.738384, 5.851034)
    c4 = (0.00143, 0.000669)
    # Slopes
    s1 = (0.279, 0.0094)
    s2 = (0.589, 0.044)
    s3 = (-0.044, 0.019)
    s4 = (0.881, 0.025)
    # Transition radius
    r1_tr = (1.229836, 0.111458)
    r2_tr = (14.31101, 4.529131)
    r3_tr = (11.328892, 4.333345)

    if radius < r1_tr:  # First branch
        c = c1
        s = s1
    elif radius > r2_tr:  # Pure fourth branch
        c = c4
        s = s4
    elif radius < r3_tr:  # Pure second branch
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

    # We use the inverse of the constant and slope,
    # so the error is recaclulated with propagation error.
    return (
        power_law(radius, 1.0 / c[0], 1.0 / s[0]),
        (1.0 / c[0], c[1] / c[0]),
        (1.0 / s[0], s[1] / s[0]),
    )


def edmonson_2023_radius(mass: float) -> tuple[float, float, float]:
    """Calculate the radius of a planet using the Edmondson et al. (2023).

    Power law approximation:
        radius = C x mass^S
    Citation:
        Edmondson, K., Norris, J., & Kerins, E. 2023, Open J. Astrophysics,
        submitted [arXiv:2310.16733]

    Parameters
    ----------
    mass : float
        Mass of the planet, in Earth masses.

    Returns
    -------
    tuple[float, tuple, tuple]
        Radius of the planet, in Earth radii, and the constant and slope used.
    """
    # Constants
    c1 = (1.01, 0.03)
    c2 = (0.53, 0.05)
    c3 = (13, 1.2)
    # Slopes
    s1 = (0.28, 0.03)
    s2 = (0.68, 0.02)
    s3 = (0.012, 0.003)
    # Transition mass
    m1_tr = 4.95
    m2_tr = 115

    if mass < m1_tr:  # First branch
        c = c1
        s = s1
    elif mass < m2_tr:  # Second branch
        c = c2
        s = s2
    else:  # Third branch
        c = c3
        s = s3

    return power_law(mass, c[0], s[0]), c, s


def edmonson_2023_mass(radius: float) -> tuple[float, float, float]:
    """Calculate the mass of a planet using the Edmondson et al. (2023).

    Power law approximation:
        mass = C x radius^S
    Citation:
        Edmondson, K., Norris, J., & Kerins, E. 2023, Open J. Astrophysics,
        submitted [arXiv:2310.16733]

    Parameters
    ----------
    radius : float
        Radius of the planet, in Earth radii.

    Returns
    -------
    tuple[float, tuple, tuple]
        Mass of the planet and the constant and slope used.
    """
    # Constants
    c1 = (1.01, 0.03)
    c2 = (0.53, 0.05)
    c3 = (13, 1.2)
    # Slopes
    s1 = (0.28, 0.03)
    s2 = (0.68, 0.02)
    s3 = (0.012, 0.003)
    # Transition radius
    r1_tr = power_law(4.95, c1[0], s1[0])  # 4.95 Earth masses
    r2_tr = power_law(115, c2[0], s2[0])  # 115 Earth masses

    # No multivariate region, because the power-law is always defined positive.

    if radius < r1_tr:  # First branch
        c = c1
        s = s1
    elif radius < r2_tr:  # Second branch
        c = c2
        s = s2
    else:  # Third branch
        c = c3
        s = s3

    return (
        power_law(radius, 1.0 / c[0], 1.0 / s[0]),
        (1.0 / c[0], c[1] / c[0]),
        (1.0 / s[0], s[1] / s[0]),
    )


def muller_2024_radius(mass: float) -> tuple[float, float, float]:
    """Calculate the radius of a planet using the Müller et al. (2024).

    Power law approximation:
        radius = C x mass^S
    Citation:
        Müller S., Baron J., Helled R., Bouchy F. & Parc L. 2024, A&A, 686, A296

    Parameters
    ----------
    mass : float
        Mass of the planet, in Earth masses.

    Returns
    -------
    tuple[float, tuple, tuple]
        Radius of the planet, in Earth radii, and the constant and slope used.
    """
    # Constants
    c1 = (1.02, 0.03)
    c2 = (0.56, 0.03)
    c3 = (18.6, 6.7)
    # Slopes
    s1 = (0.27, 0.04)
    s2 = (0.67, 0.05)
    s3 = (-0.006, 0.07)
    # Transition mass
    m1_tr = 4.37
    m2_tr = 127

    if mass < m1_tr:  # First branch
        c = c1
        s = s1
    elif mass < m2_tr:  # Second branch
        c = c2
        s = s2
    else:  # Third branch
        c = c3
        s = s3

    return power_law(mass, c[0], s[0]), c, s


def muller_2024_mass(
    radius: float, bivariate: float = 0.5, silent: bool = False
) -> tuple[float, float, float]:
    """Calculate the mass of a planet using the Müller et al. (2024).

    Power law approximation:
        mass = C x radius^S
    Citation:
        Müller S., Baron J., Helled R., Bouchy F. & Parc L. 2024, A&A, 686, A296

    Parameters
    ----------
    radius : float
        Radius of the planet, in Earth radii.
    bivariate : float, optional. Default: 0.5
        Probability that the returned mass that falls in the bivariate
        region is calculated with the second (left) branch, instead of using
        the third (right) branch of the power-law approximation. Must be a
        number between 0 and 1.
    silent : bool, optional. Default: False
        Whether to silence the warning if the radius falls in the
        bivariate region.

    Returns
    -------
    tuple[float, tuple, tuple]
        Mass of the planet, in Earth masses, and the constant and slope used.
    """
    # Constants
    c1 = (1.02, 0.03)
    c2 = (0.56, 0.03)
    c3 = (18.6, 6.7)
    # Slopes
    s1 = (0.27, 0.04)
    s2 = (0.67, 0.05)
    s3 = (-0.006, 0.07)
    # Transition radius
    r1_tr = power_law(4.37, c1[0], s1[0])  # 4.37 Earth masses
    r2_tr = power_law(127, c2[0], s2[0])  # 127 Earth masses
    r3_tr = power_law(1e4, c3[0], s3[0])  # Top: 1e4 Earth masses

    if radius < r1_tr:  # First branch
        c = c1
        s = s1
    elif radius < r3_tr:  # Pure second branch
        c = c2
        s = s2
    elif radius > r2_tr:  # No estimation
        raise ValueError(
            "Radius is greater than the maximum value used by "
            + f"Müller et al. (2024): R = {r2_tr} R_earth."
        )
    else:  # Bivariate region
        if not silent:
            warnings.warn(
                "Radius falls in the bivariate region: "
                + f"{r1_tr} < R < {r2_tr}"
                + "\n The mass-radius relation may not be accurate.",
                stacklevel=2,
            )
        if not isinstance(bivariate, (int, float)):
            raise ValueError("Bivariate must be a number between 0 and 1.")
        if bivariate < 0 or bivariate > 1:
            raise ValueError("Bivariate must be a number between 0 and 1.")
        if random.rand() < bivariate:  # Second branch
            c = c2
            s = s2
        else:  # Third branch
            c = c3
            s = s3

    # We use the inverse of the constant and slope,
    # so the error is recaclulated with propagation error.

    return (
        power_law(radius, 1.0 / c[0], 1.0 / s[0]),
        (1.0 / c[0], c[1] / c[0]),
        (1.0 / s[0], s[1] / s[0]),
    )


def estimate_mass(
    radius: float,
    radius_err_min: float = 0.0,
    radius_err_max: float = 0.0,
    multivariate: float = 0.5,
    model: str = "ck17",
    method: int = 1,
    silent: bool = False,
) -> tuple[float, float, float]:
    """Calculate the mass of a planet using a power-law approximation.

    Equation: mass = C x radius^S

    Parameters
    ----------
    radius : float
        Radius of the planet, in Earth radii.
    radius_err_min : float
        Lower error of the radius, in Earth radii.
    radius_err_max : float
        Upper error of the radius, in Earth radii.
    multivariate : float, tuple, optional. Default: 0.5
        Probability of using the (first, second, ...) branch if the estimation
        falls in a multivariate region. If a float, must be a number between
        0 and 1. If a tuple, must be a tuple of two floats between 0 and 1,
        where the sum of them must be lower equal than 1.
    model : str, optional. Default: "ck17"
        Model to use for the mass-radius power-law relation.
        "ck17": Chen & Kipping (2017)
        "o20": Otegi et al. (2020)
        "e23": Edmondson et al. (2023)
        "m24": Müller et al. (2024)
    method : int, optional. Default: 1
        Which method implement for error calculation.
        Method 1: (Naive) Error propagation with the power-law approximation,
            using the radius error as the maximum of the two extremes.
        Method 2: Evalaute the radius extremes and calculate each mass
            extreme with the power-law approximation.
    silent : bool, optional. Default: False
        Whether to silence the warning if the radius falls in a
        multivariate region.

    Returns
    -------
    tuple[float, float, float]
        Mass of the planet and its errors, in Earth masses.
    """
    # Calculate the mass
    if model == "ck17":
        mass, c, s = chen_kipp_2017_mass(radius, multivariate, silent)
    elif model == "o20":
        mass, c, s = otegi_2020_mass(radius, 0.0, 0.0, multivariate, silent)
    elif model == "e23":
        mass, c, s = edmonson_2023_mass(radius)
    elif model == "m24":
        mass, c, s = muller_2024_mass(radius, multivariate, silent)
    else:
        raise ValueError("Model not implemented.")

    # Calculate the mass error
    mass_err_min, mass_err_max = _aux_error_estimator(
        radius, radius_err_max, radius_err_min, mass, c, s, method, silent, 0
    )

    return mass, mass_err_min, mass_err_max


def estimate_radius(
    mass: float,
    mass_err_min: float = 0.0,
    mass_err_max: float = 0.0,
    bivariate: float = 0.0,
    model: str = "ck17",
    method: int = 1,
    density: float = 0.0,
    silent: bool = False,
) -> tuple[float, float, float]:
    """Calculate the radius of a planet using the power-law approximation.

    Equation: radius = (1/C) x mass^(1/S)

    Parameters
    ----------
    mass : float
        Mass of the planet, in Earth masses.
    mass_err_min : float
        Lower error of the mass, in Earth masses.
    mass_err_max : float
        Upper error of the mass, in Earth masses.
    bivariate : float, optional. Default: 0.0
        Probability of using the first branch if the estimation falls in a
        bivariate region. Must be a number between 0 and 1.
        Only used if model is "o20".
    model : str, optional. Default: "ck17"
        Model to use for the mass-radius power-law relation.
        "ck17": Chen & Kipping (2017) [trivariate]
        "o20": Otegi et al. (2020) [bivariate]
        "e23": Edmondson et al. (2023)
        "m24": Müller et al. (2024)
    method : int, optional. Default: 1
        Which method implement for error calculation.
        Method 1: (Naive) Error propagation with the power-law approximation,
            using the mass error as the maximum of the two extremes.
        Method 2: Evalaute the mass extremes and calculate each radius extreme.
    density : float, optional. Default: 0.0
        Density of the planet, in kg m^-3.
        Only used if model is "o20".
    silent : bool, optional. Default: False
        Whether to silence the warning if the radius falls in a
        multivariate region,
        or if the estimation is not accurate.
    """
    # Calculate the radius
    if model == "ck17":
        radius, c, s = chen_kipp_2017_radius(mass)
    elif model == "o20":
        if density == 0.0 and bivariate == 0:
            raise ValueError(
                "Density or bivariate must be set if model is 'o20'."
            )
        radius, c, s = otegi_2020_radius(mass, density, bivariate, silent)
    elif model == "e23":
        radius, c, s = edmonson_2023_radius(mass)
    elif model == "m24":
        radius, c, s = muller_2024_radius(mass)
    else:
        raise ValueError("Model not implemented.")

    # Calculate the radius error
    radius_err_min, radius_err_max = _aux_error_estimator(
        mass, mass_err_max, mass_err_min, radius, c, s, method, silent, 1
    )

    return radius, radius_err_min, radius_err_max


def _aux_error_estimator(
    val: float,
    val_err_max: float,
    val_err_min: float,
    output: float,
    c: tuple,
    s: tuple,
    method: int,
    silent: bool,
    which: int,
) -> tuple[float, float]:
    """Auxiliary function to estimate the error of a power-law relation."""

    # Handle errors as absolute values
    val_err_min = abs(val_err_min)
    val_err_max = abs(val_err_max)

    # Calculate the output error
    if method == 1:  # Naive error propagation
        val_err = max(val_err_min, val_err_max)
        output_err_max = power_law_error(
            val, val_err, c[0], c[1], s[0], s[1], y=output
        )
        output_err_min = -output_err_max
    elif method == 2:  # Calculate the error at output extremes
        if not silent and any([val_err_min, val_err_max]) == 0.0:
            txt = ["mass", "radius"] if which == 1 else ["radius", "mass"]
            warnings.warn(
                f"Calculating the {txt[0]} error at extremes without "
                + f"a {txt[1]} error generates no {txt[0]} error",
                stacklevel=2,
            )
        output_min = power_law(val - val_err_min, c[0], s[0])
        output_max = power_law(val + val_err_max, c[0], s[0])
        # Calculate the output error. Safe sign
        output_err_min = min(min(output_min, output_max), output) - output
        output_err_max = max(max(output_min, output_max), output) - output
    else:
        raise ValueError("Method not implemented.")

    return output_err_min, output_err_max
