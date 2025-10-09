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

"""Module to manage diverse mass-radius models.

This module provides tools for estimating the mass (or radius) from the radius
(or mass), using diverse power law models.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import warnings
from typing import Tuple, Union

import numpy as np

from resokit.units import convert

# =============================================================================
# FUNCTIONS
# =============================================================================


def power_law(x: float, c: float, s: float, x0: float = 1.0) -> float:
    r"""Calculate a power-law.

    Equation: :math:`y = c \times \\left(\frac{x}{x}\\right)^s`

    Parameters
    ----------
    x : float
        Value to calculate the power-law.
    c : float
        Constant of the power-law relation.
    s : float
        Slope of the power-law relation.
    x0 : float, optional
        Reference value of the power-law.
        Default is 1.0.

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

    return np.sqrt(
        (y / c * c_err) ** 2  # dy/dc
        + (y * np.log(x / x0) * s_err) ** 2  # dy/ds
        + (y * s / x * x_err) ** 2  # dy/dx
        + (y * s / x0 * x0_err) ** 2  # dy/dx0
    )


def chen_kipp_2017_radius(mass: float) -> Tuple[float, tuple, tuple, tuple]:
    r"""Calculate the radius of a planet using the Chen & Kipping (2017).

    Power law approximation:
        :math:`radius = C \times mass^S`
    Citation:
        Chen, J., & Kipping, D. 2017, ApJ, 834, 17
    For a complete implementation of the method, see:
        https://github.com/chenjj2/forecaster

    Parameters
    ----------
    mass : float
        Mass of the planet, in Earth masses.

    Returns
    -------
    radius : float
        Radius of the planet, in Earth radii.
    c : tuple
        Constant of the power-law relation and its error.
    s : tuple
        Slope of the power-law relation and its error.
    x0 : tuple
        Reference value of the power-law and its error.
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
    # Reference value
    x0 = (1.0, 0.0)
    # Transition mass
    m1_tr = 2.04
    m2_tr = convert(
        0.414, from_units="mj", to_units="me"
    )  # 0.414 Jupiter masses
    m3_tr = convert(0.08, from_units="ms", to_units="me")  # 0.08 Solar masses

    if mass < m1_tr:  # First branch
        return power_law(mass, c1[0], s1[0], x0[0]), c1, s1, x0
    elif mass < m2_tr:  # Second branch
        return power_law(mass, c2[0], s2[0], x0[0]), c2, s2, x0
    elif mass < m3_tr:  # Third branch
        return power_law(mass, c3[0], s3[0], x0[0]), c3, s3, x0
    # Fourth branch
    return power_law(mass, c4[0], s4[0], x0[0]), c4, s4, x0


def chen_kipp_2017_mass(
    radius: float, trivariate: tuple = (0.15, 0.8), silent: bool = False
) -> Tuple[float, tuple, tuple, tuple]:
    r"""Calculate the mass of a planet using the Chen & Kipping (2017).

    Power law approximation:
        :math:`mass = x_0 \frac{1}{C} \times radius^{1/S}`
    Citation:
        Chen, J., & Kipping, D. 2017, ApJ, 834, 17
    For a complete implementation of the method, see:
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
    mass : float
        Mass of the planet, in Earth masses.
    inv_c : tuple
        Inverse of the constant of the power-law relation, and its error.
    inv_s : tuple
        Inverse of the slope of the power-law relation, and its error.
    x0 : tuple
        Reference value of the power-law, and its error.
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
    # Reference value
    x0 = (1.0, 0.0)
    # Transition radius
    r1_tr = (1.229836, 0.111458)
    r2_tr = (14.31101, 4.529131)
    r3_tr = (11.328892, 4.333345)

    # We use the inverse of the constant and slope,
    # so the error is recaclulated with propagation error.

    if radius < r1_tr[0]:  # First branch
        return (
            power_law(radius, x0[0], 1.0 / s1[0], c1[0]),
            x0,
            (1.0 / s1[0], s1[1] / s1[0]),
            c1,
        )
    elif radius > r2_tr[0]:  # Pure fourth branch
        return (
            power_law(radius, x0[0], 1.0 / s4[0], c4[0]),
            x0,
            (1.0 / s4[0], s4[1] / s4[0]),
            c4,
        )
    elif radius < r3_tr[0]:  # Pure second branch
        return (
            power_law(radius, x0[0], 1.0 / s2[0], c2[0]),
            x0,
            (1.0 / s2[0], s2[1] / s2[0]),
            c2,
        )

    # Trivariate region
    if not silent:
        warnings.warn(
            "Radius falls in the trivariate region: "
            + f"{r3_tr[0]} < R < {r2_tr[0]}"
            + "\n The mass-radius relation may not be accurate.",
            stacklevel=2,
        )

    if isinstance(trivariate, (float, int)) or len(trivariate) != 2:
        raise ValueError("Trivariate must be a tuple|list with length 2.")

    sumb = sum(trivariate)  # Probability of second branch
    if sumb < 0 or sumb > 1:
        raise ValueError("Sum of trivariate must be a number between 0 and 1.")

    prob = np.random.rand()  # Get a random probability

    if prob < trivariate[0]:  # Use second branch
        return (
            power_law(radius, x0[0], 1.0 / s2[0], c2[0]),
            x0,
            (1.0 / s2[0], s2[1] / s2[0]),
            c2,
        )
    elif prob < sumb:  # Use third branch
        return (
            power_law(radius, x0[0], 1.0 / s3[0], c3[0]),
            x0,
            (1.0 / s3[0], s3[1] / s3[0]),
            c3,
        )
    # Use fourth branch
    return (
        power_law(radius, x0[0], 1.0 / s4[0], c4[0]),
        x0,
        (1.0 / s4[0], s4[1] / s4[0]),
        c4,
    )


def otegi_2020_radius(
    mass: float,
    density: float = 0.0,
    bivariate: float = 0.5,
    silent: bool = False,
) -> Tuple[float, tuple, tuple, tuple]:
    r"""Calculate the radius of a planet using Otegi et al. (2020).

    Power law approximation:
        :math:`radius = x_0 C \times mass^S`
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
        region is calculated with lower (rho >= 3300 kg m^-3) branch, instead
        of using the upper (rho < 3300 kg m^-3) branch of the power-law
        approximation. Must be a number between 0 and 1.
    silent : bool, optional. Default: False
        Whether to silence the warning if the radius is greater than the
        maximum value used by Otegi et al. (2020), or if the estimation falls
        in a multivariate region.

    Returns
    -------
    radius : float
        Radius of the planet, in Earth radii.
    c : tuple
        Constant of the power-law relation and its error.
    s : tuple
        Slope of the power-law relation and its error.
    x0 : tuple
        Reference value of the power-law and its error.
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
    c1 = (1.03, 0.02)  # lower branch: >= 3300 kg m^-3
    c2 = (0.70, 0.11)  # upper branch: < 3300 kg m^-3
    # Slopes
    s1 = (0.29, 0.01)  # lower branch: >= 3300 kg m^-3
    s2 = (0.63, 0.04)  # upper branch: < 3300 kg m^-3
    # Reference value
    x0 = (1.0, 0.0)
    # Density cut
    dens_cut = 3300  # kg m^-3

    # If densityty is set
    if density > 0.0:  # If density is set
        if density >= dens_cut:  # Dense planet
            return power_law(mass, c1[0], s1[0], x0[0]), c1, s1, x0
        return power_law(mass, c2[0], s2[0], x0[0]), c2, s2, x0

    # Naive mass aproach
    if mass < 5:  # Small dense planet
        return power_law(mass, c1[0], s1[0], x0[0]), c1, s1, x0
    elif mass > 40:  # Large subdense planet
        return power_law(mass, c2[0], s2[0], x0[0]), c2, s2, x0

    # In density not set... Get dens_cut in [M_ear R_ear^-3]
    scaled_dens = convert(
        dens_cut, from_units=("kg", "m"), to_units=("me", "re"), power=(1, -3)
    )

    # Try both branches
    radius1 = power_law(mass, c1[0], s1[0], x0[0])
    radius2 = power_law(mass, c2[0], s2[0], x0[0])

    # Calculate the density
    density1 = mass / (4 / 3 * np.pi * radius1**3)
    density2 = mass / (4 / 3 * np.pi * radius2**3)

    # Check if lower or upper branch

    if density1 >= scaled_dens and density2 > scaled_dens:
        # First branch is valid
        return radius1, c1, s1, x0
    elif density1 < scaled_dens and density2 < scaled_dens:
        # Second branch is valid
        return radius2, c2, s2
    elif density1 >= scaled_dens and density2 < scaled_dens:
        if not silent:
            warnings.warn(
                "The estimation falls in a multivariate region. "
                + "The power-law approximation may not be accurate.",
                stacklevel=2,
            )
        # Both branches are valid.
        if not isinstance(bivariate, (int, float)) or (
            bivariate < 0 or bivariate > 1
        ):
            raise ValueError("Bivariate must be a number between 0 and 1.")

        if np.random.rand() < bivariate:  # Use first branch
            return radius1, c1, s1, x0
        return radius2, c2, s2, x0  # Use second branch

    # If nothing works, we use the closest branch
    if abs(density1 - scaled_dens) < abs(density2 - scaled_dens):
        # First branch is closer
        return radius1, c1, s1, x0

    # Second branch is closer
    return radius2, c2, s2, x0


def otegi_2020_mass(
    radius: float,
    density: float = 0.0,
    bivariate: float = 0.5,
    silent: bool = False,
) -> Tuple[float, tuple, tuple, tuple]:
    r"""Calculate the mass of a planet using Otegi et al. (2020).

    Power law approximation:
        :math:`mass = x_0 \frac{1}{C} \times radius^{1/S}`
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
        region is calculated with lower (rho >= 3300 kg m^-3) branch, instead
        of using the upper (rho < 3300 kg m^-3) branch of the power-law
        approximation. Must be a number between 0 and 1.
    silent : bool, optional. Default: False
        Whether to print a warning if the radius is greater than the maximum
        value used by Otegi et al. (2020), or if the estimation falls
        in a multivariate region.

    Returns
    -------
    mass : float
        Mass of the planet, in Earth masses.
    inv_c : tuple
        Inverse of the constant of the power-law relation, and its error.
    inv_s : tuple
        Inverse of the slope of the power-law relation, and its error.
    x0 : tuple
        Reference value of the power-law, and its error.
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
    c1 = (0.90, 0.06)  # lower branch: >= 3300 kg m^-3
    c2 = (1.74, 0.38)  # upper branch: < 3300 kg m^-3
    # Slopes
    s1 = (3.45, 0.12)  # lower branch: >= 3300 kg m^-3
    s2 = (1.58, 0.10)  # upper branch: < 3300 kg m^-3
    # Reference value
    x0 = (1.0, 0.0)
    # Density cut
    dens_cut = 3300  # kg m^-3

    # If densityty is set
    if density > 0.0:  # If density is set
        if density >= dens_cut:  # Dense planet
            return power_law(radius, c1[0], s1[0], x0[0]), c1, s1, x0
        return power_law(radius, c2[0], s2[0], x0[0]), c2, s2, x0

    # Naive radius aproach
    if radius < 1.5:  # Small dense planet
        return power_law(radius, c1[0], s1[0], x0[0]), c1, s1, x0
    elif radius > 3.1:  # Large subdense planet
        return power_law(radius, c2[0], s2[0], x0[0]), c2, s2, x0

    # In density not set... Get dens_cut in [M_ear R_ear^-3]
    scaled_dens = convert(
        dens_cut, from_units=("kg", "m"), to_units=("me", "re"), power=(1, -3)
    )

    # Try both branches
    mass1 = power_law(radius, c1[0], s1[0], x0[0])
    mass2 = power_law(radius, c2[0], s2[0], x0[0])

    # Calculate the density
    density1 = mass1 / (4 / 3 * np.pi * radius**3)
    density2 = mass2 / (4 / 3 * np.pi * radius**3)

    # Check if lower or upper branch

    if density1 >= scaled_dens and density2 > scaled_dens:
        # First branch is valid
        return mass1, c1, s1, x0
    elif density1 < scaled_dens and density2 < scaled_dens:
        # Second branch is valid
        return mass2, c2, s2, x0
    elif density1 >= scaled_dens and density2 < scaled_dens:
        if not silent:
            warnings.warn(
                "The estimation falls in a multivariate region. "
                + "The power-law approximation may not be accurate.",
                stacklevel=2,
            )
        # Both branches are valid.
        if not isinstance(bivariate, (int, float)) or (
            bivariate < 0 or bivariate > 1
        ):
            raise ValueError("Bivariate must be a number between 0 and 1.")

        if np.random.rand() < bivariate:  # Use first branch
            return mass1, c1, s1, x0
        return mass2, c2, s2, x0  # Use second branch

    # If nothing works, we use the closest branch
    if abs(density1 - scaled_dens) < abs(density2 - scaled_dens):
        # First branch is closer
        return mass1, c1, s1, x0

    # Second branch is closer
    return mass2, c2, s2, x0


def edmonson_2023_radius(mass: float) -> Tuple[float, tuple, tuple, tuple]:
    r"""Calculate the radius of a planet using the Edmondson et al. (2023).

    Power law approximation:
        :math:`radius = x_0 C \times mass^S`
    Citation:
        Edmondson, K., Norris, J., & Kerins, E. 2023, Open J. Astrophysics,
        submitted [arXiv:2310.16733]

    Parameters
    ----------
    mass : float
        Mass of the planet, in Earth masses.

    Returns
    -------
    radius : float
        Radius of the planet, in Earth radii.
    c : tuple
        Constant of the power-law relation and its error.
    s : tuple
        Slope of the power-law relation and its error.
    x0 : tuple
        Reference value of the power-law and its error.
    """
    # Constants
    c1 = (1.01, 0.03)
    c2 = (0.53, 0.05)
    c3 = (13, 1.2)
    # Slopes
    s1 = (0.28, 0.03)
    s2 = (0.68, 0.02)
    s3 = (0.012, 0.003)
    # Reference value
    x0 = (1.0, 0.0)
    # Transition mass
    m1_tr = 4.95
    m2_tr = 115

    if mass < m1_tr:  # First branch
        return power_law(mass, c1[0], s1[0], x0[0]), c1, s1, x0
    elif mass < m2_tr:  # Second branch
        return power_law(mass, c2[0], s2[0], x0[0]), c2, s2, x0
    # Third branch
    return power_law(mass, c3[0], s3[0], x0[0]), c3, s3, x0


def edmonson_2023_mass(radius: float) -> Tuple[float, tuple, tuple, tuple]:
    r"""Calculate the mass of a planet using the Edmondson et al. (2023).

    Power law approximation:
        :math:`mass = x_0 \frac{1}{C} \times radius^{1/S}`
    Citation:
        Edmondson, K., Norris, J., & Kerins, E. 2023, Open J. Astrophysics,
        submitted [arXiv:2310.16733]

    Parameters
    ----------
    radius : float
        Radius of the planet, in Earth radii.

    Returns
    -------
    mass : float
        Mass of the planet, in Earth masses.
    inv_c : tuple
        Inverse of the constant of the power-law relation, and its error.
    inv_s : tuple
        Inverse of the slope of the power-law relation, and its error.
    x0 : tuple
        Reference value of the power-law, and its error.
    """
    # Constants
    c1 = (1.01, 0.03)
    c2 = (0.53, 0.05)
    c3 = (13, 1.2)
    # Slopes
    s1 = (0.28, 0.03)
    s2 = (0.68, 0.02)
    s3 = (0.012, 0.003)
    # Reference value
    x0 = (1.0, 0.0)
    # Transition radius
    r1_tr = power_law(4.95, c1[0], s1[0], x0[0])  # 4.95 Earth masses
    r2_tr = power_law(115, c2[0], s2[0], x0[0])  # 115 Earth masses
    rmax = power_law(1e4, c3[0], s3[0], x0[0])  # 10000 earth masses

    # No multivariate region, because the power-law is always defined positive.

    if radius < r1_tr:  # First branch
        return (
            power_law(radius, x0[0], 1.0 / s1[0], c1[0]),
            x0,
            (1.0 / s1[0], s1[1] / s1[0]),
            c1,
        )
    elif radius < r2_tr:  # Second branch
        return (
            power_law(radius, x0[0], 1.0 / s2[0], c2[0]),
            x0,
            (1.0 / s2[0], s2[1] / s2[0]),
            c2,
        )
    elif radius > rmax:  # No estimation
        print(radius, rmax)
        raise ValueError(
            "Radius is greater than the maximum value used by "
            + f"Edmonson et al. (2023): R = {rmax} R_earth."
        )

    # Third branch
    return (
        power_law(radius, x0[0], 1.0 / s3[0], c3[0]),
        x0,
        (1.0 / s3[0], s3[1] / s3[0]),
        c3,
    )


def muller_2024_radius(mass: float) -> Tuple[float, tuple, tuple, tuple]:
    r"""Calculate the radius of a planet using the Müller et al. (2024).

    Power law approximation:
        :math:`radius = x_0 C \times mass^S`
    Citation:
        Müller S., Baron J., Helled R., Bouchy F. & Parc L. 2024, A&A, 686, A296

    Parameters
    ----------
    mass : float
        Mass of the planet, in Earth masses.

    Returns
    -------
    radius : float
        Radius of the planet, in Earth radii.
    c : tuple
        Constant of the power-law relation and its error.
    s : tuple
        Slope of the power-law relation and its error.
    x0 : tuple
        Reference value of the power-law and its error.
    """
    # Constants
    c1 = (1.02, 0.03)
    c2 = (0.56, 0.03)
    c3 = (18.6, 6.7)
    # Slopes
    s1 = (0.27, 0.04)
    s2 = (0.67, 0.05)
    s3 = (-0.06, 0.07)
    # Reference value
    x0 = (1.0, 0.0)
    # Transition mass
    m1_tr = 4.37
    m2_tr = 127

    if mass < m1_tr:  # First branch
        return power_law(mass, c1[0], s1[0], x0[0]), c1, s1, x0
    elif mass < m2_tr:  # Second branch
        return power_law(mass, c2[0], s2[0], x0[0]), c2, s2, x0
    # Third branch
    return power_law(mass, c3[0], s3[0], x0[0]), c3, s3, x0


def muller_2024_mass(
    radius: float, bivariate: float = 0.5, silent: bool = False
) -> Tuple[float, tuple, tuple, tuple]:
    r"""Calculate the mass of a planet using the Müller et al. (2024).

    Power law approximation:
        :math:`mass = x_0 \frac{1}{C} \times radius^{1/S}`
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
    mass : float
        Mass of the planet, in Earth masses.
    inv_c : tuple
        Inverse of the constant of the power-law relation, and its error.
    inv_s : tuple
        Inverse of the slope of the power-law relation, and its error.
    x0 : tuple
        Reference value of the power-law, and its error.
    """
    # Constants
    c1 = (1.02, 0.03)
    c2 = (0.56, 0.03)
    c3 = (18.6, 6.7)
    # Slopes
    s1 = (0.27, 0.04)
    s2 = (0.67, 0.05)
    s3 = (-0.06, 0.07)
    # Reference value
    x0 = (1.0, 0.0)
    # Transition radius
    r1_tr = 1.64
    r2_tr = power_law(127, c2[0], s2[0], x0[0])  # 127 Earth masses
    r3_tr = power_law(1e4, c3[0], s3[0], x0[0])  # Top: 1e4 Earth masses

    # We use the inverse of the constant and slope,
    # so the error is recaclulated with propagation error.

    if radius < r1_tr:  # First branch
        return (
            power_law(radius, x0[0], 1.0 / s1[0], c1[0]),
            x0,
            (1.0 / s1[0], s1[1] / s1[0]),
            c1,
        )
    elif radius < r3_tr:  # Pure second branch
        return (
            power_law(radius, x0[0], 1.0 / s2[0], c2[0]),
            x0,
            (1.0 / s2[0], s2[1] / s2[0]),
            c2,
        )
    elif radius > r2_tr:  # No estimation
        raise ValueError(
            "Radius is greater than the maximum value used by "
            + f"Müller et al. (2024): R = {r2_tr} R_earth."
        )

    # Bivariate region
    if not silent:
        warnings.warn(
            "Radius falls in the bivariate region: "
            + f"{r3_tr} < R < {r2_tr}"
            + "\n The mass-radius relation may not be accurate.",
            stacklevel=2,
        )
    if (not isinstance(bivariate, (int, float))) or (
        bivariate < 0 or bivariate > 1
    ):
        raise ValueError("Bivariate must be a number between 0 and 1.")

    if np.random.rand() < bivariate:  # Second branch
        return (
            power_law(radius, x0[0], 1.0 / s2[0], c2[0]),
            x0,
            (1.0 / s2[0], s2[1] / s2[0]),
            c2,
        )
    # Third branch
    return (
        power_law(radius, x0[0], 1.0 / s3[0], c3[0]),
        x0,
        (1.0 / s3[0], s3[1] / s3[0]),
        c3,
    )


def estimate_mass_single(
    radius: float,
    radius_err_min: float = 0.0,
    radius_err_max: float = 0.0,
    model: str = "ck17",
    multivariate: float = 0.5,
    err_method: int = 0,
    density: float = 0.0,
    silent: bool = False,
) -> Tuple[float, float, float]:
    r"""Calculate the mass of a planet using a power-law approximation.

    Equation:
        :math:`mass = \frac{1}{C} \times radius^{1/S}`

    Parameters
    ----------
    radius : float
        Radius of the planet, in Earth radii.
    radius_err_min : float
        Lower error of the radius, in Earth radii.
    radius_err_max : float
        Upper error of the radius, in Earth radii.
    model : str, optional. Default: "ck17"
        Model to use for the mass-radius power-law relation.
        'ck17': Chen & Kipping (2017) [trivariate]
        'o20': Otegi et al. (2020) [density|bivariate]
        'e23': Edmondson et al. (2023)
        'm24': Müller et al. (2024) [bivariate]
    multivariate : float, tuple, optional. Default: 0.5
        Probability of using the (first, second, ...) branch if the estimation
        falls in a multivariate region.
        For bivariate models ('o20', 'm24'), it must be a float between
        0 and 1.
        For trivariate model "ck17", it must be a tuple of two floats between
        0 and 1, where the sum of them must be lower equal than 1.
    err_method : int, optional. Default: 0
        Which method implement for error calculation.
        Method 0: Do not calculate errors. Return both as 0.0.
        Method 1: (Naive) Error propagation with the power-law approximation,
        using the radius error as the maximum of the two extremes.
        Warning: May return excessively large errors for multivariate
        sections.
        Method 2: Evaluate the radius extremes and calculate each mass
        extreme with the power-law approximation.
        Method 3: Returns the approximate model error as value errors.
    density : float, optional. Default: 0.0
        Density of the planet, in kg m^-3.
        Only used if model is 'o20'. If equal to 0.0, the code uses
        multivariate float instead, to determine which branch to use.
    silent : bool, optional. Default: False
        Whether to silence the warning if the radius falls in a
        multivariate region, or if the estimation is not accurate.

    Returns
    -------
    mass : float
        Mass of the planet, in Earth masses.
    mass_err_min : float
        Lower error of the mass, in Earth masses. If err_method=0, it is 0.0.
    mass_err_max : float
        Upper error of the mass, in Earth masses. If err_method=0, it is 0.0.
    """
    # Check if radius is NaN
    if np.isnan(radius):
        return np.nan, np.nan, np.nan
    # Calculate the mass
    if model == "ck17":
        mass, c, s, x0 = chen_kipp_2017_mass(
            radius, trivariate=multivariate, silent=silent
        )
    elif model == "o20":
        mass, c, s, x0 = otegi_2020_mass(
            radius, density=density, bivariate=multivariate, silent=silent
        )
    elif model == "e23":
        mass, c, s, x0 = edmonson_2023_mass(radius)
    elif model == "m24":
        mass, c, s, x0 = muller_2024_mass(
            radius, bivariate=multivariate, silent=silent
        )
    else:
        raise ValueError(
            f"Model {model} not implemented. Use 'ck17', 'o20', 'e23' or 'm24'."
        )

    # Calculate the mass error
    if err_method == 0:
        return mass, 0.0, 0.0

    # Warn if necessary
    if not silent and model in ["ck17", "o20", "m24"] and err_method == 1:
        warnings.warn(
            "Using the naive error propagation method may generate"
            + " excessively large errors in multivariate sections",
            stacklevel=2,
        )

    # Use auxiliar error function
    mass_err_min, mass_err_max = _aux_error_estimator(
        radius,
        radius_err_max,
        radius_err_min,
        mass,
        c,
        s,
        x0,
        err_method,
        silent,
        0,
    )

    return mass, mass_err_min, mass_err_max


def estimate_radius_single(
    mass: float,
    mass_err_min: float = 0.0,
    mass_err_max: float = 0.0,
    model: str = "ck17",
    bivariate: float = 0.5,
    err_method: int = 0,
    density: float = 0.0,
    silent: bool = False,
) -> Tuple[float, float, float]:
    r"""Calculate the radius of a planet using the power-law approximation.

    Equation:
        :math:`radius = C \times mass^S`

    Parameters
    ----------
    mass : float
        Mass of the planet, in Earth masses.
    mass_err_min : float
        Lower error of the mass, in Earth masses.
    mass_err_max : float
        Upper error of the mass, in Earth masses.
    model : str, optional. Default: "ck17"
        Model to use for the mass-radius power-law relation.
        'ck17': Chen & Kipping (2017)
        'o20': Otegi et al. (2020) [density|bivariate]
        'e23': Edmondson et al. (2023)
        'm24': Müller et al. (2024)
    bivariate : float, optional. Default: 0.5
        Probability of using the lower branch if the estimation falls in a
        bivariate region. Must be a number between 0 and 1.
        Only used if model is 'o20'.
    err_method : int, optional. Default: 0
        Which method implement for error calculation.
        Method 0: Do not calculate errors. Return both as 0.0.
        Method 1: (Naive) Error propagation with the power-law approximation,
        using the mass error as the maximum of the two extremes.
        Method 2: Evalaute the mass extremes and calculate each radius extreme.
        Method 3: Returns the approximate model error as value errors.
    density : float, optional. Default: 0.0
        Density of the planet, in kg m^-3.
        Only used if model is 'o20'. If equal to 0.0, the code uses
        bivariate float value instead, to determine which branch to use.
    silent : bool, optional. Default: False
        Whether to silence the warning if the radius falls in a
        bivariate region, or if the estimation is not accurate.

    Returns
    -------
    radius : float
        Radius of the planet, in Earth radii.
    radius_err_min : float
        Lower error of the radius, in Earth radii. If err_method=0, it is 0.0.
    radius_err_max : float
        Upper error of the radius, in Earth radii. If err_method=0, it is 0.0.
    """
    # Check if mass is NaN
    if np.isnan(mass):
        return np.nan, np.nan, np.nan
    # Calculate the radius
    if model == "ck17":
        radius, c, s, x0 = chen_kipp_2017_radius(mass)
    elif model == "o20":
        radius, c, s, x0 = otegi_2020_radius(
            mass, density=density, bivariate=bivariate, silent=silent
        )
    elif model == "e23":
        radius, c, s, x0 = edmonson_2023_radius(mass)
    elif model == "m24":
        radius, c, s, x0 = muller_2024_radius(mass)
    else:
        raise ValueError(
            f"Model {model} not implemented. Use 'ck17', 'o20', 'e23' or 'm24'."
        )

    # Calculate the radius error
    if err_method == 0:
        return radius, 0.0, 0.0

    # Warn if necessary
    if not silent and model == "o20" and err_method == 1:
        warnings.warn(
            "Using the naive error propagation method may generate"
            + " excessively large errors in multivariate sections",
            stacklevel=2,
        )

    # Use auxiliar error function
    radius_err_min, radius_err_max = _aux_error_estimator(
        mass,
        mass_err_max,
        mass_err_min,
        radius,
        c,
        s,
        x0,
        err_method,
        silent,
        1,
    )

    return radius, radius_err_min, radius_err_max


def _aux_error_estimator(
    val: float,
    val_err_max: float,
    val_err_min: float,
    output: float,
    c: tuple,
    s: tuple,
    x0: tuple,
    method: int,
    silent: bool,
    which: int,
) -> Tuple[float, float]:
    """Auxiliary function to estimate the error of a power-law relation."""
    # Handle errors as absolute values
    val_err_min = abs(val_err_min)
    val_err_max = abs(val_err_max)

    # Calculate the output error
    if method == 1:  # Naive error propagation
        val_err = max(val_err_min, val_err_max)
        output_err_max = power_law_error(
            val, val_err, c[0], c[1], s[0], s[1], x0[0], x0[1], y=output
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
        output_min = power_law(val - val_err_min, c[0], s[0], x0[0])
        output_max = power_law(val + val_err_max, c[0], s[0], x0[0])
        # Calculate the output error. Safe sign
        output_err_min = min(min(output_min, output_max), output) - output
        output_err_max = max(max(output_min, output_max), output) - output
    elif (
        method == 3
    ):  # Give the error from the model [eg.: y+ = pw(x, c+, s+, x0-]
        output_min = power_law(val, c[0] - c[1], s[0] - s[1], x0[0] + x0[1])
        output_max = power_law(val, c[0] + c[1], s[0] + s[1], x0[0] - x0[1])
        # Calculate the output error. Safe sign
        output_err_min = min(min(output_min, output_max), output) - output
        output_err_max = max(max(output_min, output_max), output) - output
    else:
        raise ValueError(f"Error method '{method}' not implemented.")

    return output_err_min, output_err_max


estimate_mass_vec = np.vectorize(
    estimate_mass_single,
    doc="Vectorized version of :py:func:`estimate_mass_single`.",
    excluded=["model", "multivariate", "err_method", "density", "silent"],
)


estimate_radius_vec = np.vectorize(
    estimate_radius_single,
    doc="Vectorized version of :py:func:`estimate_radius_single`.",
    excluded=["model", "bivariate", "err_method", "density", "silent"],
)


def estimate_radius(
    mass: Union[float, np.ndarray],
    mass_err_min: Union[float, np.ndarray] = 0.0,
    mass_err_max: Union[float, np.ndarray] = 0.0,
    model: str = "ck17",
    bivariate: float = 0.5,
    err_method: int = -1,
    density: float = 0.0,
    silent: bool = False,
) -> Union[Tuple[float, float, float], np.ndarray]:
    r"""Calculate the radius of a planet using the power-law approximation.

    The different models available are:
    Chen & Kipping (2017),
    Otegi et al. (2020),
    Edmondson et al. (2023),
    and Müller et al. (2024).

    Equation:
        :math:`radius = C \times mass^S`

    Parameters
    ----------
    mass : float, np.ndarray
        Mass of the planet, in Earth masses.
    mass_err_min : float, np.ndarray
        Lower error of the mass, in Earth masses.
    mass_err_max : float, np.ndarray
        Upper error of the mass, in Earth masses.
    model : str, optional. Default: "ck17"
        Model to use for the mass-radius power-law relation.
        'ck17': Chen & Kipping (2017)
        'o20': Otegi et al. (2020) [density|bivariate]
        'e23': Edmondson et al. (2023)
        'm24': Müller et al. (2024)
    bivariate : float, optional. Default: 0.5
        Probability of using the lower branch if the estimation falls in a
        bivariate region. Must be a number between 0 and 1.
        Only used if model is 'o20'.
    err_method : int, optional. Default: -1
        Which method implement for error calculation.
        Method -1: Do not calculate errors. Return only the radius.
        Method 0: Do not calculate errors. Return both as 0.0.
        Method 1: (Naive) Error propagation with the power-law approximation,
        using the mass error as the maximum of the two extremes.
        Method 2: Evalaute the mass extremes and calculate each radius extreme.
        Method 3: Returns the approximate model error as value errors.
    density : float, optional. Default: 0.0
        Density of the planet, in kg m^-3.
        Only used if model is 'o20'. If equal to 0.0, the code uses
        bivariate
    silent : bool, optional. Default: False
        Whether to silence the warning if the radius falls in a
        bivariate region, or if the estimation is not accurate.

    Returns
    -------
    result : tuple[float, float, float] or np.ndarray
        Estimated radius, and its minimum and maximum errors, in Earth radii.
        If mass is a scalar, the tuple is
        (radius, radius_err_min, radius_err_max),
        else it is a (n,3) numpy array.
        If `err_method=0`, the tuple | array is (radius, 0.0, 0.0).

    References
    ----------
        `Chen, J., & Kipping, D. 2017, ApJ, 834, 17
        <https://ui.adsabs.harvard.edu/abs/2017ApJ...834...17C>`_

        `Otegi, J. F., Bouchy, F., & Helled, R. 2020, A&A, 634, A43
        <https://ui.adsabs.harvard.edu/abs/2020A&A...634A..43O>`_

        `Edmondson, K., Norris, J., & Kerins, E. 2023, Open J. Astrophysics,
        submitted <https://arxiv.org/abs/2310.16733>`_

        `Müller S., Baron J., Helled R., Bouchy F. & Parc L. 2024, A&A, 686,
        A296 <https://ui.adsabs.harvard.edu/abs/2024A&A...686A.296M>`_
    """
    err_method_aux = 0 if err_method == -1 else err_method
    if isinstance(mass, (int, float)):
        radius, radius_err_min, radius_err_max = estimate_radius_single(
            mass=mass,
            mass_err_min=mass_err_min,
            mass_err_max=mass_err_max,
            model=model,
            bivariate=bivariate,
            err_method=err_method_aux,
            density=density,
            silent=silent,
        )
    else:
        radius, radius_err_min, radius_err_max = estimate_radius_vec(
            mass=mass,
            mass_err_min=mass_err_min,
            mass_err_max=mass_err_max,
            model=model,
            bivariate=bivariate,
            err_method=err_method_aux,
            density=density,
            silent=silent,
        )

    # Return only the radius?
    if err_method == -1:
        return radius

    return np.array(
        [radius, -radius_err_min, radius_err_max]
    ).T  # -for consistency


def estimate_mass(
    radius: Union[float, np.ndarray],
    radius_err_min: Union[float, np.ndarray] = 0.0,
    radius_err_max: Union[float, np.ndarray] = 0.0,
    model: str = "ck17",
    multivariate: Union[float, tuple, list] = None,
    err_method: int = -1,
    density: float = 0.0,
    silent: bool = False,
) -> Union[Tuple[float, float, float], np.ndarray]:
    r"""Calculate the mass of a planet using a power-law approximation.

    The different models available are:
    Chen & Kipping (2017),
    Otegi et al. (2020),
    Edmondson et al. (2023),
    and Müller et al. (2024).

    Equation:
        :math:`mass = \frac{1}{C} \times radius^{1/S}`

    Parameters
    ----------
    radius : float, np.ndarray
        Radius of the planet, in Earth radii.
    radius_err_min : float, np.ndarray
        Lower error of the radius, in Earth radii.
    radius_err_max : float, np.ndarray
        Upper error of the radius, in Earth radii.
    model : str, optional. Default: "ck17"
        Model to use for the mass-radius power-law relation.
        'ck17': Chen & Kipping (2017) [trivariate]
        'o20': Otegi et al. (2020) [density|bivariate]
        'e23': Edmondson et al. (2023)
        'm24': Müller et al. (2024) [bivariate]
    multivariate : float, tuple, optional. Default: 0.5
        Probability of using the (first, second, ...) branch if the estimation
        falls in a multivariate region.
        For bivariate models ('o20', 'm24'), it must be a float between
        0 and 1. Default is 0.5.
        For trivariate model "ck17", it must be a tuple of two floats between
        0 and 1, where the sum of them must be lower equal than 1. Default is
        (0.1, 0.85).
    err_method : int, optional. Default: -1
        Which method implement for error calculation.
        Method -1: Do not calculate errors. Return only the mass.
        Method 0: Do not calculate errors. Return both as 0.0.
        Method 1: (Naive) Error propagation with the power-law approximation,
        using the radius error as the maximum of the two extremes.
        Warning: May return excessively large errors for multivariate
        sections.
        Method 2: Evaluate the radius extremes and calculate each mass
        extreme with the power-law approximation.
        Method 3: Returns the approximate model error as value errors.
    density : float, optional. Default: 0.0
        Density of the planet, in kg m^-3.
        Only used if model is 'o20'. If equal to 0.0, the code uses
        multivariate float instead, to determine which branch to use.
    silent : bool, optional. Default: False
        Whether to silence the warning if the radius falls in a
        multivariate region, or if the estimation is not accurate.

    Returns
    -------
    result : tuple[float, float, float] or np.ndarray
        Estimated mass, and its minimum and maximum errors, in Earth masses.
        If radius is a scalar, the tuple is (mass, mass_err_min, mass_err_max),
        else it is a (n,3) numpy array.
        If `err_method=0`, the tuple | array is (mass, 0.0, 0.0).

    References
    ----------
        `Chen, J., & Kipping, D. 2017, ApJ, 834, 17
        <https://ui.adsabs.harvard.edu/abs/2017ApJ...834...17C>`_

        `Otegi, J. F., Bouchy, F., & Helled, R. 2020, A&A, 634, A43
        <https://ui.adsabs.harvard.edu/abs/2020A&A...634A..43O>`_

        `Edmondson, K., Norris, J., & Kerins, E. 2023, Open J. Astrophysics,
        submitted <https://arxiv.org/abs/2310.16733>`_

        `Müller S., Baron J., Helled R., Bouchy F. & Parc L. 2024, A&A, 686,
        A296 <https://ui.adsabs.harvard.edu/abs/2024A&A...686A.296M>`_
    """
    err_method_aux = 0 if err_method == -1 else err_method
    if multivariate is None:  # Set default values
        if model == "ck17":
            multivariate = (0.1, 0.85)
        else:
            multivariate = 0.5
    if isinstance(radius, (int, float)):
        mass, mass_err_min, mass_err_max = estimate_mass_single(
            radius=radius,
            radius_err_min=radius_err_min,
            radius_err_max=radius_err_max,
            model=model,
            multivariate=multivariate,
            err_method=err_method_aux,
            density=density,
            silent=silent,
        )
    else:
        mass, mass_err_min, mass_err_max = estimate_mass_vec(
            radius=radius,
            radius_err_min=radius_err_min,
            radius_err_max=radius_err_max,
            model=model,
            multivariate=multivariate,
            err_method=err_method_aux,
            density=density,
            silent=silent,
        )

    # Return only the mass?
    if err_method == -1:
        return mass

    return np.array([mass, -mass_err_min, mass_err_max]).T  # -for consistency
