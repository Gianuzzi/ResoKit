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

"""Module with utilities for unit conversion and manipulation."""

# =============================================================================
# IMPORTS
# =============================================================================

from numpy import pi

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

# Density of Sun in kg/m^3
RHO_SUN = M_SUN / (4.0 / 3.0 * pi * R_SUN**3)  # kg/m^3
# Density of Jupiter in kg/m^3
RHO_JUP = M_JUP / (4.0 / 3.0 * pi * R_JUP**3)  # kg/m^3
# Density of Earth in kg/m^3
RHO_EARTH = M_EAR / (4.0 / 3.0 * pi * R_EAR**3)  # kg/m^3

# Hour in seconds
HOUR = 3600.0  # s
# Day in seconds
DAY = 86400.0  # s
# Year in seconds
YEAR = 3.154e7  # s

# Gravitational constant in AU^3 M_sun^-1 days^-2
G_ASD = G * AU**3 / M_SUN / DAY**2

# UNIT CONVERSIONS (Overkill, but useful)
# Mass conversions
Me2Mj = M_EAR / M_JUP  # Earth to Jupiter mass
Mj2Me = M_JUP / M_EAR  # Jupiter to Earth mass
Me2Ms = M_EAR / M_SUN  # Earth to Solar mass
Ms2Me = M_SUN / M_EAR  # Solar to Earth mass
Mj2Ms = M_JUP / M_SUN  # Jupiter to Solar mass
Ms2Mj = M_SUN / M_JUP  # Solar to Jupiter mass
# Radius conversions
Re2Rj = R_EAR / R_JUP  # Earth to Jupiter radius
Rj2Re = R_JUP / R_EAR  # Jupiter to Earth radius
Re2Rs = R_EAR / R_SUN  # Earth to Solar radius
Rs2Re = R_SUN / R_EAR  # Solar to Earth radius
Rj2Rs = R_JUP / R_SUN  # Jupiter to Solar radius
Rs2Rj = R_SUN / R_JUP  # Solar to Jupiter radius
# Distance conversions
AU2PC = AU / PC  # AU to parsec
PC2AU = PC / AU  # Parsec to AU
# Density conversions
RhoJ2RhoE = RHO_JUP / RHO_EARTH  # Jupiter to Earth density
RhoE2RhoJ = RHO_EARTH / RHO_JUP  # Earth to Jupiter density
RhoJ2RhoS = RHO_JUP / RHO_SUN  # Jupiter to Solar density
RhoS2RhoJ = RHO_SUN / RHO_JUP  # Solar to Jupiter density
RhoE2RhoS = RHO_EARTH / RHO_SUN  # Earth to Solar density
RhoS2RhoE = RHO_SUN / RHO_EARTH  # Solar to Earth density
# Angle conversions
DEG2RAD = pi / 180.0  # Degrees to radians
RAD2DEG = 180.0 / pi  # Radians to degrees
# Time conversions
YEAR2DAY = 365.25  # Years to days
DAY2YEAR = 1.0 / YEAR2DAY  # Days to years
HOUR2DAY = 24.0  # Hours to days
DAY2HOUR = 1.0 / HOUR2DAY  # Days to hours
HOUR2YEAR = HOUR / YEAR  # Hours to years
YEAR2HOUR = YEAR / HOUR  # Years to hours
SEC2HOUR = 1.0 / HOUR  # Seconds to hours
SEC2DAY = 1.0 / DAY  # Seconds to days
SEC2YEAR = 1.0 / YEAR  # Seconds to years
