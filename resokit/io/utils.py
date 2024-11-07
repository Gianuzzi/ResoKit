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

"""Module with internal utility functions for the ResoKit.io module."""

# =============================================================================
# IMPORTS
# =============================================================================

from difflib import SequenceMatcher
import platform
import sys

from resokit import __version__ as VERSION

# =============================================================================
# DEFAULTS
# =============================================================================

_DEFAULT_METADATA = {
    "ResoKit": VERSION,
    "author": "Emmanuel Gianuzzi",
    "author_email": "egianuzzi@unc.edu.ar",
    "affiliation": "FAMAF-IATE-OAC-CONICET",
    "platform": platform.platform(),
    "system_encoding": sys.getfilesystemencoding(),
    "python": sys.version,
    "license": "MIT",
}

# =============================================================================
# UTILS
# =============================================================================

_EU_MAPPING = {
    # Planet columns
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

_NASA_MAPPING = {
    # Planet columns
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
    # System columns
    "sy_snum": "n_stars",
    "sy_pnum": "n_planets",
    # Other columns
    "pl_controv_flag": "controversial",
    "default_flag": "default_set",
    "circumbinary_flag": "circumbinary",
}

# =============================================================================
# FUNCTIONS
# =============================================================================


def __similar(a, b):
    return SequenceMatcher(None, str(a), b).ratio()


def __n_close(a, b, length, n=0):
    stra = str(a)
    return (stra[:length] == str(b)) and (
        (len(stra) == length + n) or stra[length] == " "
    )
