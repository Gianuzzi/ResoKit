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

"""Module with internal utility (parse) functions for the ResoKit package."""

# =============================================================================
# IMPORTS
# =============================================================================

import importlib
import platform
import sys
from difflib import SequenceMatcher
from types import MappingProxyType
from typing import Any, Iterable, Tuple, Union

from pandas import Index, Series

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
        "cb_flag": "circumbinary",
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

# Nasa Query columns to NASA
_NASA_QUERY_MAPPING = MappingProxyType(
    {key: key for key in _NASA_MAPPING.keys()}
)  # Nasa is the same

# EU Query columns to EU
_EU_QUERY_MAPPING = MappingProxyType(
    {
        "target_name": "name",
        "mass": "mass",
        "mass_error_min": "mass_error_min",
        "mass_error_max": "mass_error_max",
        "mass_sin_i": "mass_sini",
        "mass_sin_i_error_min": "mass_sini_error_min",
        "mass_sin_i_error_max": "mass_sini_error_max",
        "radius": "radius",
        "radius_error_min": "radius_error_min",
        "radius_error_max": "radius_error_max",
        "period": "orbital_period",
        "period_error_min": "orbital_period_error_min",
        "period_error_max": "orbital_period_error_max",
        "semi_major_axis": "semi_major_axis",
        "semi_major_axis_error_min": "semi_major_axis_error_min",
        "semi_major_axis_error_max": "semi_major_axis_error_max",
        "eccentricity": "eccentricity",
        "eccentricity_error_min": "eccentricity_error_min",
        "eccentricity_error_max": "eccentricity_error_max",
        "inclination": "inclination",
        "inclination_error_min": "inclination_error_min",
        "inclination_error_max": "inclination_error_max",
        "angular_distance": "angular_distance",
        "discovered": "discovered",
        "periastron": "omega",
        "periastron_error_min": "omega_error_min",
        "periastron_error_max": "omega_error_max",
        "t_peri": "tperi",
        "t_peri_error_min": "tperi_error_min",
        "t_peri_error_max": "tperi_error_max",
        "t_conj": "tconj",
        "t_conj_error_min": "tconj_error_min",
        "t_conj_error_max": "tconj_error_max",
        "tzero_tr": "tzero_tr",
        "tzero_tr_error_min": "tzero_tr_error_min",
        "tzero_tr_error_max": "tzero_tr_error_max",
        "tzero_tr_sec": "tzero_tr_sec",
        "tzero_tr_sec_error_min": "tzero_tr_sec_error_min",
        "tzero_tr_sec_error_max": "tzero_tr_sec_error_max",
        "lambda_angle": "lambda_angle",
        "lambda_angle_error_min": "lambda_angle_error_min",
        "lambda_angle_error_max": "lambda_angle_error_max",
        "impact_parameter": "impact_parameter",
        "impact_parameter_error_min": "impact_parameter_error_min",
        "impact_parameter_error_max": "impact_parameter_error_max",
        "tzero_vr": "tzero_vr",
        "tzero_vr_error_min": "tzero_vr_error_min",
        "tzero_vr_error_max": "tzero_vr_error_max",
        "k": "k",
        "k_error_min": "k_error_min",
        "k_error_max": "k_error_max",
        "temp_calculated": "temp_calculated",
        "temp_measured": "temp_measured",
        "hot_point_lon": "hot_point_lon",
        "albedo": "geometric_albedo",
        "albedo_error_min": "geometric_albedo_error_min",
        "albedo_error_max": "geometric_albedo_error_max",
        "log_g": "log_g",
        "publication_status": "publication",
        "detection_type": "detection_type",
        "mass_detection_type": "mass_measurement_type",
        "radius_detection_type": "radius_measurement_type",
        "species": "molecules",
        "star_name": "star_name",
        "ra": "ra",
        "dec": "dec",
        "mag_v": "mag_v",
        "mag_i": "mag_i",
        "mag_j": "mag_j",
        "mag_h": "mag_h",
        "mag_k": "mag_k",
        "star_distance": "star_distance",
        "star_distance_error_min": "star_distance_error_min",
        "star_distance_error_max": "star_distance_error_max",
        "star_metallicity": "star_metallicity",
        "star_mass": "star_mass",
        "star_radius": "star_radius",
        "star_spec_type": "star_sp_type",
        "star_age": "star_age",
        "star_teff": "star_teff",
        "detected_disc": "star_detected_disc",
        "magnetic_field": "star_magnetic_field",
        "alt_target_name": "alternate_names",  # To edit
        "modification_date": "updated",  # To edit
    }
)

# Query column mappings for the different data sources
QUERY_MAPPINGS = MappingProxyType(
    {"eu": _EU_QUERY_MAPPING, "nasa": _NASA_QUERY_MAPPING}
)

# Missing NASA columns in Query
_NASA_QUERY_MISSING = frozenset()

# Missing EU columns in Query
_EU_QUERY_MISSING = frozenset(
    k for k in _EU_MAPPING.keys() if k not in _EU_QUERY_MAPPING.values()
)

# Missing column in queries
QUERY_MISSING = MappingProxyType(
    {"eu": _EU_QUERY_MISSING, "nasa": _NASA_QUERY_MISSING}
)


# Similarity ratio threshold for the best match
RATIOS_THRESHOLD = 0.92  # Similarity ratio threshold

# =============================================================================
# FUNCTIONS
# =============================================================================


def assert_module_imported(
    imported: bool,
    module_name: str,
    message: str = "",
    retry: bool = True,
    alias: Union[str, None] = None,
    package: Union[str, None] = None,
):
    """Assert that the specified module is imported.

    Note
    ----
    This function is not intended to by explicitly executed by the user.

    Parameters
    ----------
    imported : bool
        Boolean indicating whether the module is imported.
    module_name : str
        Name of the module to check.
    message : str, optional. Default: ""
        Error message to display if the module is not imported.
    retry : bool, optional. Default: True
        Whether to retry the import if the module is not imported.
    alias : str, optional. Default: None
        Alias for the module.
    package : str, optional. Default: None
        Package to import the module from.

    Returns
    -------
    bool
        Whether the module is imported.
    """
    if alias is None:
        alias = module_name
    if not imported:
        if retry:
            try:
                importlib.import_module(module_name, package)
                return True
            except ImportError:
                pass
        raise ImportError(f"{alias} is required for this function. {message}")

    return True  # Module is imported


def parse_to_iter(value: Any, to: type = list) -> Iterable:
    """Parse a value to an iterable if it is not already.

    Note
    ----
    This function is not intended to by explicitly executed by the user.

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

    Note
    ----
    This function is not intended to by explicitly executed by the user.

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


def _similar(a: str, b: str) -> float:
    """Calculate the similarity ratio between two strings."""
    return SequenceMatcher(None, str(a), b).ratio()


def _n_close(a: Any, b: str, length: int, n=0) -> bool:
    """Check if two strings are n spaces-close."""
    stra = str(a)  # Convert to string

    return (stra[:length] == str(b)) and (
        (len(stra) == length + n) or stra[length] == " "
    )


def find_best_match(
    raw_series: Series,
    name: str,
    parse: Union[bool, None] = True,
    force: bool = False,
) -> Tuple[Index, Series, float]:
    """Find the best match for a name in a series.

    Note
    ----
    This function is not intended to by explicitly executed by the user.

    Parameters
    ----------
    raw_series : pd.Series
        Series to search in.
    name : str
        Name to search for.
    parse : bool, optional. Default: True.
        Whether to parse the raw_series and name.
        If `False`, the raw_series and name are not parsed.
        If `None`, the raw_series is not parsed, but the name is.
    force : bool, optional. Default: False.
        Whether to force the removal of trailing letters.

    Returns
    -------
    index : pd.Index
        Index of the best match.
    values : pd.Series
        Values of the best match.
    ratio : float
        Similarity ratio.
    """
    # Edit (clean) the series
    edited_series = raw_series.copy()  # Copy the series

    # Parse the series?
    if parse is not None and parse:
        edited_series = edited_series.astype(str).apply(
            parse_name, force=force
        )

    # Edit (clean) the name
    original_name = str(name)

    # Parse the name?
    if parse is None or parse:
        name = parse_name(name, force)

    # Check for exact matches
    exact_matches = edited_series[edited_series == name]
    if not exact_matches.empty:
        # exact_matches = raw_series.loc[exact_matches.index]
        exact_matches = raw_series[edited_series == name]
        if exact_matches.values[0] == original_name:
            return exact_matches.index, exact_matches.values, 1.0

        return exact_matches.index, exact_matches.values, 0.99999  # Almost 1

    # If no exact matches, search for 1 space-close names
    length = len(name)
    close_matches = edited_series.apply(lambda x: _n_close(x, name, length, 1))

    if close_matches.any():  # If 1 space-close names found
        return raw_series[close_matches].index, raw_series[close_matches], 0.9

    # If no 1 space-close names, search for 2 space-close names
    close_matches = edited_series.apply(lambda x: _n_close(x, name, length, 2))

    if close_matches.any():  # If 2 space-close names found
        return raw_series[close_matches].index, raw_series[close_matches], 0.8

    # If no 2 space-close names, search for similar names
    similarity_ratios = edited_series.apply(lambda x: _similar(x, name))
    good_matches = similarity_ratios >= RATIOS_THRESHOLD

    if not good_matches.any():  # No similar names found
        top_3_indices = similarity_ratios.nlargest(3).index
        good_matches = similarity_ratios.index.isin(top_3_indices)

    # Get the good matches
    similarity_ratios = similarity_ratios[good_matches]

    # Return the index, values, and the minimum similarity ratio
    return (
        similarity_ratios.index,
        raw_series[good_matches],
        similarity_ratios.values.min(),
    )
