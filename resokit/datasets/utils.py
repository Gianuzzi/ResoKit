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

"""Module with internal utility functions for the datasets package."""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from types import MappingProxyType
from typing import Any, Callable, List, Set, Tuple, Union
from zipfile import ZIP_DEFLATED, ZipFile

from bs4 import BeautifulSoup

from pandas import DataFrame, concat, merge, read_csv

import requests

from resokit.utils.parser import parse_to_iter


# =============================================================================
# CONSTANTS
# =============================================================================

# Path to the datasets directory
DATASETS_DIR = Path(os.path.expanduser(os.path.join("~", ".resokit_data")))

# -------------------------- EU and NASA DATASETS -----------------------------

# Name for the ZIP archive
DATASET_ZIPNAMES = {"eu": "exoplanet_eu.zip", "nasa": "nasa_exoplanets.zip"}

# Filenames and URLs for the datasets
DATASET_FILENAMES = {"eu": "exoplanet_eu.csv", "nasa": "nasa.csv"}
DATASET_URLS = {
    "eu": "https://exoplanet.eu/catalog/csv/",
    "nasa": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    + "query=select+*+from+ps&format=csv",
}

# Index columns for each dataset
INDEX_COLUMNS = {"eu": ["name", "star_name"], "nasa": ["pl_name", "hostname"]}

# EU dtypes
EU_MAPPING = {
    "name": "object",
    "planet_status": "object",
    "mass": "float64",
    "mass_error_min": "float64",
    "mass_error_max": "float64",
    "mass_sini": "float64",
    "mass_sini_error_min": "float64",
    "mass_sini_error_max": "float64",
    "radius": "float64",
    "radius_error_min": "float64",
    "radius_error_max": "float64",
    "orbital_period": "float64",
    "orbital_period_error_min": "float64",
    "orbital_period_error_max": "float64",
    "semi_major_axis": "float64",
    "semi_major_axis_error_min": "float64",
    "semi_major_axis_error_max": "float64",
    "eccentricity": "float64",
    "eccentricity_error_min": "float64",
    "eccentricity_error_max": "float64",
    "inclination": "float64",
    "inclination_error_min": "float64",
    "inclination_error_max": "float64",
    "angular_distance": "float64",
    "discovered": "float64",
    "updated": "object",
    "omega": "float64",
    "omega_error_min": "float64",
    "omega_error_max": "float64",
    "tperi": "float64",
    "tperi_error_min": "float64",
    "tperi_error_max": "float64",
    "tconj": "float64",
    "tconj_error_min": "float64",
    "tconj_error_max": "float64",
    "tzero_tr": "float64",
    "tzero_tr_error_min": "float64",
    "tzero_tr_error_max": "float64",
    "tzero_tr_sec": "float64",
    "tzero_tr_sec_error_min": "float64",
    "tzero_tr_sec_error_max": "float64",
    "lambda_angle": "float64",
    "lambda_angle_error_min": "float64",
    "lambda_angle_error_max": "float64",
    "impact_parameter": "float64",
    "impact_parameter_error_min": "float64",
    "impact_parameter_error_max": "float64",
    "tzero_vr": "float64",
    "tzero_vr_error_min": "float64",
    "tzero_vr_error_max": "float64",
    "k": "float64",
    "k_error_min": "float64",
    "k_error_max": "float64",
    "temp_calculated": "float64",
    "temp_calculated_error_min": "float64",
    "temp_calculated_error_max": "float64",
    "temp_measured": "float64",
    "hot_point_lon": "float64",
    "geometric_albedo": "float64",
    "geometric_albedo_error_min": "float64",
    "geometric_albedo_error_max": "float64",
    "log_g": "float64",
    "publication": "object",
    "detection_type": "object",
    "mass_measurement_type": "object",
    "radius_measurement_type": "object",
    "alternate_names": "object",
    "molecules": "object",
    "star_name": "object",
    "ra": "float64",
    "dec": "float64",
    "mag_v": "float64",
    "mag_i": "float64",
    "mag_j": "float64",
    "mag_h": "float64",
    "mag_k": "float64",
    "star_distance": "float64",
    "star_distance_error_min": "float64",
    "star_distance_error_max": "float64",
    "star_metallicity": "float64",
    "star_metallicity_error_min": "float64",
    "star_metallicity_error_max": "float64",
    "star_mass": "float64",
    "star_mass_error_min": "float64",
    "star_mass_error_max": "float64",
    "star_radius": "float64",
    "star_radius_error_min": "float64",
    "star_radius_error_max": "float64",
    "star_sp_type": "object",
    "star_age": "float64",
    "star_age_error_min": "float64",
    "star_age_error_max": "float64",
    "star_teff": "float64",
    "star_teff_error_min": "float64",
    "star_teff_error_max": "float64",
    "star_detected_disc": "object",
    "star_magnetic_field": "object",
    "star_alternate_names": "object",
}

# Nasa dtypes
NASA_MAPPING = {
    "pl_name": "object",
    "pl_letter": "object",
    "hostname": "object",
    "hd_name": "object",
    "hip_name": "object",
    "tic_id": "object",
    "gaia_id": "object",
    "default_flag": "int64",
    "pl_refname": "object",
    "sy_refname": "object",
    "disc_pubdate": "object",
    "disc_year": "int64",
    "discoverymethod": "object",
    "disc_locale": "object",
    "disc_facility": "object",
    "disc_instrument": "object",
    "disc_telescope": "object",
    "disc_refname": "object",
    "ra": "float64",
    "rastr": "object",
    "dec": "float64",
    "decstr": "object",
    "glon": "float64",
    "glat": "float64",
    "elon": "float64",
    "elat": "float64",
    "pl_orbper": "float64",
    "pl_orbpererr1": "float64",
    "pl_orbpererr2": "float64",
    "pl_orbperlim": "float64",
    "pl_orbperstr": "object",
    "pl_orblpererr1": "float64",
    "pl_orblper": "float64",
    "pl_orblpererr2": "float64",
    "pl_orblperlim": "float64",
    "pl_orblperstr": "object",
    "pl_orbsmax": "float64",
    "pl_orbsmaxerr1": "float64",
    "pl_orbsmaxerr2": "float64",
    "pl_orbsmaxlim": "float64",
    "pl_orbsmaxstr": "object",
    "pl_orbincl": "float64",
    "pl_orbinclerr1": "float64",
    "pl_orbinclerr2": "float64",
    "pl_orbincllim": "float64",
    "pl_orbinclstr": "object",
    "pl_orbtper": "float64",
    "pl_orbtpererr1": "float64",
    "pl_orbtpererr2": "float64",
    "pl_orbtperlim": "float64",
    "pl_orbtperstr": "object",
    "pl_orbeccen": "float64",
    "pl_orbeccenerr1": "float64",
    "pl_orbeccenerr2": "float64",
    "pl_orbeccenlim": "float64",
    "pl_orbeccenstr": "object",
    "pl_eqt": "float64",
    "pl_eqterr1": "float64",
    "pl_eqterr2": "float64",
    "pl_eqtlim": "float64",
    "pl_eqtstr": "object",
    "pl_occdep": "float64",
    "pl_occdeperr1": "float64",
    "pl_occdeperr2": "float64",
    "pl_occdeplim": "float64",
    "pl_occdepstr": "object",
    "pl_insol": "float64",
    "pl_insolerr1": "float64",
    "pl_insolerr2": "float64",
    "pl_insollim": "float64",
    "pl_insolstr": "object",
    "pl_dens": "float64",
    "pl_denserr1": "float64",
    "pl_denserr2": "float64",
    "pl_denslim": "float64",
    "pl_densstr": "object",
    "pl_trandep": "float64",
    "pl_trandeperr1": "float64",
    "pl_trandeperr2": "float64",
    "pl_trandeplim": "float64",
    "pl_trandepstr": "object",
    "pl_tranmid": "float64",
    "pl_tranmiderr1": "float64",
    "pl_tranmiderr2": "float64",
    "pl_tranmidlim": "float64",
    "pl_tranmidstr": "object",
    "pl_trandur": "float64",
    "pl_trandurerr1": "float64",
    "pl_trandurerr2": "float64",
    "pl_trandurlim": "float64",
    "pl_trandurstr": "object",
    "sy_kmagstr": "object",
    "sy_umag": "float64",
    "sy_umagerr1": "float64",
    "sy_umagerr2": "float64",
    "sy_umagstr": "object",
    "sy_rmag": "float64",
    "sy_rmagerr1": "float64",
    "sy_rmagerr2": "float64",
    "sy_rmagstr": "object",
    "sy_imag": "float64",
    "sy_imagerr1": "float64",
    "sy_imagerr2": "float64",
    "sy_imagstr": "object",
    "sy_zmag": "float64",
    "sy_zmagerr1": "float64",
    "sy_zmagerr2": "float64",
    "sy_zmagstr": "object",
    "sy_w1mag": "float64",
    "sy_w1magerr1": "float64",
    "sy_w1magerr2": "float64",
    "sy_w1magstr": "object",
    "sy_w2mag": "float64",
    "sy_w2magerr1": "float64",
    "sy_w2magerr2": "float64",
    "sy_w2magstr": "object",
    "sy_w3mag": "float64",
    "sy_w3magerr1": "float64",
    "sy_w3magerr2": "float64",
    "sy_w3magstr": "object",
    "sy_w4mag": "float64",
    "sy_w4magerr1": "float64",
    "sy_w4magerr2": "float64",
    "sy_w4magstr": "object",
    "sy_gmag": "float64",
    "sy_gmagerr1": "float64",
    "sy_gmagerr2": "float64",
    "sy_gmagstr": "object",
    "sy_gaiamag": "float64",
    "sy_gaiamagerr1": "float64",
    "sy_gaiamagerr2": "float64",
    "sy_gaiamagstr": "object",
    "sy_tmag": "float64",
    "sy_tmagerr1": "float64",
    "sy_tmagerr2": "float64",
    "sy_tmagstr": "object",
    "pl_controv_flag": "int64",
    "pl_tsystemref": "object",
    "st_metratio": "object",
    "st_spectype": "object",
    "sy_kepmag": "float64",
    "sy_kepmagerr1": "float64",
    "sy_kepmagerr2": "float64",
    "sy_kepmagstr": "float64",
    "st_rotp": "float64",
    "st_rotperr1": "float64",
    "st_rotperr2": "float64",
    "st_rotplim": "float64",
    "st_rotpstr": "object",
    "pl_projobliq": "float64",
    "pl_projobliqerr1": "float64",
    "pl_projobliqerr2": "float64",
    "pl_projobliqlim": "float64",
    "pl_projobliqstr": "object",
    "x": "float64",
    "y": "float64",
    "z": "float64",
    "htm20": "int64",
    "pl_rvamp": "float64",
    "pl_rvamperr1": "float64",
    "pl_rvamperr2": "float64",
    "pl_rvamplim": "float64",
    "pl_rvampstr": "object",
    "pl_radj": "float64",
    "pl_radjerr1": "float64",
    "pl_radjerr2": "float64",
    "pl_radjlim": "float64",
    "pl_radjstr": "object",
    "pl_rade": "float64",
    "pl_radeerr1": "float64",
    "pl_radeerr2": "float64",
    "pl_radelim": "float64",
    "pl_radestr": "object",
    "pl_ratror": "float64",
    "pl_ratrorerr1": "float64",
    "pl_ratrorerr2": "float64",
    "pl_ratrorlim": "float64",
    "pl_ratrorstr": "object",
    "pl_ratdor": "float64",
    "pl_trueobliq": "float64",
    "pl_trueobliqerr1": "float64",
    "pl_trueobliqerr2": "float64",
    "pl_trueobliqlim": "float64",
    "pl_trueobliqstr": "object",
    "sy_icmag": "float64",
    "sy_icmagerr1": "float64",
    "sy_icmagerr2": "float64",
    "sy_icmagstr": "object",
    "rowupdate": "object",
    "pl_pubdate": "object",
    "st_refname": "object",
    "releasedate": "object",
    "dkin_flag": "int64",
    "pl_ratdorerr1": "float64",
    "pl_ratdorerr2": "float64",
    "pl_ratdorlim": "float64",
    "pl_ratdorstr": "object",
    "pl_imppar": "float64",
    "pl_impparerr1": "float64",
    "pl_impparerr2": "float64",
    "pl_impparlim": "float64",
    "pl_impparstr": "object",
    "pl_cmassj": "float64",
    "pl_cmassjerr1": "float64",
    "pl_cmassjerr2": "float64",
    "pl_cmassjlim": "float64",
    "pl_cmassjstr": "object",
    "pl_cmasse": "float64",
    "pl_cmasseerr1": "float64",
    "pl_cmasseerr2": "float64",
    "pl_cmasselim": "float64",
    "pl_cmassestr": "object",
    "pl_massj": "float64",
    "pl_massjerr1": "float64",
    "pl_massjerr2": "float64",
    "pl_massjlim": "float64",
    "pl_massjstr": "object",
    "pl_masse": "float64",
    "pl_masseerr1": "float64",
    "pl_masseerr2": "float64",
    "pl_masselim": "float64",
    "pl_massestr": "object",
    "pl_bmassj": "float64",
    "pl_bmassjerr1": "float64",
    "pl_bmassjerr2": "float64",
    "pl_bmassjlim": "float64",
    "pl_bmassjstr": "object",
    "pl_bmasse": "float64",
    "pl_bmasseerr1": "float64",
    "pl_bmasseerr2": "float64",
    "pl_bmasselim": "float64",
    "pl_bmassestr": "object",
    "pl_bmassprov": "object",
    "pl_msinij": "float64",
    "pl_msinijerr1": "float64",
    "pl_msinijerr2": "float64",
    "pl_msinijlim": "float64",
    "pl_msinijstr": "object",
    "pl_msinie": "float64",
    "pl_msinieerr1": "float64",
    "pl_msinieerr2": "float64",
    "pl_msinielim": "float64",
    "pl_msiniestr": "object",
    "st_teff": "float64",
    "st_tefferr1": "float64",
    "st_tefferr2": "float64",
    "st_tefflim": "float64",
    "st_teffstr": "object",
    "st_met": "float64",
    "st_meterr1": "float64",
    "st_meterr2": "float64",
    "st_metlim": "float64",
    "st_metstr": "object",
    "st_radv": "float64",
    "st_radverr1": "float64",
    "st_radverr2": "float64",
    "st_radvlim": "float64",
    "st_radvstr": "object",
    "st_vsin": "float64",
    "st_vsinerr1": "float64",
    "st_vsinerr2": "float64",
    "st_vsinlim": "float64",
    "st_vsinstr": "object",
    "st_lum": "float64",
    "st_lumerr1": "float64",
    "st_lumerr2": "float64",
    "st_lumlim": "float64",
    "st_lumstr": "object",
    "st_logg": "float64",
    "st_loggerr1": "float64",
    "st_loggerr2": "float64",
    "st_logglim": "float64",
    "st_loggstr": "object",
    "st_age": "float64",
    "st_ageerr1": "float64",
    "st_ageerr2": "float64",
    "st_agelim": "float64",
    "st_agestr": "object",
    "st_mass": "float64",
    "st_masserr1": "float64",
    "st_masserr2": "float64",
    "st_masslim": "float64",
    "st_massstr": "object",
    "st_dens": "float64",
    "st_denserr1": "float64",
    "st_denserr2": "float64",
    "st_denslim": "float64",
    "st_densstr": "object",
    "st_rad": "float64",
    "st_raderr1": "float64",
    "st_raderr2": "float64",
    "st_radlim": "float64",
    "st_radstr": "object",
    "ttv_flag": "int64",
    "ptv_flag": "int64",
    "tran_flag": "int64",
    "rv_flag": "int64",
    "ast_flag": "int64",
    "obm_flag": "int64",
    "micro_flag": "int64",
    "etv_flag": "int64",
    "ima_flag": "int64",
    "pul_flag": "int64",
    "soltype": "object",
    "sy_snum": "int64",
    "sy_pnum": "int64",
    "sy_mnum": "int64",
    "cb_flag": "int64",
    "st_nphot": "int64",
    "st_nrvc": "int64",
    "st_nspec": "int64",
    "pl_nespec": "int64",
    "pl_ntranspec": "int64",
    "pl_ndispec": "int64",
    "pl_nnotes": "int64",
    "sy_pm": "float64",
    "sy_pmerr1": "float64",
    "sy_pmerr2": "float64",
    "sy_pmstr": "object",
    "sy_pmra": "float64",
    "sy_pmraerr1": "float64",
    "sy_pmraerr2": "float64",
    "sy_pmrastr": "object",
    "sy_pmdec": "float64",
    "sy_pmdecerr1": "float64",
    "sy_pmdecerr2": "float64",
    "sy_pmdecstr": "object",
    "sy_plx": "float64",
    "sy_plxerr1": "float64",
    "sy_plxerr2": "float64",
    "sy_plxstr": "object",
    "sy_dist": "float64",
    "sy_disterr1": "float64",
    "sy_disterr2": "float64",
    "sy_diststr": "object",
    "sy_bmag": "float64",
    "sy_bmagerr1": "float64",
    "sy_bmagerr2": "float64",
    "sy_bmagstr": "object",
    "sy_vmag": "float64",
    "sy_vmagerr1": "float64",
    "sy_vmagerr2": "float64",
    "sy_vmagstr": "object",
    "sy_jmag": "float64",
    "sy_jmagerr1": "float64",
    "sy_jmagerr2": "float64",
    "sy_jmagstr": "object",
    "sy_hmag": "float64",
    "sy_hmagerr1": "float64",
    "sy_hmagerr2": "float64",
    "sy_hmagstr": "object",
    "sy_kmag": "float64",
    "sy_kmagerr1": "float64",
    "sy_kmagerr2": "float64",
}

# Mapping of dataset names to their respective dtypes
DATASET_DTYPES = MappingProxyType({"eu": EU_MAPPING, "nasa": NASA_MAPPING})

# URL to query the dataset length
QUERY_LENGTH_URL = {
    "eu": "https://exoplanet.eu/home/",
    "nasa": "https://exoplanetarchive.ipac.caltech.edu/index.html",
}

# --------------------------- BINARY SYSTEMS DATASETS --------------------------

# No ZIP here, only txt

# Filenames and URLs for the binaries datasets
BINARIES_FILENAMES = {"p": "plan_circ.txt", "s": "plan_bin500aun.txt"}
BINARIES_URLS = {
    "p": "https://lesia.obspm.fr/perso/philippe-thebault/plan_circ.txt",
    "s": "https://lesia.obspm.fr/perso/philippe-thebault/plan_bin500aun.txt",
}

# Columns of the binaries datasets
BINARIES_COLUMNS = [
    "star0_name",
    "alternate_name",
    "star0_mass",
    "star1_mass",
    "dist",
    "disc_method",
    "a",
    "e",
    "nplanets",
    "planet_a",
    "planet_e",
    "planet_mass",
    "planet_HW_crit",
    "imut",
]

# Default key columns to use to match old and new rows
DEFAULT_KEY_COLS = {
    "nasa": [
        "pl_name",
        "pl_letter",
        "hostname",
        "default_flag",
        "pl_controv_flag",
        "releasedate",
        "pl_refname",
    ],
    "eu": [
        "name",
        "updated",
        "star_name",
    ],
}

# =============================================================================
# FUNCTIONS
# =============================================================================


def resolve_paths(
    to_file: Union[bool, str, Path],
    to_zip: Union[bool, str, Path],
    dir_path: Union[bool, str, Path, None],
    default_file: str,
    default_zip: str,
    default_dir: Path,
) -> Tuple[Set[Path], Set[Path], Set[Path]]:
    """
    Normalize to_file, to_zip, and dir_path into full output file and zip paths.

    Note
    ----
    This function is not intended to by explicitly executed by the user.

    Parameters
    ----------
        to_file: File name or path (True = use default file name in dir_path).
        to_zip: Zip name or path (True = use default zip name in dir_path).
        dir_path: Base directory (True = use default_dir).
        default_file: Default file name if to_file is True.
        default_zip: Default zip name if to_zip is True.
        default_dir: Default directory if dir_path is True.

    Returns
    -------
        Tuple containing:
            - Set of base directories (Path): physical directories containing
               files/zips
            - Set of resolved file paths (Path)
            - Set of resolved zip output paths (Path to file inside zip)
    """
    if to_file is True and to_zip is True and dir_path is True:
        return (
            set([default_dir]),
            set(),
            set([default_dir / default_zip / default_file]),
        )

    # Short-circuit: if to_file is explicitly False, return all empty
    if to_file is False:
        return set(), set(), set()

    def parse_path_input(value, default_name) -> Tuple[Path, Path]:
        if value is True:
            return None, Path(default_name)
        elif isinstance(value, (str, Path)):
            value = Path(value)
            if value.is_absolute() or value.parent != Path("."):
                return value.parent.resolve(), value.name
            else:
                return None, value.name
        elif value in [False, None]:
            return None, None
        else:
            raise ValueError(f"Invalid path input: {value}")

    # Parse inputs
    file_dir, file_name = parse_path_input(to_file, default_file)
    zip_dir, zip_name = parse_path_input(to_zip, default_zip)

    # Resolve base directory
    if dir_path is True:
        base_dir = default_dir.resolve()
    elif isinstance(dir_path, (str, Path)):
        base_dir = Path(dir_path).resolve()
    else:
        base_dir = None

    # If zip_dir is not provided and dir_path is, assume zip is in base_dir
    if zip_dir is None and zip_name is not None and base_dir is not None:
        zip_dir = base_dir

    fpaths = set()
    zfpaths = set()
    base_paths = set()

    # Resolve file path
    if file_name:
        # If user gave no file_dir and zip_dir == base_dir,
        # skip adding file outside zip
        if file_dir is None and zip_dir == base_dir and base_dir is not None:
            pass  # Do not add to fpaths
        elif file_dir is None and base_dir is None and zip_name is not None:
            pass
        elif file_dir and (file_dir.name.endswith(".zip")):
            full_zfile_path = file_dir / file_name
            zfpaths.add(full_zfile_path)
        else:
            if file_dir:
                full_file_path = file_dir / file_name
            elif base_dir and not (zip_dir == base_dir):
                full_file_path = base_dir / file_name
            else:
                full_file_path = Path(file_name)
            fpaths.add(full_file_path)
            base_paths.add(full_file_path.parent)

    # Resolve zip file path (and zip-internal file path)
    if zip_name:
        if zip_dir:
            zip_file_path = zip_dir / zip_name
        elif base_dir:
            zip_file_path = base_dir / zip_name
        else:
            zip_file_path = Path(zip_name)
        if file_name:
            full_zip_entry = zip_file_path / file_name
        else:
            full_zip_entry = zip_file_path
        zfpaths.add(full_zip_entry)
        base_paths.add(
            zip_file_path.parent
        )  # <-- Only the containing folder, not the zip itself

    return base_paths, fpaths, zfpaths


def check_file_age(
    file_path: Union[str, Path],
    zip_path: Union[str, Path, None],
    verbose: bool = True,
) -> int:
    """Check the dataset file's age in days.

    Note
    ----
    This function is not intended to by explicitly executed by the user.

    Parameters
    ----------
    file_path : str or Path, optional. Default: False.
        Path to the file.
    zip_path : str or Path or None, optional. Default: False.
        Path to the ZIP archive.
    verbose : bool, optional. Default: True.
        If `True`, prints messages about the process.

    Returns
    -------
    age : int
        Age of the file in days.
    """
    file_path = Path(file_path)  # Convert to Path object
    # Get the file's last modified date
    if zip_path:
        file_name = Path(file_path).name
        with ZipFile(zip_path, "r") as zipf:  # Open the ZIP archive
            # Check if the file is in the ZIP archive
            if file_name not in zipf.namelist():
                raise FileNotFoundError(
                    f"File {file_name} not found in {zip_path}."
                )
            # Get the file's last modified date
            date_info = zipf.getinfo(file_name).date_time
            creation = datetime(*date_info)
    else:
        creation = datetime.fromtimestamp(file_path.stat().st_mtime)

    # Calculate age in days
    age = (datetime.now() - creation).days

    if verbose:
        print(f" Last modified: {creation} ({age} days ago).")

    return age


def load_from_zip(
    zip_path: Path,
    file_name: str,
    source: Union[str, None] = None,
    skip_rows: Union[int, Callable, None] = None,
    usecols: Union[list, Callable, None] = None,
    verbose: bool = True,
    custom_load: Union[Callable, None] = None,
) -> Union[DataFrame, Any]:
    """Load the dataset from a ZIP archive.

    Reads the dataset from a ZIP archive and returns it as a pandas DataFrame.

    Note
    ----
    This function is not intended to by explicitly executed by the user.

    Parameters
    ----------
    zip_path : Path
        Full path to the ZIP archive.
    file_name : str
        Name of the file to load from the ZIP archive.
    source : str, optional
        Identifier for the data source ('eu' or 'nasa').
        Used to define the dtypes of the dataset.
        If not 'eu' or 'nasa', no dtypes are defined.
    skip_rows : int, optional
        Number of rows to skip.
    usecols : list, optional
        Columns to load.
    verbose : bool, optional
        If `True`, prints messages about the process.
    custom_load : callable, optional
        Custom function to load the data from the ZIP archive.
        If provided, it is used instead of the default read_csv.
        It neglects the skip_rows, usecols, and dtypes parameters.

    Returns
    -------
    Union[pd.DataFrame, any]
        data : pd.DataFrame
            Loaded dataset as a pandas DataFrame.
        data : any
            Loaded data from the ZIP archive, if custom_load is provided.
    """
    # Check if the zip exists
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP archive {zip_path} not found.")

    # Load the dataset from the ZIP archive
    with ZipFile(zip_path, "r") as zipf:  # Open the ZIP archive
        if verbose:  # Print message if verbose
            print(f"  Reading {file_name} " + f"directly from {zip_path}...")
        # Load directly from the .zip
        dtypes = None if source is None else DATASET_DTYPES.get(source, None)
        skip_rows = 0 if skip_rows is None else skip_rows
        with zipf.open(file_name) as file:
            if custom_load is not None:
                return custom_load(file)
            data = read_csv(
                file,
                header=0,
                skiprows=skip_rows,
                usecols=usecols,
                dtype=dtypes,
            )

    return data


def remove_from_zip(zipfname: str, *filenames: str, verbose: bool = False):
    """Remove files from a zip archive.

    This function removes files from a zip archive without extracting it.
    It is unefficient (especially for large archives) because it decompresses
    and recompresses the whole archive.

    Note
    ----
    This function is not intended to by explicitly executed by the user.

    Parameters
    ----------
    zipfname : str
        Path to the zip archive.
    filenames : str
        Names of the files to remove from the archive.
    verbose : bool, optional
        If True, print messages about the process.

    Returns
    -------
    None
    """
    # Check if any of the files to remove is in the archive
    has_files = False
    with ZipFile(zipfname, "r") as zipread:
        for filename in filenames:
            if filename in zipread.namelist():
                has_files = True
                break
    if not has_files:
        return

    # Create a temporary directory
    tempdir = mkdtemp()
    try:
        # Create a new zip archive
        tempname = os.path.join(tempdir, "new.zip")
        # Read the original archive
        with ZipFile(zipfname, "r") as zipread:
            # Write the new archive
            with ZipFile(tempname, "w", compression=ZIP_DEFLATED) as zipwrite:
                # Copy all files except the ones to remove
                for item in zipread.infolist():
                    if item.filename not in filenames:
                        data = zipread.read(item.filename)
                        zipwrite.writestr(item, data)
        # Replace the original archive with the new one
        shutil.move(tempname, zipfname)
        if verbose:
            print(f"Removed files: {', '.join(filenames)} from {zipfname}")
    finally:
        # Remove the temporary directory
        shutil.rmtree(tempdir)


def _get_request_with_refresh(url: str, verbose: bool = True, **kwargs):
    """Request a URL and automatically follows HTML meta refresh redirects."""
    response = requests.get(url, **kwargs)

    # Regex for meta refresh
    meta_match = re.search(
        r'<meta\s+[^>]*http-equiv=["\']?Refresh["\']?[^>]'
        r'*content=["\']?\s*\d+\s*;\s*URL=([^"\'>\s]+)',
        response.text,
        flags=re.IGNORECASE,
    )

    if meta_match is not None:
        redirect_url = meta_match.group(1).strip()
        # If the redirect is relative, build absolute
        if not redirect_url.lower().startswith("http"):
            redirect_url = requests.compat.urljoin(response.url, redirect_url)

        if verbose:
            print(f"Following meta refresh to: {redirect_url}")
        return requests.get(redirect_url, **kwargs)

    return response


def request_dataset(
    url: str,
    verbose: bool = True,
    chunk_size: int = 1024,
    print_size: float = 0.15,
) -> bytes:
    """Download the data from a specified URL.

    Note
    ----
    This function is not intended to by explicitly executed by the user.

    Parameters
    ----------
    url : str
        URL to download the data from.
    verbose : bool, optional
        If True, print messages about the download process.
    chunk_size : int, optional
        Size of the chunks (in bytes) to download the data.
    print_size : float, optional. Default is 0.15
        Update frequency (in MB, or KB if <=0.001) for the
        download progress bar.
        Used only if verbose is True. Useful for large downloads,
        to avoid IO overhead, especially in Jupyter notebooks.

    Returns
    -------
    content : bytes
        The downloaded data.
    """
    # Print message
    if verbose:
        print(f"Downloading data from {url}...")

    # Check if the print size is in MB or KB
    print_unit = "MB"
    bytes_unit = 1e6
    if print_size <= 0.001:
        print_size *= 1e3
        print_unit = "KB"
        bytes_unit = 1e3

    # Check if Jupyter notebook is running
    if is_notebook() and verbose:
        print(f" Note: Progress is shown at every {print_size} {print_unit}")

    # Send a GET request with streaming enabled
    response = _get_request_with_refresh(url, verbose=verbose, stream=True)

    # Check for errors
    response.raise_for_status()

    # Initialize the downloaded data as a byte array
    downloaded_data = bytearray()

    # Initialize the number of downloaded bytes
    downloaded_size = 0

    # Iterate over the response content
    increment = 0.0
    for chunk in response.iter_content(chunk_size=chunk_size):
        if chunk:  # Filter out keep-alive new chunks
            downloaded_data.extend(chunk)  # Append the chunk to the data
            downloaded_size += len(
                chunk
            )  # Update the downloaded size (in bytes)

            # Print the download progress (in MB or KB)
            if verbose and downloaded_size >= bytes_unit * increment:
                print(
                    f" Downloaded: {downloaded_size / bytes_unit:.3f} "
                    + f"{print_unit}",
                    end="\r",
                )
                increment += print_size
    # Check printed 0.0 MB
    if verbose and downloaded_size <= 1e3 and print_unit == "MB":
        print(
            f" Downloaded: {downloaded_size / 1e3:.3f} KB",
            end="\r",
        )

    return bytes(
        downloaded_data
    )  # Return the downloaded data as inmutable bytes


def is_notebook() -> bool:
    """Check if the code is running in a Jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def __parse_date(date_str: str, soft: bool = True) -> Union[datetime, bool]:
    """Try different date formats to robustly parse the last update date.

    Parameters
    ----------
    date_str : str
        The date string to parse.
        soft : bool, optional. Default: True.
        If True, return False if parsing fails instead of raising an error.

    Returns
    -------
    Union[datetime, bool]
        Parsed date as a datetime object if successful,
        or False if parsing fails and soft is True.
    """
    # Full & abbreviated months
    date_formats = ["%B %d, %Y", "%b %d, %Y", "%b. %d, %Y", "%m/%d/%Y"]

    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue  # Try the next format

    # If all fail
    if soft:
        return False
    raise ValueError(f"Could not parse date: {date_str}")


def check_outdated_dataset(
    source: str, verbose: bool = True
) -> Tuple[int, Union[int, None]]:
    """Web scrap the length (count) of the dataset from the specified source.

    Note
    ----
    This function is not intended to by explicitly executed by the user.

    Parameters
    ----------
    source : str
        Data source identifier ('eu' or 'nasa').
    verbose : bool, optional. Default: True.
        Print query information.

    Returns
    -------
    length : int
        Amount of entries (rows) in the dataset.
    last_update : datetime
        Date of the last update of the dataset.
    If no match is found, -1 is returned for length and None for last_update.
    """
    source = source.lower()  # Ensure lowercase

    # Message
    if verbose:
        print(f"Checking online dataset from {source}...")

    # Call subfunction
    if source == "eu":
        length_str, date_str = _check_outdated_eu(verbose=verbose)
    elif source == "nasa":
        length_str, date_str = _check_outdated_nasa(verbose=verbose)
    else:
        raise ValueError("Invalid source. Must be 'eu' or 'nasa'.")

    # Parse the planets
    try:
        # Remove comma if present
        length = int(str(length_str).replace(",", "").replace(".", ""))
    except ValueError:
        if verbose:
            print(
                " Could not parse the amount of planets in "
                + f"online {source} database."
            )
        length = -1

    # Parse the Date
    date = __parse_date(str(date_str), soft=True)
    if date is False:
        if verbose:
            print(" Could not parse the last update date.")
        date = None
        last_update = None
    else:
        last_update = (datetime.now() - date).days

    # Message
    if verbose:
        if length > 0:
            print(f" Number of planets in online dataset: {length}")
        if date:
            print(f" Last online update: {date} ({last_update} days ago)")

    return length, last_update


def _check_outdated_eu(
    verbose: bool = True,
) -> Tuple[str, Union[str, None]]:
    """Web scrap the length of the exoplanet.eu dataset.

    Parameters
    ----------
    verbose : bool, optional. Default: True.
        Print query information.

    Returns
    -------
    length : str
        Amount of entries (rows) in the dataset.
        If no match is found, -1 is returned.
    last_update : str
        Date of the last update of the dataset.
        If no match is found, None is returned.
    """
    try:
        # Fetch the webpage content
        response = requests.get(QUERY_LENGTH_URL["eu"], timeout=10)
        response.raise_for_status()  # Raise an error for HTTP issues
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the text containing "Last update"
        text = None
        for p in soup.find_all("p"):
            if "Last update" in p.text:
                text = p.text.strip()
                break

        if text is None:
            if verbose:
                print(" No 'Last update' text found on the webpage.")
            return "-1", None

        # Extract the date and number of planets using regex
        aux = (
            r"Last\s+update[d]?\s*:\s*([\w\.]+\s+\d{1,2},\s+\d{4})"
            + r"\s+currently\s+([\d,]+)\s+planets"
        )
        match = re.search(aux, text)

        if not match:
            if verbose:
                print(" No match found in the extracted text.")
            return "-1", None

        # Parse extracted values
        last_update, length = match.groups()

        return str(length), last_update

    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"Error fetching the webpage: {e}")

    return "-1", None


def _check_outdated_nasa(
    verbose: bool = True,
) -> Tuple[Union[str, int], Union[str, None]]:
    """Web scrap the length of the NASA dataset.

    Parameters
    ----------
    verbose : bool, optional. Default: True.
        Print query information.

    Returns
    -------
    length : str
        Amount of entries (rows) in the dataset.
        If no match is found, -1 is returned.
    last_update : str
        Date of the last update of the dataset.
        If no match is found, None is returned.
    """
    try:
        # Fetch the webpage content
        response = requests.get(QUERY_LENGTH_URL["nasa"])
        response.raise_for_status()  # Raise an error for HTTP issues
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the number of confirmed planets
        planet_count_div = soup.find("div", class_="stat")
        if planet_count_div:
            length = (
                planet_count_div.text.strip().replace(",", "").replace(".", "")
            )
        else:
            if verbose:
                print(" No match found in the extracted text.")
            length = -1

        # Extract the last update date
        date_div = soup.find("div", class_="date")
        if date_div:
            last_update = date_div.text.strip()
        else:
            if verbose:
                print(" No match found in the extracted text.")
            last_update = None

        return str(length), last_update

    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"Error fetching the webpage: {e}")

    return "-1", None


def check_outdated_binary(source: str, verbose: bool = True) -> int:
    """Web scrap the length (count) of the file from the specified source.

    This function is kind of dumb, because it downloads the file (although
    if does not parse it, just count the lines).

    Note
    ----
    This function is not intended to by explicitly executed by the user.

    Parameters
    ----------
    source : str
        Data source identifier ('p' or 's').
    verbose : bool, optional. Default: True.
        Print query information.

    Returns
    -------
    length : int
        Amount of entries (rows) in the file.
        If no match is found, -1 is returned.
    """
    source = source.lower()  # Ensure lowercase
    if source not in BINARIES_URLS:
        raise ValueError(f"Invalid source: {source}. Must be 'p' or 's'.")

    # Message
    if verbose:
        print(f"Checking online dataset from {source}-type binaries...")

    # Call subfunction
    url = BINARIES_URLS[source]
    length = 0
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        for _ in response.iter_lines():
            length += 1
    except requests.RequestException as e:
        if verbose:
            print(f"Error: {e}")
            print(
                " Could not parse the amount of lines in "
                + f"online {source}-type binaries file."
            )
        length = -1

    # Message
    if verbose:
        if length > 0:
            print(
                " Number of lines (including header) "
                + f"in online dataset: {length}"
            )

    return length


def merge_old_and_new(
    old_df: DataFrame,
    new_df: DataFrame,
    source: Union[str, None] = None,
    verbose: bool = True,
    key_cols: Union[str, List, None] = None,
) -> DataFrame:
    """
    Merge old and new DataFrames based on a set of key columns.

    This function compares two datasets (`old_df` and `new_df`)
    and performs an outer merge using
    specified key columns. It returns a DataFrame that includes:
    - Rows only in the old data
    - Rows in both old and new (with new values preferred)
    - Rows only in the new data

    Note
    ----
    This function is not intended to by explicitly executed by the user.

    Parameters
    ----------
    old_df : pd.DataFrame
        The original (historical) dataset.

    new_df : pd.DataFrame
        The newly downloaded or updated dataset to be merged.

    source : str or None, optional
        If `key_cols` is not provided, this value will be used to
        look up default key columns
        from `DEFAULT_KEY_COLS[source]`.

    verbose : bool, default=True
        If True, prints the number of rows in each merge category.

    key_cols : str, list, or None, optional
        Column name(s) used to join the old and new data. If None,
        `source` must be provided.

    Returns
    -------
    pd.DataFrame
        A merged DataFrame that contains all unique records from the
        old and new datasets, giving priority to values from the new
        dataset where overlap occurs.
    """
    assert isinstance(old_df, DataFrame), (
        "Expected old_df to be a pd.DataFrame, "
        + f"got {type(old_df)} instead."
    )
    assert isinstance(new_df, DataFrame), (
        "Expected old_df to be a pd.DataFrame, "
        + f"got {type(new_df)} instead."
    )

    # Check if empty
    if len(new_df) == 0:
        if verbose:
            print("No merge required.")
        return old_df

    # Get key cols
    if key_cols is None and source is None:
        raise ValueError("Can not be both 'source' and 'key_cols' None")
    elif key_cols is None:
        key_cols = DEFAULT_KEY_COLS[source]
    else:
        key_cols = parse_to_iter(key_cols)

    # Merge into 1
    merged = merge(
        old_df,
        new_df,
        on=key_cols,
        suffixes=("_old", "_new"),
        how="outer",
        indicator=True,
    )

    def clean(df, suffix="new", key_cols=None):
        """Return cleaned DataFrame with only one suffix + key cols."""
        cols = [c for c in df.columns if c.endswith(f"_{suffix}")]
        base = [c.replace(f"_{suffix}", "") for c in cols]

        result = df[cols].copy()
        result.columns = base

        # Reattach key columns from original (no suffix)
        if key_cols is not None:
            for key in key_cols:
                if key not in result.columns and key in df.columns:
                    result[key] = df[key]

        return result.reset_index(drop=True)

    # Get cleaned
    old_clean = clean(
        merged[merged["_merge"] == "left_only"],
        suffix="old",
        key_cols=key_cols,
    )
    upd_clean = clean(
        merged[merged["_merge"] == "both"], suffix="new", key_cols=key_cols
    )
    new_clean = clean(
        merged[merged["_merge"] == "right_only"],
        suffix="new",
        key_cols=key_cols,
    )

    # Message
    if verbose:
        nold = len(old_clean)
        nupd = len(upd_clean)
        nnew = len(new_clean)
        print(f" Rows in only old | both | only new: {nold} | {nupd} | {nnew}")

    # Concatenate
    latest = concat([old_clean, upd_clean, new_clean], ignore_index=True)

    # latest.drop_duplicates(inplace=True)
    latest = latest[old_df.columns]  # Reorder

    return latest
