# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2024, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Module with internal utility functions for the datasets module."""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import shutil
from tempfile import mkdtemp
from types import MappingProxyType
from zipfile import ZIP_DEFLATED, ZipFile

from resokit.utils.utils import assert_module_imported

try:
    import requests

    requests_imported = True
except ImportError:
    requests_imported = False

# =============================================================================
# CONSTANTS
# =============================================================================

# EU dtypes
_EU_MAPPING = {
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
_NASA_MAPPING = {
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
DATASET_DTYPES = MappingProxyType({"eu": _EU_MAPPING, "nasa": _NASA_MAPPING})


# =============================================================================
# FUNCTIONS
# =============================================================================


def remove_from_zip(zipfname: str, *filenames: str, verbose: bool = False):
    """Remove files from a zip archive.

    This function removes files from a zip archive without extracting it.
    It is unefficient (especially for large archives) because it decompresses
    and recompresses the whole archive.

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


def request_dataset(
    url: str,
    verbose: bool = True,
) -> bytes:
    """Download the data from a specified URL.

    Parameters
    ----------
    url : str
        URL to download the data from.
    verbose : bool, optional
        If True, print messages about the download process.

    Returns
    -------
    content : bytes
        The downloaded data.
    """
    # Check if requests is imported
    assert_module_imported(requests_imported, "requests")

    if verbose:
        print(f" Downloading data from {url}...")

    # Download the file
    response = requests.get(url=url)  # Download the file
    response.raise_for_status()  # Check for errors

    return response.content
