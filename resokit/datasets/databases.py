# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2024, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Module to manage provided exoplanet datasets from exoplanet.eu and NASA."""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import datetime
import pandas as pd
from pathlib import Path
from zipfile import ZipFile

from resokit.utils import __assert_module_imported

try:
    import requests

    requests_imported = True
except ImportError:
    requests_imported = False

# =============================================================================
# CONSTANTS
# =============================================================================

# Base directory path
BASE_PATH = Path(os.path.abspath(os.path.dirname(__file__)))

# Filenames and URLs for the datasets
DATASET_FILENAMES = {"eu": "exoplanet_eu.csv", "nasa": "nasa.csv"}
DATASET_URLS = {
    "eu": "https://exoplanet.eu/catalog/csv/",
    "nasa": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    + "query=select+*+from+ps&format=csv",
}
ZIP_FILENAME = "datasets.zip"  # Name for the ZIP archive

# In-memory storage for datasets and indexes
IN_MEMORY_DATASETS = {"eu": None, "nasa": None}
IN_MEMORY_INDEXES = {"eu": None, "nasa": None}

# Index columns for each dataset
INDEX_COLUMNS = {"eu": ["name", "star_name"], "nasa": ["pl_name", "hostname"]}

# =============================================================================
# FUNCTIONS
# =============================================================================


def download_dataset(
    source: str,
    overwrite: bool = False,
    verbose: bool = True,
    return_data: bool = False,
    store: bool = False,
):
    """
    Downloads the dataset from a specified source and saves it locally as CSV.

    Parameters
    ----------
    source : str
        Identifier for the data source ('eu' or 'nasa').
    overwrite : bool, optional
        If True, overwrites the existing file if it exists.
    verbose : bool, optional
        If True, displays messages about the download process.
    return_data : bool, optional
        If True, returns the downloaded dataset as a DataFrame.
    store : bool, optional
        If True, stores the dataset in memory.

    Returns
    -------
    Path or None
        Path to the downloaded dataset if successful, else None.
    """
    __assert_module_imported(requests_imported, "requests")
    source = source.lower()
    if source not in DATASET_FILENAMES:
        raise ValueError(f"Invalid source: {source}. Must be 'eu' or 'nasa'.")

    file_path = BASE_PATH / DATASET_FILENAMES[source]
    url = DATASET_URLS[source]
    zip_path = BASE_PATH / ZIP_FILENAME

    # Check if file exists or exists in ZIP
    if file_path.exists() and not overwrite:
        if verbose:
            print(
                f"{file_path} already exists. \n"
                + "Use overwrite=True to redownload and overwrite."
            )
        return
    elif (
        zip_path.exists()
        and DATASET_FILENAMES[source] in ZipFile(zip_path).namelist()
        and not overwrite
    ):
        if verbose:
            print(
                f"{DATASET_FILENAMES[source]} found in {ZIP_FILENAME}. \n"
                + "Use overwrite=True to redownload, or run load_dataset"
                + f"(source='{source}', extract=True) to extract it."
            )
        return

    # Download if not found
    if verbose:
        print(f"Downloading data from {url}...")
    response = requests.get(url=url)
    response.raise_for_status()
    if verbose:
        print("Writing data to file...")
    with open(file_path, "wb") as f:
        f.write(response.content)
    if verbose:
        print(f"File {file_path} successfully downloaded and saved.")
    if return_data:
        return load_dataset(source=source, verbose=verbose, store=store)

    return


def create_zip_archive(verbose: bool = True):
    """
    Create a ZIP archive containing both EU and NASA dataset CSV files.

    Parameters
    ----------
    verbose : bool, optional
        If True, print messages about the zipping process.

    Returns
    -------
    Path
        Path to the created ZIP file.
    """
    zip_path = BASE_PATH / ZIP_FILENAME
    with ZipFile(zip_path, "w") as zipf:
        for _, filename in DATASET_FILENAMES.items():
            file_path = BASE_PATH / filename
            if file_path.exists():
                zipf.write(file_path, arcname=filename)
                if verbose:
                    print(f"Added {filename} to {ZIP_FILENAME}")
            else:
                if verbose:
                    print(f"{filename} not found, please download it first.")
    if verbose:
        print(f"Created ZIP archive: {zip_path}")
    return zip_path


def check_file_age(source: str, from_zip: bool = False):
    """
    Checks the dataset file's age and prints a warning if it's outdated.

    Parameters
    ----------
    file_path : Path
        Path to the dataset file.
    zip_path : Path
        Path to the ZIP archive containing the dataset.
    source : str
        Identifier for the data source ('eu' or 'nasa').
    from_zip : bool, optional
        If True, the file was loaded from the ZIP archive.

    Returns
    -------
    int
        Age of the file in days.
    """
    file_path = BASE_PATH / DATASET_FILENAMES[source]
    if from_zip:
        zip_path = BASE_PATH / ZIP_FILENAME
        with ZipFile(zip_path, "r") as zipf:
            date_info = zipf.getinfo(DATASET_FILENAMES[source]).date_time
            creation = datetime.datetime(*date_info)
    else:
        creation = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
    age = (datetime.datetime.now() - creation).days
    print(f"Last modified: {creation} ({age} days ago)")
    return age


def load_dataset(
    source: str,
    check_age: bool = False,
    download_if_missing: bool = False,
    extract: bool = False,
    verbose: bool = True,
    store: bool = False,
):
    """
    Loads the dataset from a specified source and optionally extracts from ZIP.

    Parameters
    ----------
    source : str
        Identifier for the data source ('eu' or 'nasa').
    check_age : bool, optional
        If True, displays the last modified date of the dataset.
    download_if_missing : bool, optional
        If True, downloads the dataset if it's not found locally.
    extract : bool, optional
        If True, extracts the dataset from the ZIP archive.
    verbose : bool, optional
        If True, displays messages about the process.
    store : bool, optional
        If True, stores the loaded dataset in memory.

    Returns
    -------
    pd.DataFrame or None
        The loaded dataset as a DataFrame, or None if not found.
    """
    return _load_dataset_expanded(
        source=source,
        check_age=check_age,
        download_if_missing=download_if_missing,
        extract=extract,
        only_index=False,
        only_rows=None,
        verbose=verbose,
        store=store,
    )


def _load_dataset_expanded(
    source: str,
    check_age: bool = False,
    download_if_missing: bool = False,
    extract: bool = False,
    only_index: bool = False,
    only_rows: list | int = [],
    verbose: bool = True,
    store: bool = False,
):
    """
    Expands loading of the dataset with options for checking age and
    memory storage.

    Parameters
    ----------
    source : str
        Identifier for the data source ('eu' or 'nasa').
    check_age : bool, optional
        If True, displays the file's last modified date.
    download_if_missing : bool, optional
        If True, downloads if dataset is missing.
    extract : bool, optional
        If True, extracts from ZIP archive if available.
    only_index : bool, optional
        If True, loads only the index columns.
    only_rows : list|int, optional
        If provided, loads only the specified rows.
    verbose : bool, optional
        If True, prints messages about the process.
    store : bool, optional
        If True, stores the dataset in memory.

    Returns
    -------
    pd.DataFrame or None
        The loaded dataset as a DataFrame, or None if not found.
    """
    if source not in DATASET_FILENAMES:
        raise ValueError(f"Invalid source: {source}. Must be 'eu' or 'nasa'.")
    if only_rows and only_index:
        raise ValueError("Cannot specify both only_rows and only_index.")
    elif only_rows:
        whole = False
        if isinstance(only_rows, int):
            only_rows = [only_rows]
        if source == "nasa":
            only_rows = [
                x + 291 for x in only_rows
            ]  # NASA data starts at row 292
        only_rows = [0] + [
            x + 1 for x in set(only_rows)
        ]  # Add header row move 1-indexed

        def skip_rows(x):
            return x not in only_rows

    else:
        whole = True
        skip_rows = 291 if source == "nasa" else None

    if IN_MEMORY_DATASETS[source] is not None and not only_index:
        if verbose:
            print("Loading memory stored dataset.")
        return IN_MEMORY_DATASETS[source].copy()  # dataframes are mutable
    elif IN_MEMORY_INDEXES[source] is not None and only_index:
        if verbose:
            print("Loading memory stored index columns.")
        return IN_MEMORY_INDEXES[source].copy()  # dataframes are mutable

    file_path = BASE_PATH / DATASET_FILENAMES[source]
    zip_path = BASE_PATH / ZIP_FILENAME

    if only_index:
        usecols = INDEX_COLUMNS[source]
    else:
        usecols = None

    try:
        # Check if the .csv is in the .zip without extracting
        if not file_path.exists() and zip_path.exists():
            with ZipFile(zip_path, "r") as zipf:
                if DATASET_FILENAMES[source] in zipf.namelist():
                    if verbose:
                        aux = (
                            "only the index columns from "
                            if only_index
                            else ""
                        )
                        print(
                            f"Loading {aux}{DATASET_FILENAMES[source]} "
                            + f"directly from {ZIP_FILENAME}..."
                        )
                    # Load directly from the .zip
                    with zipf.open(DATASET_FILENAMES[source]) as file:
                        data = pd.read_csv(
                            file, skiprows=skip_rows, usecols=usecols
                        )
                        from_zip = True
                        if extract:
                            file.seek(0)
                            with open(file_path, "wb") as f:
                                f.write(file.read())
        else:
            # Fallback: Load the dataset from the extracted file if present
            data = pd.read_csv(file_path, skiprows=skip_rows, usecols=usecols)
            if verbose:
                print(f"Loading {file_path}...")
            from_zip = False

        if check_age:
            check_file_age(
                file_path=file_path,
                zip_path=zip_path,
                source=source,
                from_zip=from_zip,
            )

    except FileNotFoundError:
        if download_if_missing:
            print(f"{file_path} not found, attempting download...")
            download_dataset(source=source, verbose=verbose)
            data = pd.read_csv(file_path, skiprows=skip_rows, usecols=usecols)
        else:
            print(
                f"{file_path} not found.\n"
                + "Use download_if_missing=True to download."
            )
            return

    if store:
        if not only_index and whole:
            if verbose:
                print("Storing the entire dataset into memory.")
            IN_MEMORY_DATASETS[source] = data.copy()
            IN_MEMORY_INDEXES[source] = data[INDEX_COLUMNS[source]]
        elif only_index:
            if verbose:
                print("Storing the index columns into memory.")
            IN_MEMORY_INDEXES[source] = data.copy()

    return data


def clear_memory(source: str):
    """
    Clear the memory address of stored datasets.

    Parameters
    ----------
    source : str
        If provided, only clears the memory for the specified source.
        If 'both', clears both sources.
    """
    source = source.lower()
    if source == "both":
        for key in IN_MEMORY_DATASETS:
            IN_MEMORY_DATASETS[key] = None
        return
    if source not in IN_MEMORY_DATASETS:
        raise ValueError(f"Invalid source: {source}. Must be 'eu' or 'nasa'.")
    IN_MEMORY_DATASETS[source] = None
    return
