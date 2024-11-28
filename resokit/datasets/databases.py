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

import datetime
import os
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import pandas as pd

from resokit.utils.utils import assert_module_imported

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
) -> Union[Path, pd.DataFrame, None]:
    """
    Downloads the dataset from a specified source and saves it locally as CSV.

    Parameters
    ----------
    source : str
        Identifier for the data source ('eu' or 'nasa').
    overwrite : bool, optional. Default: False.
        If True, overwrites the existing file if it exists.
    verbose : bool, optional. Default: True.
        If True, displays messages about the download process.
    return_data : bool, optional. Default: False.
        If True, returns the downloaded dataset as a DataFrame.
    store : bool, optional. Default: False.
        If True, stores the dataset in memory.

    Returns
    -------
    Path or None or pd.DataFrame
        Path to the downloaded dataset
        or the dataset itself if return_data=True,
        or None if the file already exists and overwrite=False.
    """

    # Check if requests is imported
    assert_module_imported(requests_imported, "requests")

    source = source.lower()  # Ensure lowercase

    # Check if source is valid
    if source not in DATASET_FILENAMES:
        raise ValueError(f"Invalid source: {source}. Must be 'eu' or 'nasa'.")

    # Define paths and URLs
    file_path = BASE_PATH / DATASET_FILENAMES[source]
    url = DATASET_URLS[source]
    zip_path = BASE_PATH / ZIP_FILENAME

    if (  # Check if the file exists and not overwrite
        file_path.exists() and not overwrite
    ):
        if verbose:
            print(
                f" {file_path} already exists. \n"
                + "Use overwrite=True to redownload and overwrite."
            )

        return

    elif (  # Check if ZIP exists and contains the file, but not overwrite
        zip_path.exists()
        and DATASET_FILENAMES[source] in ZipFile(zip_path).namelist()
        and not overwrite
    ):

        if verbose:
            print(
                f" {DATASET_FILENAMES[source]} found in {ZIP_FILENAME}. \n"
                + "Use overwrite=True to redownload, or run load_dataset"
                + f"(source='{source}', extract=True) to extract it."
            )

        return

    # Download if not found
    if verbose:
        print(f" Downloading data from {url}...")

    response = requests.get(url=url)  # Download the file
    response.raise_for_status()  # Check for errors

    # Write the file
    if verbose:
        print(" Writing data to file...")

    # Save the file
    with open(file_path, "wb") as f:
        f.write(response.content)

    # Print message if verbose
    if verbose:
        print(f" File {file_path} successfully downloaded and saved.")

    # Store in memory if requested, and return if requested
    if return_data:
        return load_dataset(source=source, verbose=verbose, store=store)

    return file_path


def create_zip_archive(overwrite: bool = False, verbose: bool = True) -> Path:
    """
    Create a ZIP archive containing both EU and NASA dataset CSV files.

    Parameters
    ----------
    overwrite : bool, optional. Default: False.
        If True, overwrites the existing ZIP file if it exists.
    verbose : bool, optional. Default: True.
        If True, print messages about the zipping process.

    Returns
    -------
    Path
        Path to the created ZIP file.
    """

    zip_path = BASE_PATH / ZIP_FILENAME  # Path to the ZIP archive

    if zip_path.exists() and not overwrite:  # Check if ZIP already exists
        raise FileExistsError(
            "ZIP archive already exists. Use overwrite=True."
        )

    with ZipFile(zip_path, "w") as zipf:
        for _, filename in DATASET_FILENAMES.items():

            file_path = BASE_PATH / filename  # Path to the dataset file

            # Check if the file exists before adding to ZIP
            if file_path.exists():
                zipf.write(file_path, arcname=filename)  # Add to ZIP
                if verbose:
                    print(f" Added {filename} to {ZIP_FILENAME}")
            else:
                raise FileNotFoundError(
                    f"{filename} not found, please download it first."
                )

    # Print message if verbose
    if verbose:
        print(f" Created ZIP archive: {zip_path}")

    return zip_path


def check_file_age(source: str, from_zip: bool = False) -> int:
    """
    Checks the dataset file's age and prints a warning if it's outdated.

    Parameters
    ----------
    source : str
        Identifier for the data source ('eu' or 'nasa').
    from_zip : bool, optional. Default: False.
        If True, the file was loaded from the ZIP archive.

    Returns
    -------
    int
        Age of the file in days.
    """

    source = source.lower()  # Ensure lowercase

    file_path = (
        BASE_PATH / DATASET_FILENAMES[source]
    )  # Path to the dataset file

    if from_zip:  # Get the creation date from inside the ZIP archive

        zip_path = BASE_PATH / ZIP_FILENAME  # Path to the ZIP archive

        with ZipFile(zip_path, "r") as zipf:  # Open the ZIP archive
            date_info = zipf.getinfo(DATASET_FILENAMES[source]).date_time
            creation = datetime.datetime(*date_info)
    else:
        creation = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)

    # Calculate age in days
    age = (datetime.datetime.now() - creation).days

    print(f"Last modified: {creation} ({age} days ago)")

    return age


def load_dataset(
    source: str,
    check_age: bool = False,
    download_if_missing: bool = False,
    extract: bool = False,
    only_index: bool = False,
    only_rows: Union[list, int] = [],
    verbose: bool = True,
    store: bool = False,
    store_index: bool = True,
) -> pd.DataFrame:
    """
    Loads the dataset from a specified source and optionally extracts from ZIP.

    Parameters
    ----------
    source : str
        Identifier for the data source ('eu' or 'nasa').
    check_age : bool, optional. Default: False.
        If True, displays the file's last modified date.
    download_if_missing : bool, optional
        If True, downloads if dataset is missing.
    extract : bool, optional. Default: False.
        If True, extracts from ZIP archive if available.
    only_index : bool, optional. Default: False.
        If True, loads only the index columns.
    only_rows : list|int, optional. Default: [].
        If provided, loads only the specified rows.
    verbose : bool, optional. Default: True.
        If True, prints messages about the process.
    store : bool, optional. Default: False.
        If True, stores the dataset in memory.
    store_index : bool, optional. Default: True.
        If True, stores the dataset index in memory.

    Returns
    -------
    pd.DataFrame or None
        The loaded dataset as a DataFrame.
    """

    source = source.lower()  # Ensure lowercase

    if source not in DATASET_FILENAMES:  # Check if source is valid
        raise ValueError(f"Invalid source: {source}. Must be 'eu' or 'nasa'.")

    if (
        only_rows and only_index
    ):  # Check if only one of the options is provided
        raise ValueError("Cannot specify both only_rows and only_index.")

    elif only_rows:  # If only_rows is provided, set up the skip_rows function

        whole = False  # Flag to store the whole dataset in memory

        if isinstance(only_rows, int):
            only_rows = [only_rows]

        if source == "nasa":
            only_rows = [
                x + 291 for x in only_rows
            ]  # NASA data starts at row 292

        only_rows = [0] + [
            x + 1 for x in set(only_rows)
        ]  # Add header row move 1-indexed

        def skip_rows(x: int) -> bool:  # Skip rows not in the list
            return x not in only_rows

    else:  # If not only_rows...
        whole = True  # Flag to store the whole dataset in memory
        skip_rows = 291 if source == "nasa" else None

    # Check if the dataset is already stored in memory
    if IN_MEMORY_DATASETS[source] is not None and not only_index:
        if verbose:
            print(" Loading memory stored dataset...")
        return IN_MEMORY_DATASETS[source].copy()  # dataframes are mutable

    # Check if the index columns are already stored in memory
    elif IN_MEMORY_INDEXES[source] is not None and only_index:
        if verbose:
            print(" Loading memory stored index columns...")
        return IN_MEMORY_INDEXES[source].copy()  # dataframes are mutable

    # Define paths and ZIP extraction flag
    file_path = BASE_PATH / DATASET_FILENAMES[source]
    zip_path = BASE_PATH / ZIP_FILENAME

    # Define columns to load
    if only_index:
        usecols = INDEX_COLUMNS[source]
    else:
        usecols = None

    # Define dtype for object columns in NASA dataset, to avoid mixed types
    dtype_dict = {4: "object", 5: "object"} if source == "nasa" else None

    try:
        # Check if the .csv is in the .zip without extracting
        if not file_path.exists() and zip_path.exists():

            with ZipFile(zip_path, "r") as zipf:  # Open the ZIP archive

                if DATASET_FILENAMES[source] in zipf.namelist():
                    if verbose:  # Print message if verbose
                        aux = (
                            "only the index columns from "
                            if only_index
                            else ""
                        )
                        print(
                            f" Loading {aux}{DATASET_FILENAMES[source]} "
                            + f"directly from {ZIP_FILENAME}..."
                        )

                    # Load directly from the .zip
                    with zipf.open(DATASET_FILENAMES[source]) as file:
                        data = pd.read_csv(
                            file,
                            skiprows=skip_rows,
                            usecols=usecols,
                            dtype=dtype_dict,
                        )
                        from_zip = True

                        if extract:  # Extract the file if requested
                            file.seek(0)
                            with open(file_path, "wb") as f:
                                f.write(file.read())

        else:

            # Fallback: Load the dataset from the extracted file if present
            data = pd.read_csv(
                file_path,
                skiprows=skip_rows,
                usecols=usecols,
                dtype=dtype_dict,
            )

            if verbose:  # Print message if verbose
                print(f" Loading {file_path}...")

            from_zip = False

        if check_age:  # Check the file's age if requested
            check_file_age(
                source=source,
                from_zip=from_zip,
            )

    except FileNotFoundError as error:
        if download_if_missing:

            print(f" {file_path} not found, attempting download...")
            download_dataset(source=source, verbose=verbose)
            data = pd.read_csv(file_path, skiprows=skip_rows, usecols=usecols)
        else:

            print(
                f" {file_path} not found.\n"
                + "Use download_if_missing=True to download."
            )
            raise error

    if not whole:  # Can't store just part of the dataset
        return data

    if store_index:
        if store and not only_index:
            if verbose:  # Print message if verbose
                print(" Storing the entire dataset into memory...")

            IN_MEMORY_DATASETS[source] = data.copy()
            IN_MEMORY_INDEXES[source] = data[INDEX_COLUMNS[source]]

        else:
            if verbose:  # Print message if verbose
                print(" Storing the index columns into memory...")

            IN_MEMORY_INDEXES[source] = data.copy()

    return data


def clear_memory(source: str) -> None:
    """
    Clear the memory address of stored datasets.

    Parameters
    ----------
    source : str
        If provided, only clears the memory for the specified source.
        If 'both', clears both sources.
    """

    source = source.lower()  # Ensure lowercase

    if source == "both":
        for key in IN_MEMORY_DATASETS:
            IN_MEMORY_DATASETS[key] = None  # Clear the memory address
        return

    if source not in IN_MEMORY_DATASETS:
        raise ValueError(f"Invalid source: {source}. Must be 'eu' or 'nasa'.")

    IN_MEMORY_DATASETS[source] = None  # Clear the memory address

    return
