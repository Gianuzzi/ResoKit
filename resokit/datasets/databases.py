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
import warnings
from io import BytesIO
from pathlib import Path
from typing import Tuple, Union
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from resokit.core import df_to_resokit
from resokit.datasets.utils import (
    DATASET_DTYPES,
    remove_from_zip,
    request_data,
)
from resokit.utils.utils import parse_to_iter


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
IN_MEMORY_DATASETS = {"eu": pd.DataFrame(), "nasa": pd.DataFrame()}
IN_MEMORY_INDEXES = {"eu": None, "nasa": None}
IS_FULLY_STORED = {"eu": False, "nasa": False}

# Index columns for each dataset
INDEX_COLUMNS = {"eu": ["name", "star_name"], "nasa": ["pl_name", "hostname"]}

# =============================================================================
# FUNCTIONS
# =============================================================================


def update_dataset(
    source: str, verbose: bool = True, in_zip: bool = True, store: bool = False
) -> Union[Path, None]:
    """Update the dataset from a specified source and saves it locally.

    The dataset is downloaded from the provided URL and saved as a CSV file.
    This csv file can be stored in a ZIP archive if requested, updating it.

    Note: Requires the requests library to download the dataset.

    Parameters
    ----------
    source : str
        Identifier for the data source ('eu' or 'nasa').
    verbose : bool, optional. Default: True.
        If `True`, displays messages about the download and update
        process.
    in_zip : bool, optional. Default: True.
        If `True`, updates the dataset in the ZIP archive,
        otherwise updates the file directly.
    store : bool, optional. Default: False.
        If `True`, stores the dataset in memory.

    Returns
    -------
    updated : Path or None
        `Path` to the downloaded dataset (if `in_zip=False`),
        or `None` if `in_zip=True`.
    """
    source = source.lower()  # Ensure lowercase

    # Check if source is valid
    if source not in DATASET_FILENAMES:
        raise ValueError(f"Invalid source: {source}. Must be 'eu' or 'nasa'.")

    # Define paths and URLs
    file_path = BASE_PATH / DATASET_FILENAMES[source]
    url = DATASET_URLS[source]
    zip_path = BASE_PATH / ZIP_FILENAME

    # Download the dataset
    data = request_data(url, verbose=verbose)

    # Create a dataframe from the data
    df = pd.read_csv(BytesIO(data), dtype=DATASET_DTYPES[source])

    # Make a simple check to see if the data is valid
    if df.empty:
        raise ValueError(f"Empty dataset downloaded from {url}.")

    if verbose:
        print(" Data downloaded successfully.")

    # Store the data in memory
    if store:
        IN_MEMORY_INDEXES[source] = df[INDEX_COLUMNS[source]].copy()
        IN_MEMORY_DATASETS[source] = df.copy()
        IS_FULLY_STORED[source] = True

    # Save the file...

    # Check if we are updating the ZIP archive
    if in_zip:
        # Check if the ZIP archive exists
        if not zip_path.exists():
            warnings.warn(
                f"ZIP archive {ZIP_FILENAME} not found.", stacklevel=2
            )
            if verbose:
                print(f" Creating the ZIP archive {ZIP_FILENAME}...")
        else:
            # Remove the file from the ZIP archive
            remove_from_zip(
                zip_path, DATASET_FILENAMES[source], verbose=verbose
            )

        # Write (and create if necessary) the file to the ZIP archive
        with ZipFile(zip_path, "a", compression=ZIP_DEFLATED) as zipf:
            zipf.writestr(DATASET_FILENAMES[source], data)

        if verbose:
            print(f" Updated {DATASET_FILENAMES[source]} in {ZIP_FILENAME}")

        return

    # Check if the file exists
    if not file_path.exists():
        warnings.warn(f"File {file_path} not found.", stacklevel=2)
        if verbose:
            print(f" Creating the file {file_path}...")

    # Write the file
    with open(file_path, "wb") as f:
        f.write(data)

    if verbose:
        print(f" Updated {file_path}")

    return file_path


def check_file_age(source: str, from_zip: bool = True) -> int:
    """Check the dataset file's age and prints a warning if it's outdated.

    Parameters
    ----------
    source : str
        Identifier for the data source ('eu' or 'nasa').
    from_zip : bool, optional. Default: False.
        If `True`, check the file inside the ZIP archive.

    Returns
    -------
    age : int
        Age of the file in days.
    """
    source = source.lower()  # Ensure lowercase

    file_path = (
        BASE_PATH / DATASET_FILENAMES[source]
    )  # Path to the dataset file

    if not from_zip:
        try:
            creation = datetime.datetime.fromtimestamp(
                file_path.stat().st_mtime
            )
        except FileNotFoundError:
            warnings.warn(
                f"File {file_path} not found. Checking age from ZIP.",
                stacklevel=2,
            )
            from_zip = True

    if from_zip:  # Get the creation date from inside the ZIP archive

        zip_path = BASE_PATH / ZIP_FILENAME  # Path to the ZIP archive

        with ZipFile(zip_path, "r") as zipf:  # Open the ZIP archive
            date_info = zipf.getinfo(DATASET_FILENAMES[source]).date_time
            creation = datetime.datetime(*date_info)

    # Calculate age in days
    age = (datetime.datetime.now() - creation).days

    print(f"Last modified: {creation} ({age} days ago)")

    return age


def _load_stored_rows(
    source: str,
    rows: Union[list, None] = None,
    full: bool = False,
    copy: bool = True,
) -> Tuple[pd.DataFrame, list]:
    """Load specific rows by index from memory.

    Parameters
    ----------
    source : str
        Dataset source ('eu' or 'nasa').
    rows : list, optional
        Row indexes to load (0-indexed).
    full : bool, optional
        Whether to load the full dataset.
    copy : bool, optional
        Whether to return a copy of the DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, list]
        The loaded dataset as a DataFrame and a list of not stored rows.
    """
    if full:
        if not IS_FULLY_STORED[source]:
            raise ValueError(f"Source {source} is not fully stored.")
        aux = IN_MEMORY_DATASETS[source]
        not_stored = []

    elif rows is not None:
        stored = [x for x in rows if x in IN_MEMORY_DATASETS[source].index]
        aux = IN_MEMORY_DATASETS[source].loc[stored]
        not_stored = [x for x in rows if x not in stored]

    else:
        raise ValueError("No rows provided.")

    if copy:
        return aux.copy(), not_stored
    return aux, not_stored


def _store_rows(
    source: str,
    rows_df: pd.DataFrame,
    verbose: bool = True,
) -> None:
    """Store specific rows by index in memory.

    Parameters
    ----------
    source : str
        Dataset source ('eu' or 'nasa').
    rows_df : pd.DataFrame
        DataFrame with rows to store.
    verbose : bool, optional
        Whether to print informational messages.
    """
    if IS_FULLY_STORED[source] or rows_df.empty:
        return  # No need to store if already fully stored, or empty

    not_stored = [
        x for x in rows_df.index if x not in IN_MEMORY_DATASETS[source].index
    ]

    IN_MEMORY_DATASETS[source] = pd.concat(
        [IN_MEMORY_DATASETS[source], rows_df.loc[not_stored]]
    )

    if verbose and not_stored:
        print(f" Stored rows {not_stored} in memory for source {source}.")

    return


def load_dataset(
    source: str,
    to_resokit: bool = True,
    check_age: bool = False,
    download_if_missing: bool = False,
    extract: bool = False,
    only_index: bool = False,
    only_rows: Union[list, int] = False,
    verbose: bool = True,
    store: bool = False,
    store_index: bool = True,
) -> pd.DataFrame:
    """Load the dataset from a specified source and optionally extract it.

    The dataset is loaded from the provided CSV file and stored in memory.
    If the dataset is already stored in memory, it is returned directly.

    Parameters
    ----------
    source : str
        Identifier for the data source ('eu' or 'nasa').
    to_resokit : bool, optional. Default: True.
        If `True`, returns the dataset including only the columns
        required by ResoKit.
        Note: This option is only available for the full dataset.
    check_age : bool, optional. Default: False.
        If `True`, displays the file's last modified date.
        used by ResoKit.
    download_if_missing : bool, optional
        If `True`, downloads if dataset is missing.
    extract : bool, optional. Default: False.
        If `True`, extracts from ZIP archive if available.
    only_index : bool, optional. Default: False.
        If `True`, loads only the index columns.
    only_rows : list|int, optional. Default: [].
        If provided, loads only the specified rows.
        Remember that python is 0-indexed, so
        the first row (system) is 0.
    verbose : bool, optional. Default: True.
        If `True`, prints messages about the process.
    store : bool, optional. Default: False.
        If `True`, stores the dataset in memory.
    store_index : bool, optional. Default: True.
        If `True`, stores the dataset index in memory.

    Returns
    -------
    dataset : DataFrame
        The loaded dataset as a pandas Data frame.
    """
    source = source.lower()  # Ensure lowercase

    if source not in DATASET_FILENAMES:  # Check if source is valid
        raise ValueError(f"Invalid source: {source}. Must be 'eu' or 'nasa'.")

    # Check store_index
    if store and only_index:
        store_index = True

    # Check if only rows and only index
    if (
        only_rows and only_index
    ):  # Check if only one of the options is provided
        raise ValueError("Cannot specify both only_rows and only_index.")

    elif (
        not isinstance(only_rows, bool) and isinstance(only_rows, int)
    ) or only_rows:  # If only_rows is provided, set up the skip_rows function

        if isinstance(only_rows, bool):
            raise ValueError("only_rows must be a list or an integer.")

        only_rows = parse_to_iter(only_rows)  # Convert to iterable

        # Remove duplicates
        seen = set()
        seen_add = seen.add
        requested_rows = [
            x for x in only_rows if not (x in seen or seen_add(x))
        ]

        # Check no negative values
        if any(x < 0 for x in requested_rows):
            raise ValueError("only_rows must be positive integers.")

        # Load stored rows if available
        data_stored, not_stored_rows = _load_stored_rows(
            source,
            rows=requested_rows,
            full=False,
        )

        # Define stored_rows and not_stored
        stored_rows = list(data_stored.index)

        # Message
        if verbose and not data_stored.empty:
            print(
                f" Loading memory stored rows {stored_rows} "
                + f"from {source}..."
            )

        if len(data_stored) == len(requested_rows):
            if to_resokit:
                return df_to_resokit(
                    data_stored,
                    source=source,
                    drop=True,
                    copy=False,
                    sort_by=False,
                    return_df=True,
                )
            return data_stored

        # Add header and update only_rows
        only_rows = [0] + [
            x + 1 for x in requested_rows if x in not_stored_rows
        ]

        def skip_rows(x: int) -> bool:  # Skip rows not in the list
            return x not in only_rows

    elif only_rows:
        raise ValueError("only_rows must be a list or an integer.")

    else:  # If not only_rows...
        skip_rows = None
        only_rows = False

    # Check if the index columns are already stored in memory
    if only_index and IN_MEMORY_INDEXES[source] is not None:
        if verbose:
            print(" Loading memory stored index columns...")
        return IN_MEMORY_INDEXES[source].copy()  # dataframes are mutable

    # Check if the dataset is already stored in memory
    if not (only_index or only_rows) and IS_FULLY_STORED[source]:
        if verbose:
            print(" Loading memory stored dataset...")
        df = _load_stored_rows(source, full=True)[0]
        # Re-sort dataset
        df = df.reindex(sorted(df.index), copy=False)
        if to_resokit:
            return df_to_resokit(
                df,
                source=source,
                drop=True,
                copy=False,
                sort_by=False,
                return_df=True,
            )
        return df

    # Define paths and ZIP extraction flag
    file_path = BASE_PATH / DATASET_FILENAMES[source]
    zip_path = BASE_PATH / ZIP_FILENAME

    # Define columns to load
    usecols = INDEX_COLUMNS[source] if only_index else None

    # Aux message
    if verbose:  # Print message if verbose
        aux = (
            "only the index columns from "
            if only_index
            else (f"rows {not_stored_rows} from " if only_rows else "")
        )

    try:
        # Check if the .csv is in the .zip without extracting
        if not file_path.exists() and zip_path.exists():

            with ZipFile(zip_path, "r") as zipf:  # Open the ZIP archive

                if DATASET_FILENAMES[source] in zipf.namelist():
                    if verbose:  # Print message if verbose
                        print(
                            f" Loading {aux}{DATASET_FILENAMES[source]} "
                            + f"directly from {ZIP_FILENAME}..."
                        )
                    # Load directly from the .zip
                    with zipf.open(DATASET_FILENAMES[source]) as file:
                        data = pd.read_csv(
                            file,
                            header=0,
                            skiprows=skip_rows,
                            usecols=usecols,
                            dtype=DATASET_DTYPES[source],
                        )
                        from_zip = True

                        if extract:  # Extract the file if requested
                            file.seek(0)
                            with open(file_path, "wb") as f:
                                f.write(file.read())
                            if verbose:
                                print(
                                    f" Extracted {DATASET_FILENAMES[source]} "
                                    + f"from {ZIP_FILENAME} into {file_path}"
                                )

        else:

            # Fallback: Load the dataset from the extracted file if present
            data = pd.read_csv(
                file_path,
                header=0,
                skiprows=skip_rows,
                usecols=usecols,
                dtype=DATASET_DTYPES[source],
            )

            if verbose:  # Print message if verbose
                print(f" Loading {aux}{file_path} ")

            from_zip = False

        if check_age:  # Check the file's age if requested
            check_file_age(
                source=source,
                from_zip=from_zip,
            )

    except FileNotFoundError as error:
        if download_if_missing:

            print(f" {file_path} not found, attempting download...")
            update_dataset(source=source, verbose=verbose, in_zip=False)
            data = pd.read_csv(
                file_path,
                header=0,
                skiprows=skip_rows,
                usecols=usecols,
                dtype=DATASET_DTYPES[source],
            )
        else:

            print(
                f" {file_path} not found.\n"
                + "Use download_if_missing=True to download."
            )
            raise error

    # Check empty dataset
    if data.empty and not only_rows:
        warnings.warn("Empty dataset loaded.", stacklevel=2)

    # Reindex according to only_rows if provided
    elif only_rows:

        # Get ordered list of rows to keep
        sorted_rows = sorted(not_stored_rows)

        n_used_rows = len(data)  # Number of rows effectively used

        # Warn if the number of rows is less than the requested
        # This means that the user requested more rows than the dataset has
        if n_used_rows < len(sorted_rows):
            out_of_bounds_rows = sorted_rows[n_used_rows:]
            warnings.warn(
                f"Rows {out_of_bounds_rows} are out of bounds.",
                stacklevel=2,
            )

        used_rows = sorted_rows[:n_used_rows]  # Keep only the used rows

        # Reindex the dataset
        data.set_index(pd.Index(used_rows), inplace=True)

        # Concatenate the stored rows with the loaded rows
        if not data_stored.empty:
            data = pd.concat([data_stored, data])

        # Finally, get the original order
        new_index = [x for x in requested_rows if x in used_rows + stored_rows]

        data = data.reindex(new_index, copy=False)

    # Check storeing
    if not store_index and not store:  # If not storing, return the data
        if to_resokit:
            return df_to_resokit(
                data,
                source=source,
                drop=True,
                copy=False,
                sort_by=False,
                return_df=True,
            )
        return data

    # Store with only_rows
    if only_rows and store:
        _store_rows(source, rows_df=data, verbose=verbose)

    elif store_index and not only_rows and IN_MEMORY_INDEXES[source] is None:
        if verbose:
            print(" Storing the index columns into memory...")
        IN_MEMORY_INDEXES[source] = data[INDEX_COLUMNS[source]].copy()

    if (
        store
        and not only_index
        and not only_rows
        and not IS_FULLY_STORED[source]
    ):
        if verbose:
            print(" Storing the entire dataset into memory...")
        IN_MEMORY_DATASETS[source] = data.copy()
        IS_FULLY_STORED[source] = True

    # Return the dataset
    if to_resokit:
        return df_to_resokit(
            data,
            source=source,
            drop=True,
            copy=False,
            sort_by=False,
            return_df=True,
        )
    return data


def clear_memory(source: str, verbose: bool = True) -> None:
    """Clear the memory address of stored datasets.

    Parameters
    ----------
    source : str
        Source to clear ('eu' or 'nasa' or 'both').
    verbose : bool, optional. Default: True.
        If `True`, prints messages about the process.
    """
    source = source.lower()  # Ensure lowercase

    if source == "both":
        for key in IN_MEMORY_DATASETS:
            IN_MEMORY_DATASETS[key] = None  # Clear the memory address
            IS_FULLY_STORED[key] = False
            if verbose:
                print(f" Cleared memory for source: {key}")
        return

    if source not in IN_MEMORY_DATASETS:
        raise ValueError(f"Invalid source: {source}. Must be 'eu' or 'nasa'.")

    IN_MEMORY_DATASETS[source] = pd.DataFrame()  # Clear the memory address
    IS_FULLY_STORED[source] = False  # Reset the fully stored flag

    if verbose:
        print(f" Cleared memory for source: {source}")

    return
