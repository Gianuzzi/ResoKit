# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2024, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Module to manage provided multi-star system datasets in text format."""

# =============================================================================
# IMPORTS
# =============================================================================

import os
from io import StringIO, TextIOWrapper
from itertools import product
from pathlib import Path
from typing import BinaryIO, Callable, List, Tuple, Union
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from resokit.core import StaticBinaryStar, _create_static_star
from resokit.datasets.utils import (
    ZIP_FILENAME,
    load_from_zip,
    remove_from_zip,
    request_dataset,
)
from resokit.utils.utils import DEFAULT_METADATA

# =============================================================================
# CONSTANTS
# =============================================================================

# Base directory path
BASE_PATH = Path(os.path.abspath(os.path.dirname(__file__)))

# Define the sources
_BINARIES_URLS = {
    "p": "https://lesia.obspm.fr/perso/philippe-thebault/plan_circ.txt",
    "s": "https://lesia.obspm.fr/perso/philippe-thebault/plan_bin500au.txt",
}

# Filenames and URLs for the datasets
_BINARIES_FILENAMES = {"p": "plan_circ.txt", "s": "plan_bin500au.txt"}

_BINARIES_COLUMNS = [
    "star1_name",
    "alternate_name",
    "star1_mass",
    "star2_mass",
    "star_dist",
    "disc_method",
    "a",
    "e",
    "nplanets",
    "planet_a",
    "planet_e",
    "planet_mass",
    "planet_HW_crit",
    "imutual",
]

# =============================================================================
# VARIABLES
# =============================================================================

# Store the datasets in memory. These files have less than 300 rows, so we can
# force to store them in memory if needed or requested once. We wont even need
# to store them as ResoKit datasets, as we can just use pandas DataFrames.
_IN_MEMORY_BINARIES_HEADERS = {
    "p": "",
    "s": "",
}
_IN_MEMORY_BINARIES = {
    "p": pd.DataFrame(),
    "s": pd.DataFrame(),
}

# =============================================================================
# FUNCTIONS
# =============================================================================


def _extract_header_and_data(
    lines: List[str], circumbinary: bool, inferr: bool
) -> Tuple[str, pd.DataFrame]:
    """Extract header and data from lines of the dataset."""
    # Find the index of the last line that starts with "------"
    # (or any number of hyphens)
    separator_index = next(
        i
        for i, line in enumerate(reversed(lines))
        if line.strip().startswith("-")
    )
    separator_index = len(lines) - separator_index

    # The header is everything before the separator line
    header = "".join(lines[:separator_index]).strip()

    # The data starts after the "----------" line, so we extract the data
    data_lines = [line.replace("\t", " ") for line in lines[separator_index:]]

    # Define widths for fixed-width formatted data
    kwargs = {}
    if inferr:
        kwargs["colspecs"] = "infer"
    elif circumbinary:
        kwargs["widths"] = [15, 10, 6, 6, 8, 2, 7, 7, 2, 10, 6, 9, 7, 8]
    else:
        kwargs["widths"] = [15, 10, 6, 6, 8, 2, 7, 7, 2, 8, 6, 9, 7, 8]

    # Use pandas to read the fixed-width formatted data
    # starting after the header
    data = pd.read_fwf(StringIO("".join(data_lines)), header=None, **kwargs)

    return header, data


def load_binary(
    circumbinary: bool = False,
    inferr: bool = False,
    ret_header: bool = False,
    from_memory: bool = True,
    from_zip: Union[str, bool] = True,
    from_file: Union[str, bool] = True,
    dir_path: Union[str, Path, bool] = True,
    rename_columns: bool = True,
    clean: bool = True,
    verbose: bool = True,
) -> Union[pd.DataFrame, str]:
    """Read the provided multi-star system dataset.

    Note
    ----
    If both `from_file` and `from_zip` are provided, it is assumed that the
    file inside the ZIP archive is the same as the one provided in `from_file`.
    Finally, the path constructed is: `dir_path / zip_name / file_name`.

    Parameters
    ----------
    circumbinary : bool, optional. Default is False.
        If True, read the circumbinary dataset.
        If False, read the binary dataset.
    inferr : bool, optional. Default is False.
        If True, read the inferred dataset. If False, read the observed dataset.
    ret_header : bool, optional. Default is False.
        If True, return the header.
        If False, return the data.
    from_memory : bool, optional. Default: True.
        If `True`, loads the dataset from memory if available.
    from_zip : str or Path or bool, optional. Default: True.
        Path to the ZIP archive to load the dataset.
        If `True`, default ZIP filename is used. (datasets.zip)
        If `False`, the file is not loaded from the ZIP archive.
    from_file : str or Path or bool, optional. Default: True.
        Path to the file to load the dataset.
        If `True`, default filename is used.
        If `False`, the file is not loaded.
    dir_path : str, Path or bool, optional. Default: True.
        Directory path to load the dataset from.
        If `True` or `None` the default directory is used. (resokit.datasets)
    rename_columns : bool, optional. Default is True.
        If True, rename the columns for human readability.
    clean : bool, optional. Default is True.
        If True, replace the unknown values with NaN.
    verbose : bool, optional. Default is False.
        If True, print the header and messages.

    Returns
    -------
    Union[pd.DataFrame, str]
        header : str if ret_header is True.
            The header of the dataset.
        data : pd.DataFrame if ret_header is False.
    """
    # Assert the circumbinary parameter
    if not isinstance(circumbinary, bool):
        raise TypeError("circumbinary must be a boolean.")
    # Define the filename based on the circumbinary parameter
    letter = "p" if circumbinary else "s"

    # Check if something to do
    if not from_memory and not from_zip and not from_file:
        raise ValueError(
            "Nothing to do. Set at least one of "
            + "from_memory, from_zip, or from_file."
        )

    # Redefine dir path
    if dir_path is None or dir_path is True:
        dir_path = BASE_PATH
        # Default directory
        dir_path = BASE_PATH
    elif not dir_path:
        # Assuming no file or ZIP required
        from_zip = False
        from_file = False
    else:
        # Convert to Path
        dir_path = Path(dir_path)

    # Define paths and ZIP extraction flag
    if from_zip:
        if from_zip is True:
            from_zip = ZIP_FILENAME

    # Define file path
    if from_file:
        if from_file is True:
            from_file = _BINARIES_FILENAMES[letter]

    # Load the dataset from memory
    if from_memory:
        if ret_header and _IN_MEMORY_BINARIES_HEADERS[letter] != "":
            if verbose:
                print(f"Loading the type-{letter} header from memory.")
            return str(_IN_MEMORY_BINARIES_HEADERS[letter])  # Return a copy
        elif not _IN_MEMORY_BINARIES[letter].empty:
            if verbose:
                print(f"Loading the type-{letter} dataset from memory.")
            df = _IN_MEMORY_BINARIES[letter].copy()
            # Rename columns if requested
            if rename_columns:
                df.columns = _BINARIES_COLUMNS
            return df

    # Load the dataset from the ZIP archive
    if from_zip:
        if verbose:
            print(f"Loading the type-{letter} dataset from the ZIP archive.")
        zip_path = dir_path / from_zip
        file_name = from_file if from_file else _BINARIES_FILENAMES[letter]
        my_open: Callable[[BinaryIO], List[str]] = lambda file: TextIOWrapper(
            file, encoding="utf-8"
        ).readlines()
        lines = load_from_zip(
            zip_path=zip_path,
            file_name=file_name,
            verbose=verbose,
            custom_load=my_open,
        )

    # Load the dataset from the file
    elif from_file:
        if verbose:
            print(f"Loading the type-{letter} dataset from the file.")
        file_path = dir_path / from_file
        with open(file_path, "r") as f:
            lines = f.readlines()

    # Extract header and data from lines
    header, data = _extract_header_and_data(
        lines=lines, circumbinary=circumbinary, inferr=inferr
    )

    # Store the data and header in memory
    _IN_MEMORY_BINARIES_HEADERS[letter] = header
    _IN_MEMORY_BINARIES[letter] = data
    if verbose:
        print(f"Stored the type-{letter} dataset and header into memory.")

    # Clean data
    if clean:
        data.loc[data[7] > 98, 7] = pd.NA  # eccentricity
        data.loc[data[13] > 998, 13] = pd.NA  # imutual

    # Rename columns
    if rename_columns:
        data.columns = _BINARIES_COLUMNS

    # Return the header if requested
    if ret_header:
        return header

    return data


def download_binary(
    circumbinary: bool,
    dir_path: Union[str, Path, None] = None,
    to_file: Union[str, Path, bool, None] = False,
    to_zip: Union[str, Path, bool, None] = False,
    to_memory: bool = True,
    return_data: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
    chunk_size: int = 1024,
    print_size: float = 0.00001,
) -> Union[Path, pd.DataFrame, None]:
    """Download a dataset from a specified source and save it locally.

    The dataset is downloaded from the internet and can be stored in a file,
    a ZIP archive, in memory, and/or simply returned.

    Note
    ----
    Requires the requests library.

    Parameters
    ----------
    circumbinary : bool
        If True, download the circumbinary dataset.
    dir_path : str or Path
        Directory path to save the dataset, or path to the ZIP archive.
        If `None`, the default directory is used (resokit.datasets).
    to_file : str or Path or bool, optional. Default: False.
        Path or str to the file to store the dataset.
        If `True`, default filename is used.
        If `False`, the file is not saved nor created.
    to_zip : str or Path or bool, optional. Default: False.
        Path or str to the ZIP archive to store the dataset.
        If `True`, default ZIP filename is used. (datasets.zip)
        If `False`, the file is not saved nor created in the ZIP archive.
    to_memory : bool, optional. Default: False.
        If `True`, stores the dataset in memory.
    return_data : bool, optional. Default: True.
        If `True`, returns the dataset.
    overwrite : bool, optional. Default: False.
        If `True`, overwrites the file if it already exists.
        It also overwrites the stored dataset in memory.
    verbose : bool, optional. Default: True.
        If `True`, displays messages about the download process.
    chunk_size : int, optional. Default: 1024.
        Size of the chunks to download the dataset, in bytes.
        Default is 1024 bytes (1 KB).
    print_size: float, optional. Default: 0.15.
        Update frequency for the download progress bar.

    Returns
    -------
    downloaded : Path or pd.DataFrame or str or None
        `Path` to the downloaded dataset (and or zip archive),
        or the dataset if return_data is `True`, or `None`.
    """
    # Define the letter based on the circumbinary parameter
    letter = "p" if circumbinary else "s"

    # Check if something to do
    if not to_file and not to_zip and not to_memory and not return_data:
        raise ValueError(
            "Nothing to do. Set at least one of "
            + "to_file, to_zip, to_memory, or to_resokit."
        )
    if (
        not to_file
        and not to_zip
        and to_memory
        and not return_data
        and not _IN_MEMORY_BINARIES[letter].empty
        and not overwrite
    ):
        raise ValueError(
            "Nothing to do. Dataset is already stored in memory and "
            + "overwrite is False."
        )

    # Define URS
    url = _BINARIES_URLS[letter]

    # Define path
    if dir_path is None:
        dir_path = BASE_PATH
    else:
        dir_path = Path(dir_path)

    # Check if directory exists
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory {dir_path} not found.")

    # Check if file exists
    if to_file:
        if to_file is True:
            file_name = _BINARIES_FILENAMES[letter]
        else:
            file_name = to_file
        file_path = dir_path / file_name
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"File {file_path} already exists.")
    else:
        file_path = None
        file_name = _BINARIES_FILENAMES[letter]

    # Check if zip exists
    if to_zip:
        if to_zip is True:
            zip_name = ZIP_FILENAME
        else:
            zip_name = to_zip
        zip_path = dir_path / zip_name
        if zip_path.exists() and not overwrite:
            raise FileExistsError(f"ZIP archive {zip_path} already exists.")
    else:
        zip_path = None
        zip_name = ZIP_FILENAME

    # Download the dataset
    data = request_dataset(
        url, verbose=verbose, chunk_size=chunk_size, print_size=print_size
    )

    # Check if the data is valid. If not, raise an error. Check length > 0 too
    if not data or len(data) == 0:
        raise ValueError(f"Empty dataset downloaded from {url}.")
    elif verbose:
        if len(data) < 1e6:
            print(f" Data downloaded successfully. ({len(data)/1e3:.3f} KB)")
        else:
            print(f" Data downloaded successfully. ({len(data)/1e6:.3f} MB)")

    # Store the data in ZIP
    if zip_path is not None:
        if not zip_path.exists():
            if verbose:
                print(f" Creating the ZIP archive {zip_path}...")
        else:
            # Remove the file from the ZIP archive
            remove_from_zip(zip_path, file_name, verbose=verbose)
        # Write (and create if necessary) the file to the ZIP archive
        with ZipFile(zip_path, "a", compression=ZIP_DEFLATED) as zipf:
            zipf.writestr(file_name, data)
        # Print message
        if verbose:
            print(f" Written {file_name} to {zip_path}.")

    # Store the data in file
    if file_path is not None:
        if not file_path.exists() and verbose:
            print(f" Creating the file {file_path}...")
        # Write the file
        with open(file_path, "wb") as f:
            f.write(data)
        # Print message
        if verbose:
            print(f" Written {file_path}.")

    # Store the data in memory? Only if to_memory or return_data
    if to_memory or return_data:
        header, df = _extract_header_and_data(
            lines=StringIO(data.decode(encoding="utf-8")).readlines(),
            circumbinary=circumbinary,
            inferr=False,
        )
        if to_memory:
            # Store the data in memory
            _IN_MEMORY_BINARIES_HEADERS[letter] = header
            _IN_MEMORY_BINARIES[letter] = df
            if verbose:
                print(f" Stored the type-{letter} dataset in memory.")

    # Return the data
    if return_data:
        # Try to rename the columns
        try:
            df.columns = _BINARIES_COLUMNS
        except ValueError:
            if verbose:
                print("Columns could not be renamed.")
        return df

    # Return the path
    if file_path and zip_path:
        return file_path, zip_path
    if file_path:
        return file_path
    if zip_path:
        return zip_path

    return


def _create_static_binary_star_from_binary(
    binary_row: pd.Series,
    source="user",
    metadata=None,
) -> StaticBinaryStar:
    """Create a :py:class:`StaticBinaryStar` instance.

    Parameters
    ----------
    binary_row : pd.Series
        Pandas Series with the binary data.
    source : str, optional. Default: 'user'.
        Source of the data.
    metadata : dict, optional. Default: {}.
        Additional metadata about the star.

    Returns
    -------
    StaticBinaryStar
        A new StaticBinaryStar instance.
    """
    if metadata is None:
        metadata = {}

    # Get the systems name
    name = binary_row["star1_name"]
    star1_name = name + " A"
    star2_name = name + " B"

    # Create the necessary series for the StaticStar instances
    # First, the shared parameters we usually get in a StaticStar
    shared_resokit = binary_row[["star_dist", "disc_method"]].copy()
    # Then, the parameters for each star
    star1_resokit = pd.Series(
        {
            "star_name": star1_name,
            "star_mass": binary_row["star1_mass"],
        }
    )
    star2_resokit = pd.Series(
        {
            "star_name": star2_name,
            "star_mass": binary_row["star2_mass"],
        }
    )

    # Create the StaticStar dfs
    star1_df = pd.concat([shared_resokit, star1_resokit], axis=0)
    star2_df = pd.concat([shared_resokit, star2_resokit], axis=0)

    # Rename the columns. If the column starts with "star_" we remove it.
    star1_df.rename(lambda x: str(x).replace("star_", ""), inplace=True)
    star2_df.rename(lambda x: str(x).replace("star_", ""), inplace=True)

    # Set the dataframe names
    star1_df.name = star1_name
    star2_df.name = star2_name

    # Create the StaticStar instances
    star1 = _create_static_star(star1_df, source=source, metadata=metadata)
    star2 = _create_static_star(star2_df, source=source, metadata=metadata)

    # Add the total binary mass to binary_row
    binary_row["mass"] = binary_row["star1_mass"] + binary_row["star2_mass"]

    # Create the StaticBinaryStar instance
    return StaticBinaryStar(
        star1=star1,
        star2=star2,
        binary_df=binary_row.to_frame(name=name).T,
        name=name,
        metadata=metadata,
    )


def load_from_binary(
    name: str, soft: bool = True, as_pandas: bool = False, verbose: bool = True
) -> StaticBinaryStar:
    """Load a binary star system from the dataset.

    Parameters
    ----------
    name : str
        Name of the binary star system to load.
    soft : bool, optional. Default is True.
        If True, return None if the star is not found.
        If False, raise an error if the star is not found.
    as_pandas : bool, optional. Default is False.
        If True, return the data as a pandas DataFrame.
    verbose : bool, optional. Default is True.
        If True, print messages.

    Returns
    -------
    StaticBinaryStar
        The loaded binary star system.
    """
    # Load the datasets
    datas = load_binary(
        circumbinary=False, verbose=verbose, rename_columns=True, clean=True
    )
    datap = load_binary(
        circumbinary=True, verbose=verbose, rename_columns=True, clean=True
    )

    # Find the row with the given name. It can be in either dataset,
    # so we try both. It can be in the star1_name or alternate_name columns.
    for dataset, col in product(
        [datas, datap], ["star1_name", "alternate_name"]
    ):
        row = dataset[dataset[col] == name]
        if not row.empty:
            # Which dataset was it? circumbinary means it was in datap
            circumbinary = True if dataset is datap else False
            break

    # Check if the row was found
    if row.empty:
        if soft:
            if verbose:
                print(f"Star {name} not found in the binary datasets.")
            return None
        raise ValueError(f"Star {name} not found in the binary datasets.")

    # Message if found
    if verbose:
        print(f"Star {name} found in the binary datasets.")

    # Extract the data
    row = row.iloc[0]

    # Return as a pandas DataFrame if requested
    if as_pandas:
        return row

    # Add metadata
    metadata = dict(DEFAULT_METADATA)
    metadata["circumbinary"] = circumbinary

    # Define the star system
    binary = _create_static_binary_star_from_binary(
        row, source="binary", metadata=metadata
    )

    return binary
