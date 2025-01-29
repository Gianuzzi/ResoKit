# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2024, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Module to manage provided exoplanet datasets from exoplanet.eu and NASA."""

# =============================================================================
# IMPORTS
# =============================================================================

import datetime
import os
import warnings
from io import BytesIO, StringIO, TextIOWrapper
from pathlib import Path
from typing import BinaryIO, Callable, List, Tuple, Union
from zipfile import ZIP_DEFLATED, ZipFile

import attrs

import pandas as pd

from resokit.core import MetaData, df_to_resokit
from resokit.datasets.utils import (
    DATASET_DTYPES,
    ZIP_FILENAME,
    load_from_zip,
    remove_from_zip,
    request_dataset,
)
from resokit.utils.parser import (
    DEFAULT_METADATA,
    parse_name,
    parse_to_iter,
)

# =============================================================================
# CONSTANTS
# =============================================================================

# Base directory path
BASE_PATH = Path(os.path.abspath(os.path.dirname(__file__)))

# -------------------------- EU and NASA DATASETS -----------------------------

# Filenames and URLs for the datasets
_DATASET_FILENAMES = {"eu": "exoplanet_eu.csv", "nasa": "nasa.csv"}
_DATASET_URLS = {
    "eu": "https://exoplanet.eu/catalog/csv/",
    "nasa": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    + "query=select+*+from+ps&format=csv",
}

# Index columns for each dataset
_INDEX_COLUMNS = {"eu": ["name", "star_name"], "nasa": ["pl_name", "hostname"]}

# --------------------------- BINARY SYSTEMS DATASETS --------------------------

# Define the sources for the binaries datasets
_BINARIES_URLS = {
    "p": "https://lesia.obspm.fr/perso/philippe-thebault/plan_circ.txt",
    "s": "https://lesia.obspm.fr/perso/philippe-thebault/plan_bin500au.txt",
}

# Filenames and URLs for the binaries datasets
_BINARIES_FILENAMES = {"p": "plan_circ.txt", "s": "plan_bin500au.txt"}

# Columns of the binaries datasets
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
# CLASSES
# =============================================================================


@attrs.define(frozen=True, slots=True, repr=False)
class ResoKitDataset:
    """Class to store a ResoKit dataset.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset as a pandas DataFrame.
    source : str
        The source of the dataset ('eu' or 'nasa').
    age : int
        The age of the dataset in days.
    origin : str
        The origin of the dataset (file in zip, file, or mixed).
    is_full : bool
        Whether the dataset is complete.
    metadata : dict
        Metadata for the dataset.
    """

    dataset: pd.DataFrame = attrs.field(
        validator=attrs.validators.instance_of(pd.DataFrame),
    )
    source: str = attrs.field(
        validator=attrs.validators.in_({"eu", "nasa"}),
        converter=str.lower,
    )
    age: int = attrs.field(validator=attrs.validators.instance_of(int))
    origin: str = attrs.field(
        validator=attrs.validators.in_(
            {"file", "zip", "mixed", "internet", "null"}
        ),
        converter=str.lower,
    )
    is_full: bool = attrs.field(validator=attrs.validators.instance_of(bool))
    metadata: dict = attrs.field(converter=MetaData, factory=MetaData)

    def __attrs_post_init__(self):
        """Post-init method to set the metadata."""
        # Check wrong configurations
        if self.origin == "null":
            if self.age != -1:
                raise ValueError("Age must be -1 if origin is 'null'.")
            if self.is_full:
                raise ValueError("is_full must be False if origin is 'null'.")
            if not self.dataset.empty:
                raise ValueError("Dataset must be empty if origin is 'null'.")
        if self.age < 0:
            if self.age != -1:
                raise ValueError("Age must be -1 or positive.")
            if self.is_full:
                raise ValueError("is_full must be False if age is -1.")
            if not self.dataset.empty:
                raise ValueError("Dataset must be empty if age is -1.")
        if self.is_full:
            if self.dataset.empty:
                raise ValueError("Dataset cannot be empty if is_full is True.")

    def __len__(self):
        """len(x) <=> x.__len__()."""
        return len(self.dataset)

    def __getitem__(self, key):
        """x[y] <==> x.__getitem__(y)."""
        sliced = self.dataset.__getitem__(key)
        if isinstance(sliced, pd.Series):
            sliced = sliced.to_frame().T
        is_full = self.is_full and len(sliced) == len(self.dataset)
        return ResoKitDataset(
            dataset=sliced,
            source=self.source,
            age=self.age,
            origin=self.origin,
            is_full=is_full,
            metadata=dict(self.metadata),
        )

    def __dir__(self):
        """dir(pdf) <==> pdf.__dir__()."""
        return super().__dir__() + dir(self.dataset)

    def __getattr__(self, a):
        """getattr(x, y) <==> x.__getattr__(y) <==> getattr(x, y)."""
        return getattr(self.dataset, a)

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""
        with pd.option_context("display.show_dimensions", False):
            df_body = repr(self.dataset).splitlines()
        # Construct the repr
        aux = "Full" if self.is_full else "Partial"
        parts = [
            f"{aux} ResokitDataset - {self.dataset.shape[0]} rows x "
            + f"{self.dataset.shape[1]} columns",
            f"Source: {self.source}",
            f"Age: {self.age} days",
            f"Origin: {self.origin}",
            *df_body,
        ]

        return "\n".join(parts)

    def _repr_html_(self):
        """Return a HTML representation of the DataFrame."""
        ad_id = id(self)  # Unique ID for the div container
        # Header and footer
        aux = "Full" if self.is_full else "Partial"
        rows = f"{self.dataset.shape[0]} rows"
        columns = f"{self.dataset.shape[1]} columns"
        footer = f" {aux} ResokitDataSet - {rows} x {columns}"
        # HTML representation of the DataFrame
        with pd.option_context("display.show_dimensions", False):
            df_html = self.dataset._repr_html_()
        # Construct the HTML
        parts = [
            f'<div class="resokit-data-container" id={ad_id}>',
            df_html,
            footer,
            "</div>",
        ]
        # Join the parts
        html = "".join(parts)

        return html

    def to_dataframe(
        self, columns: list = None, copy: bool = True, sort: bool = False
    ) -> pd.DataFrame:
        """Convert data to pandas data frame.

        This method constructs a data frame with the data inside the
        dataset attribute.

        Parameters
        ----------
        columns : list, optional. Default: None.
            Specific columns to return.
            If `None`, return all columns.
        copy : bool, optional. Default: True.
            Whether to return a copy of the `DataFrame`, or the original.
        sort : bool, optional. Default: False.
            Whether to sort the dataset by the index columns.

        Returns
        -------
        df: DataFrame
            Data frame with the requested columns.
        """
        if columns is not None:
            used_cols = [
                col for col in list(columns) if col in self.dataset.columns
            ]
            df = self.dataset[used_cols]
        else:
            df = self.dataset

        if copy and sort:
            return df.sort_index(inplace=False).copy()
        elif copy:
            return df.copy()
        elif sort:
            return df.sort_index(inplace=False)
        return df

    def to_dict(self) -> dict:
        """Convert metadata to a dictionary.

        This method constructs a dictionary with the data inside the
        metadata attribute. It also adds the age, source, and origin.

        Returns
        -------
        full_metadata : dict
            Dictionary with the metadata.
        """
        extra = {"age": self.age, "source": self.source, "origin": self.origin}
        return {
            **extra,
            **self.metadata,
        }

    def copy(self) -> "ResoKitDataset":
        """Create and return copy of the :py:class:`ResoKitDataset`.

        Returns
        -------
        ResoKitDataset
            Copy of the ResoKitDataset.
        """
        return ResoKitDataset(
            dataset=self.dataset.copy(),
            source=self.source,
            age=self.age,
            origin=self.origin,
            is_full=self.is_full,
            metadata=dict(self.metadata),
        )

    def to_resokit(self, sort: bool = False) -> "ResoKitDataset":
        """Convert the dataset to a pure ResoKitDataset.

        This method converts the dataset to a ResoKitDataset containing
          only the columns required by ResoKit.

        Parameters
        ----------
        sort : bool, optional. Default: False.
            Whether to sort the dataset by the index columns.

        Returns
        -------
        dataset : ResoKitDataset
            ResoKitDataset.
        """
        dataset = self.to_dataframe(copy=False, sort=sort)
        df = df_to_resokit(
            dataset,
            source=self.source,
            drop=True,
            copy=True,
            sort_by=False,
            return_df=True,
        )
        return ResoKitDataset(
            dataset=df,
            source=self.source,
            age=self.age,
            origin=self.origin,
            is_full=self.is_full,
            metadata=dict(self.metadata),
        )


# =============================================================================
# VARIABLES
# =============================================================================

# ---------------------- EU and NASA DATASETS IN MEMORY -----------------------

# Store the datasets in memory
_IN_MEMORY_INDEXES = {
    "eu": ResoKitDataset(
        dataset=pd.DataFrame(),
        source="eu",
        age=-1,
        origin="null",
        is_full=False,
    ),
    "nasa": ResoKitDataset(
        dataset=pd.DataFrame(),
        source="nasa",
        age=-1,
        origin="null",
        is_full=False,
    ),
}
_IN_MEMORY_DATASETS = {
    "eu": ResoKitDataset(
        dataset=pd.DataFrame(),
        source="eu",
        age=-1,
        origin="null",
        is_full=False,
    ),
    "nasa": ResoKitDataset(
        dataset=pd.DataFrame(),
        source="nasa",
        age=-1,
        origin="null",
        is_full=False,
    ),
}
_IS_FULLY_STORED = {"eu": False, "nasa": False}

# Store parsed indexes for even faster lookup
_IN_MEMORY_PARSED_INDEXES = {
    "eu": None,
    "nasa": None,
}

# -------------------- BINARY SYSTEMS DATASETS IN MEMORY ----------------------

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

# --------------------------- EU AND NASA DATASETS ----------------------------


def df_to_dataset(
    df: pd.DataFrame,
    source: str,
    age: int = -1,
    origin: str = "null",
    is_full: bool = False,
    metadata: dict = None,
    copy: bool = True,
    as_resokit: bool = True,
) -> ResoKitDataset:
    """Convert a pandas DataFrame to a ResoKitDataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to convert.
    source : str
        Source of the dataset ('eu' or 'nasa').
    age : int, optional. Default: -1.
        Age of the dataset in days.
    origin : str, optional. Default: 'unknown'.
        Origin of the dataset. Can be one of:
        ('file', 'zip', 'mixed', 'internet', or 'unknown').
    is_full : bool, optional. Default: False.
        Whether the dataset is complete.
    metadata : dict, optional. Default: None.
        Metadata for the dataset.
    copy : bool, optional. Default: True.
        Whether to return a copy of the DataFrame.
        Despite this, the output will be a `ResoKitDataset`.
    as_resokit : bool, optional. Default: True.
        Whether to perform the column conversion to ResoKit columns.

    Returns
    -------
    dataset : ResoKitDataset
        ResoKitDataset.
    """
    # Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a DataFrame. Got: {type(df)} instead.")

    if as_resokit:
        df = df_to_resokit(
            df,
            source=source,
            drop=True,
            copy=copy,
            sort_by=False,
            return_df=True,
        )

    if metadata is None:
        metadata = dict(DEFAULT_METADATA)

    return ResoKitDataset(
        dataset=df,
        source=source,
        age=age,
        origin=origin,
        is_full=is_full,
        metadata=metadata,
    )


def _mk_empty_dataset(source: str) -> ResoKitDataset:
    """Create an empty dataset.

    Parameters
    ----------
    source : str
        Source of the dataset ('eu' or 'nasa').

    Returns
    -------
    dataset : ResoKitDataset
        Empty ResoKitDataset.
    """
    return ResoKitDataset(
        dataset=pd.DataFrame(),
        source=source,
        age=-1,
        origin="null",
        is_full=False,
        metadata=dict(DEFAULT_METADATA),
    )


def _update_stored_dataset(
    new_df: pd.DataFrame,
    source: str,
    age: int,
    origin: str,
    is_full: bool,
    verbose: bool = True,
    index_only: bool = False,
    sort: bool = True,
    metadata: dict = None,
    overwrite: bool = False,
) -> None:
    """Update the stored dataset in memory.

    Parameters
    ----------
    new_df : pd.DataFrame
        New dataset to store.
    source : str
        Dataset source ('eu' or 'nasa').
    age : int
        Age of the dataset in days.
    origin : str
        Origin of the dataset.
    is_full : bool
        Whether the dataset is complete.
    verbose : bool, optional
        Whether to print informational messages.
    index_only : bool, optional
        Whether to store only the index.
    sort : bool, optional
        Whether to sort the dataset in memory.
        Index is always sorted (by default).
    metadata : dict, optional
        Metadata to update the dataset.
    overwrite : bool, optional
        Whether to overwrite the stored dataset.
    """
    if _IS_FULLY_STORED[source] and not overwrite:
        return  # No need to update if already fully stored

    if new_df.empty:
        if verbose:
            print(" No rows to store in memory.")
        return

    # Store the rows in the index
    if is_full:

        # Check if the index is already stored
        if not _IN_MEMORY_INDEXES[source].dataset.empty and not overwrite:
            return

        # Get the index columns
        new_index = new_df[_INDEX_COLUMNS[source]].copy()

        # Update the index. ONLY IF FULL READED THE INDEX
        _IN_MEMORY_INDEXES[source] = df_to_dataset(
            new_index,
            source=source,
            age=age,
            origin=origin,
            is_full=is_full,
            metadata=metadata,
            copy=True,
            as_resokit=False,
        )

        # Silently, parse and store the index
        _IN_MEMORY_PARSED_INDEXES[source] = new_index.astype(str)
        _IN_MEMORY_PARSED_INDEXES[source][_INDEX_COLUMNS[source][0]] = (
            _IN_MEMORY_PARSED_INDEXES[source][_INDEX_COLUMNS[source][0]].apply(
                parse_name
            )
        )
        _IN_MEMORY_PARSED_INDEXES[source][_INDEX_COLUMNS[source][1]] = (
            _IN_MEMORY_PARSED_INDEXES[source][_INDEX_COLUMNS[source][1]].apply(
                parse_name, force=True
            )
        )

        if verbose:
            print(" Updated stored index in memory.")

    if index_only:
        return

    if is_full or _IN_MEMORY_DATASETS[source].dataset.empty:
        # Update the stored dataset
        _IN_MEMORY_DATASETS[source] = df_to_dataset(
            new_df,
            source=source,
            age=age,
            origin=origin,
            is_full=is_full,
            metadata=metadata,
            copy=True,
            as_resokit=False,
        )

        if is_full:
            _IS_FULLY_STORED[source] = True
            if verbose:
                print(" Stored dataset in memory.")
        else:
            # Get the rows new to be stored
            new_to_store = new_df.index.to_list()
            if verbose:
                print(f" Stored rows {new_to_store} in memory...")
        return

    # Get the rows new to be stored
    new_to_store = [
        x
        for x in new_df.index
        if x not in _IN_MEMORY_DATASETS[source].dataset.index
    ]

    # Check if all rows are stored
    if not new_to_store and not overwrite:
        return

    elif new_to_store and not overwrite:
        # Remove the rows already stored
        new_df = new_df.loc[new_to_store]

        # Create updated dataset
        updated_df = pd.concat([_IN_MEMORY_DATASETS[source].dataset, new_df])

        # Get age and origin
        age_old = _IN_MEMORY_DATASETS[source].age
        origin_old = _IN_MEMORY_DATASETS[source].origin

        # Get metadata
        meta_old = dict(_IN_MEMORY_DATASETS[source].metadata)

    elif overwrite:
        repeated = [
            x
            for x in new_df.index
            if x in _IN_MEMORY_DATASETS[source].dataset.index
        ]
        # Something to store?
        if not repeated:
            return
        # Check if all stored is in the new dataset
        elif len(repeated) == len(_IN_MEMORY_DATASETS[source].dataset):
            # Create updated dataset
            updated_df = new_df

            # Get age and origin (from new)
            age_old = age
            origin_old = origin

            # Get metadata (from new)
            meta_old = metadata

        else:  # Partial overwrite
            to_keep = [
                x
                for x in _IN_MEMORY_DATASETS[source].dataset.index
                if x not in new_df.index
            ]

            # Extract the rows to keep
            keep_df = _IN_MEMORY_DATASETS[source].dataset.loc[to_keep]

            # Create updated dataset
            updated_df = pd.concat([keep_df, new_df])

            # Get age and origin
            age_old = _IN_MEMORY_DATASETS[source].age
            origin_old = _IN_MEMORY_DATASETS[source].origin

            # Get metadata
            meta_old = dict(_IN_MEMORY_DATASETS[source].metadata)

        # Update new_to_store
        new_to_store = new_df.index.to_list()

    # Sort the dataset
    if sort:
        updated_df.sort_index(inplace=True)

    # Check if metadata is provided
    if metadata is not None:
        meta_old.update(metadata)

    # Update the dataset
    _IN_MEMORY_DATASETS[source] = df_to_dataset(
        updated_df,
        source=source,
        age=max(age_old, age),
        origin="mixed" if origin_old != origin else origin_old,
        is_full=False,
        metadata=meta_old,
        copy=True,
        as_resokit=False,
    )

    if verbose:
        print(f" Stored rows {new_to_store} in memory.")

    return


def download(
    source: str,
    dir_path: Union[str, Path, None] = None,
    to_file: Union[str, Path, bool, None] = False,
    to_zip: Union[str, Path, bool, None] = False,
    to_memory: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
    to_resokit: Union[bool, None] = None,
    chunk_size: int = 1024,
    print_size: float = 0.15,
) -> Union[Path, pd.DataFrame, ResoKitDataset]:
    """Download a dataset from a specified source and save it locally.

    The dataset is downloaded from the internet, from the online NASA
    or exoplanet.eu databases, and can be stored in a file, a ZIP archive,
    in memory, and/or simply returned.

    Note
    ----
    Requires the requests library.

    Parameters
    ----------
    source : str
        Identifier for the data source ('eu' or 'nasa').
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
    overwrite : bool, optional. Default: False.
        If `True`, overwrites the file if it already exists.
        It also overwrites the stored dataset in memory.
    verbose : bool, optional. Default: True.
        If `True`, displays messages about the download process.
    to_resokit : bool, dict, optional. Default: None.
        If `True`, returns the dataset as a ResoKitDataset.
        If `False`, returns the dataset as a pandas DataFrame.
        If `None`, returns the path to the downloaded file.
    chunk_size : int, optional. Default: 1024.
        Size of the chunks to download the dataset, in bytes.
        Default is 1024 bytes (1 KB).
    print_size: float, optional. Default: 0.15.
        Update frequency for the download progress bar.

    Returns
    -------
    downloaded : Path or pd.DataFrame or None
        `Path` to the downloaded dataset (and or zip archive),
        or the dataset if `to_resokit` is not `None`.
    """
    # Check if source is valid
    source = source.lower()  # Ensure lowercase
    if source not in _DATASET_FILENAMES:
        raise ValueError(f"Invalid source: {source}. Must be 'eu' or 'nasa'.")

    # Check if something to do
    if not to_file and not to_zip and not to_memory and to_resokit is None:
        raise ValueError(
            "Nothing to do. Set at least one of "
            + "to_file, to_zip, to_memory, or to_resokit."
        )
    if (
        not to_file
        and not to_zip
        and to_memory
        and to_resokit is None
        and _IS_FULLY_STORED[source]
        and not overwrite
    ):
        raise ValueError(
            "Nothing to do. Dataset is already fully stored in memory and "
            + "overwrite is False."
        )

    # Define URS
    url = _DATASET_URLS[source]

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
            file_name = _DATASET_FILENAMES[source]
        else:
            file_name = to_file
        file_path = dir_path / file_name
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"File {file_path} already exists.")
    else:
        file_path = None
        file_name = _DATASET_FILENAMES[source]

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
        print(f" Data downloaded successfully. ({len(data)/1e6:.2f} MB)")

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

    # Store the data in memory? Only if to_memory or to_resokit
    if to_memory or to_resokit is not None:
        # Create a dataframe from the data
        df = pd.read_csv(BytesIO(data), dtype=DATASET_DTYPES[source])

    # Create metadata
    metadata = dict(
        {
            "downloaded": datetime.datetime.now().isoformat(),
            "url": url,
        }
    )
    metadata = dict(DEFAULT_METADATA, **metadata)

    # Store the data in memory
    if to_memory:
        _update_stored_dataset(
            df,
            source,
            age=0,
            origin="internet",
            is_full=True,
            verbose=verbose,
            index_only=False,
            sort=True,
            metadata=metadata,
            overwrite=overwrite,
        )

    # Return the data
    if to_resokit is not None:
        return df_to_dataset(
            df,
            source=source,
            age=0,
            origin="internet",
            is_full=True,
            copy=True,
            as_resokit=to_resokit,
        )

    # Return the path
    if file_path and zip_path:
        return file_path, zip_path
    if file_path:
        return file_path
    if zip_path:
        return zip_path

    return


def update_eu(**kwargs) -> ResoKitDataset:
    """Update the exoplanet.eu dataset.

    This function updates the exoplanet.eu dataset, downloading it from the
    internet, and storing it in a file, a ZIP archive, in memory, and/or
    simply returning it. It is a wrapper for the `download` function.

    Note
    ----
    Requires the requests library.

    Parameters
    ----------
    **kwargs
        Keyword arguments to pass to the `download` function.

    Returns
    -------
    dataset : ResoKitDataset
        Updated exoplanet.eu dataset.
    """
    return download("eu", **kwargs)


def update_nasa(**kwargs) -> ResoKitDataset:
    """Update the NASA dataset.

    This function updates the NASA dataset, downloading it from the internet,
    and storing it in a file, a ZIP archive, in memory, and/or simply returning
    it. It is a wrapper for the `download` function.

    Note
    ----
    Requires the requests library.

    Parameters
    ----------
    **kwargs
        Keyword arguments to pass to the `download` function.

    Returns
    -------
    dataset : ResoKitDataset
        Updated NASA dataset.
    """
    return download("nasa", **kwargs)


def _check_file_age(
    file_path: Union[str, Path],
    zip_path: Union[str, Path, None],
    verbose: bool = True,
) -> int:
    """Check the dataset file's age in days.

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
    # Get the file's last modified date
    if zip_path:
        file_name = Path(file_path).name
        with ZipFile(zip_path, "r") as zipf:  # Open the ZIP archive
            date_info = zipf.getinfo(file_name).date_time
            creation = datetime.datetime(*date_info)
    else:
        creation = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)

    # Calculate age in days
    age = (datetime.datetime.now() - creation).days

    if verbose:
        print(f"Last modified: {creation} ({age} days ago)")

    return age


def _load_stored_full(
    source: str,
    to_df: bool = False,
    to_resokit: bool = True,
    sort: bool = True,
) -> Union[pd.DataFrame, ResoKitDataset]:
    """Load the fully stored dataset from memory.

    Parameters
    ----------
    source : str
        Dataset source ('eu' or 'nasa').
    to_df : bool, optional
        Whether to return as a pandas DataFrame.
    to_resokit : bool, optional
        Whether to return the dataset as a ResoKitDataset
    sort : bool, optional
        Whether to sort the dataset by the index columns

    Returns
    -------
    data : Union[pd.DataFrame, ResoKitDataset]
        The loaded dataset as a DataFrame or a ResoKitDataset.
    """
    # Check if the dataset is fully stored
    if not _IS_FULLY_STORED[source]:
        raise ValueError(f"Source {source} is not fully stored.")
    # Return the dataset
    if not to_df:  # Return as a ResoKitDataset
        if sort:
            sortd = _IN_MEMORY_DATASETS[source].dataset.sort_index()
            return df_to_dataset(
                sortd,
                source=source,
                age=_IN_MEMORY_DATASETS[source].age,
                origin=_IN_MEMORY_DATASETS[source].origin,
                is_full=_IN_MEMORY_DATASETS[source].is_full,
                metadata=dict(_IN_MEMORY_DATASETS[source].metadata),
                copy=False,
                as_resokit=to_resokit,
            )
        elif to_resokit:  # Return as a ResoKitDataset
            return _IN_MEMORY_DATASETS[source].to_resokit()
        return _IN_MEMORY_DATASETS[source]  # Return as a ResoKitDaframe
    if not to_resokit:  # Return as a DataFrame
        return _IN_MEMORY_DATASETS[source].to_dataframe(sort=sort)
    return _IN_MEMORY_DATASETS[source].to_resokit().to_dataframe(sort=sort)


def _load_stored_rows(
    source: str,
    rows: Union[list, None] = None,
    full: bool = False,
) -> Union[Tuple[pd.DataFrame, list, int, str], ResoKitDataset]:
    """Load specific rows by index from memory.

    Parameters
    ----------
    source : str
        Dataset source ('eu' or 'nasa').
    rows : list, optional
        Row indexes to load (0-indexed).
    full : bool, optional
        Whether to load the full dataset.
        If `True`, the `rows` parameter is ignored,
        and the full dataset is loaded as a ResoKitDataset.

    Returns
    -------
    Tuple[pd.DataFrame, list, int, str] or ResoKitDataset
        The loaded dataset as a DataFrame, a list of the rows not stored,
        the stored dataset age, and stored dataset origin.
        If `full` is `True`, returns the dataset as a ResoKitDataset.
    """
    # If all rows are requested, check if the dataset is fully stored
    if full:
        return _load_stored_full(source)

    # If specific rows are requested
    if rows is not None:
        stored = [
            x for x in rows if x in _IN_MEMORY_DATASETS[source].dataset.index
        ]
        data = _IN_MEMORY_DATASETS[source].dataset.loc[stored].copy()
        not_stored = [x for x in rows if x not in stored]  # Get age and origin
        age = _IN_MEMORY_DATASETS[source].age
        origin = _IN_MEMORY_DATASETS[source].origin

        return data, not_stored, age, origin

    raise ValueError("No rows provided.")


def _load_stored_index(
    source: str, to_df: bool = False, to_resokit: bool = True
) -> Union[pd.DataFrame, ResoKitDataset]:
    """Load the stored index from memory.

    Parameters
    ----------
    source : str
        Dataset source ('eu' or 'nasa').
    to_df : bool, optional
        Whether to return the dataset as a pandas DataFrame.
    to_resokit : bool, optional
        Whether to return the dataset as a ResoKitDataset

    Returns
    -------
    dataset : pd.DataFrame or ResoKitDataset
        The loaded dataset as a DataFrame or a ResoKitDataset.
    """
    if not to_df:
        if to_resokit:
            return _IN_MEMORY_INDEXES[source].to_resokit()
        return _IN_MEMORY_INDEXES[source]
    return _IN_MEMORY_INDEXES[source].to_dataframe()


def load_eu(
    from_memory: bool = True,
    from_zip: Union[str, bool] = True,
    from_file: Union[str, bool] = True,
    dir_path: Union[str, Path, bool] = True,
    to_resokit: bool = True,
    check_age: bool = False,
    only_rows: Union[list, int] = False,
    verbose: bool = True,
    store: Union[bool, str] = False,
    store_index: Union[bool, str] = False,
) -> ResoKitDataset:
    """Load the exoplanet.eu dataset.

    The dataset is loaded from a ZIP archive or a CSV file, or from memory
    if already stored. The priority is given to the memory saved dataset,
    then to the zip archive, and finally to the file.

    Note
    ----
    Storing the dataset in memory is useful for faster access and to avoid
    reading the file multiple times.

    Note
    ----
    If both `from_file` and `from_zip` are provided, it is assumed that the
    file inside the ZIP archive is the same as the one provided in `from_file`.
    Finally, the path constructed is: `dir_path / zip_name / file_name`.

    Parameters
    ----------
    from_memory : bool, optional. Default: True.
        If `True`, loads the dataset from memory if available.
    from_zip : str or Path or bool, optional. Default: True.
        Path to the ZIP archive to load the dataset.
        If `True`, default ZIP filename is used. (datasets.zip)
        If `False`, the file is not loaded from the ZIP archive.
    from_file : str or Path or bool, optional. Default: True.
        Path to the file to load the dataset.
        If `True`, default filename is used. (exoplanet_eu.csv)
        If `False`, the file is not loaded.
    dir_path : str, Path or bool, optional. Default: True.
        Directory path to load the dataset from.
        If `True` or `None` the default directory is used. (resokit.datasets)
    to_resokit : bool, optional. Default: True.
        If `True`, returns the dataset with only the columns
        required by ResoKit.
    check_age : bool, optional. Default: False.
        If `True`, displays the file's last modified date.
    only_rows : list|int, optional. Default: [].
        If provided, loads only the specified rows.
        Remember that python is 0-indexed, so
        the first row (system) is 0.
    verbose : bool, optional. Default: True.
        If `True`, prints messages about the process.
    store : bool, str, optional. Default: False.
        If `True`, stores the dataset in memory.
        If `str` and starts with "f" or "y" or "s" or "o", then
        overwrites the stored dataset.
    store_index : bool, str, optional. Default: True.
        If `True`, stores the dataset index in memory.
        If `only_rows` is provided, the index is not stored.
        If `str` and starts with "f" or "y" or "s" or "o", then
        overwrites the stored dataset index.

    Returns
    -------
    dataset : ResoKitDataset
        The loaded dataset as a ResoKitDataset.
    """
    return load_full(
        source="eu",
        from_memory=from_memory,
        from_zip=from_zip,
        from_file=from_file,
        dir_path=dir_path,
        to_resokit=to_resokit,
        to_df=False,
        check_age=check_age,
        only_rows=only_rows,
        verbose=verbose,
        store=store,
        store_index=store_index,
    )


def load_nasa(
    from_memory: bool = True,
    from_zip: Union[str, bool] = True,
    from_file: Union[str, bool] = True,
    dir_path: Union[str, Path, None] = None,
    to_resokit: bool = True,
    check_age: bool = False,
    only_rows: Union[list, int] = False,
    verbose: bool = True,
    store: Union[bool, str] = False,
    store_index: Union[bool, str] = False,
) -> ResoKitDataset:
    """Load the nasa dataset.

    The dataset is loaded from a ZIP archive or a CSV file, or from memory
    if already stored. The priority is given to the memory saved dataset,
    then to the zip archive, and finally to the file.

    Note
    ----
    Storing the dataset in memory is useful for faster access and to avoid
    reading the file multiple times.

    Note
    ----
    If both `from_file` and `from_zip` are provided, it is assumed that the
    file inside the ZIP archive is the same as the one provided in `from_file`.
    Finally, the path constructed is: `dir_path / zip_name / file_name`.

    Parameters
    ----------
    from_memory : bool, optional. Default: True.
        If `True`, loads the dataset from memory if available.
    from_zip : str or Path or bool, optional. Default: True.
        Path to the ZIP archive to load the dataset.
        If `True`, default ZIP filename is used. (datasets.zip)
        If `False`, the file is not loaded from the ZIP archive.
    from_file : str or Path or bool, optional. Default: True.
        Path to the file to load the dataset.
        If `True`, default filename is used. (nasa.csv)
        If `False`, the file is not loaded.
    dir_path : str, Path or bool, optional. Default: True.
        Directory path to load the dataset from.
        If `True` or `None` the default directory is used. (resokit.datasets)
    to_resokit : bool, optional. Default: True.
        If `True`, returns the dataset with only the columns
        required by ResoKit.
    check_age : bool, optional. Default: False.
        If `True`, displays the file's last modified date.
    only_rows : list|int, optional. Default: [].
        If provided, loads only the specified rows.
        Remember that python is 0-indexed, so
        the first row (system) is 0.
    verbose : bool, optional. Default: True.
        If `True`, prints messages about the process.
    store : bool, str, optional. Default: False.
        If `True`, stores the dataset in memory.
        If `str` and starts with "f" or "y" or "s" or "o", then
        overwrites the stored dataset.
    store_index : bool, str, optional. Default: True.
        If `True`, stores the dataset index in memory.
        If `only_rows` is provided, the index is not stored.
        If `str` and starts with "f" or "y" or "s" or "o", then
        overwrites the stored dataset index.

    Returns
    -------
    dataset : ResoKitDataset
        The loaded dataset as a ResoKitDataset.
    """
    return load_full(
        source="nasa",
        from_memory=from_memory,
        from_zip=from_zip,
        from_file=from_file,
        dir_path=dir_path,
        to_resokit=to_resokit,
        to_df=False,
        check_age=check_age,
        only_rows=only_rows,
        verbose=verbose,
        store=store,
        store_index=store_index,
    )


def __aux_load_full(
    df: pd.DataFrame,
    source: str,
    age: int,
    origin: str,
    is_full: bool,
    to_resokit: bool,
    to_df: bool,
    metadata: dict = None,
) -> Union[pd.DataFrame, ResoKitDataset]:
    """Auxiliary function to load the dataset from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to load.
    source : str
        Identifier for the data source ('eu' or 'nasa').
    age : int
        Age of the dataset in days.
    origin : str
        Origin of the dataset.
    is_full : bool
        Whether the dataset is complete.
    to_resokit : bool
        Whether to return the dataset as a ResoKitDataset.
    to_df : bool
        Whether to return the raw DataFrame.
    metadata : dict, optional
        Metadata for the dataset, if any.

    Returns
    -------
    dataset : pd.DataFrame or ResoKitDataset
        The loaded dataset as a DataFrame or a ResoKitDataset
    """
    if not to_df:  # Return as ResoKitDataset
        return df_to_dataset(
            df,
            source=source,
            age=age,  # age from stored dataset
            origin=origin,  # origin from stored dataset
            is_full=is_full,
            metadata=metadata,
            copy=False,
            as_resokit=to_resokit,
        )
    if to_resokit:  # Return as resokit dataframe
        return df_to_resokit(
            df,
            source=source,
            drop=True,
            copy=False,
            sort_by=False,
            metadata=metadata,
            return_df=True,
        )
    return df  # Return as raw dataframe


def load_full(
    source: str,
    from_memory: bool = True,
    from_zip: Union[str, bool] = True,
    from_file: Union[str, bool] = True,
    dir_path: Union[str, Path, bool] = True,
    to_resokit: bool = True,
    to_df: bool = False,
    check_age: bool = False,
    only_index: bool = False,
    only_rows: Union[list, int] = False,
    verbose: bool = True,
    store: Union[bool, str] = False,
    store_index: Union[bool, str] = True,
) -> Union[pd.DataFrame, ResoKitDataset]:
    """Load the dataset from a specified source.

    The dataset is loaded from a ZIP archive or a CSV file, or from memory
    if already stored. The priority is given to the memory saved dataset,
    then to the zip archive, and finally to the file.

    Note
    ----
    Storing the dataset in memory is useful for faster access and to avoid
    reading the file multiple times.

    Note
    ----
    If both `from_file` and `from_zip` are provided, it is assumed that the
    file inside the ZIP archive is the same as the one provided in `from_file`.
    Finally, the path constructed is: `dir_path / zip_name / file_name`.

    Parameters
    ----------
    source : str
        Identifier for the data source ('eu' or 'nasa').
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
    to_resokit : bool, optional. Default: True.
        If `True`, returns the dataset including only the columns
        required by ResoKit.
    to_df : bool, optional. Default: False.
        If `True`, returns the raw dataset as a pandas DataFrame.
        If `False`, returns the dataset as a ResoKitDataset.
    check_age : bool, optional. Default: False.
        If `True`, displays the file's last modified date.
        used by ResoKit.
    only_index : bool, optional. Default: False.
        If `True`, loads only the index columns.
        If `p` or a string starting with "p", loads the parsed index
        columns. Only compatible with `from_memory=True`. If not previously
        stored, `None` is returned.
    only_rows : list|int, optional. Default: [].
        If provided, loads only the specified rows.
        Remember that python is 0-indexed, so
        the first row (system) is 0.
    verbose : bool, optional. Default: True.
        If `True`, prints messages about the process.
    store : bool, str, optional. Default: False.
        If `str`, then "f" or "y" or "s" or "o" overwrites the stored dataset.
        If `True`, stores the dataset in memory.
    store_index : bool, str, optional. Default: True.
        If `True`, stores the dataset index in memory.
        If `only_rows` is provided, the index is not stored.
        If `str`, then "f" or "y" or "s" or "o" overwrites the stored index.

    Returns
    -------
    dataset : DataFrame or ResoKitDataset
        The loaded dataset as a pandas DataFrame or a ResoKitDataset.
    """
    source = source.lower()  # Ensure lowercase

    if source not in _DATASET_FILENAMES:  # Check if source is valid
        raise ValueError(f"Invalid source: {source}. Must be 'eu' or 'nasa'.")

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
            from_file = _DATASET_FILENAMES[source]

    # Check store_index
    if store and only_index:
        store_index = True

    # Define origin and age
    origin = []
    age = -1

    # Define overwrite
    overwrite = False
    if (
        isinstance(store, str) and store.lower()[0] in ["o", "f", "y", "s"]
    ) or (
        isinstance(store_index, str)
        and store_index.lower()[0] in ["o", "f", "y", "s"]
    ):
        overwrite = True

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
        if from_memory:
            data_stored, not_stored_rows, xage, xorigin = _load_stored_rows(
                source,
                rows=requested_rows,
                full=False,
            )
            # Update origin
            if not data_stored.empty:
                age = max(age, xage)
                origin.append(xorigin)

        else:
            # If not from memory, set data_stored to empty
            data_stored = pd.DataFrame()
            not_stored_rows = requested_rows

        # Define stored_rows and not_stored
        stored_rows = list(data_stored.index)

        # Message
        if verbose and not data_stored.empty:
            print(
                f" Loaded rows {stored_rows} "
                + f"from {source} memory stored dataset..."
            )

        # Check if all rows are stored
        if len(data_stored) == len(requested_rows):
            # Check if the dataset is fully stored (and loaded)
            is_full = (
                len(data_stored) == len(_IN_MEMORY_DATASETS[source])
            ) and _IS_FULLY_STORED[source]

            # No need to load the dataset or store the rows
            # (because they are already stored)

            return __aux_load_full(
                df=data_stored,
                source=source,
                age=age,
                origin=origin[0],
                is_full=is_full,
                to_resokit=to_resokit,
                to_df=to_df,
                metadata=dict(_IN_MEMORY_DATASETS[source].metadata),
            )

        elif (not from_zip) and (not from_file):  # If no file or ZIP provided
            raise ValueError(
                "Some rows are not stored and no file or ZIP provided."
            )

        # Add header and update only_rows
        only_rows = [0] + [
            x + 1 for x in requested_rows if x in not_stored_rows
        ]

        def skip_rows(x: int) -> bool:  # Skip rows not in the list
            return x not in only_rows

    elif only_rows:  # If only_rows is True...
        raise ValueError("only_rows must be a list or an integer.")

    else:  # If not only_rows...
        skip_rows = None
        only_rows = False

    # Check if the index columns are already stored in memory
    if only_index and from_memory:
        # Check if parsed requested
        if isinstance(only_index, str) and only_index.lower()[0] == "p":
            data = _IN_MEMORY_PARSED_INDEXES[source]
            if data is None and verbose:
                print(" Parsed index not stored.")
            elif data is not None and verbose:
                print(
                    " Loaded parsed index columns from memory stored dataset."
                )
            return data
        data = _load_stored_index(source, to_df=to_df, to_resokit=to_resokit)
        if not data.empty:
            if verbose:
                print(" Loaded index columns from memory stored datasets.")
            return data

    # Check if the dataset is already stored in memory
    if (
        not (only_index or only_rows)
        and _IS_FULLY_STORED[source]
        and from_memory
    ):
        data = _load_stored_full(
            source, to_df=to_df, to_resokit=to_resokit, sort=True
        )
        if verbose:
            print(" Loaded full dataset from memory stored datasets.")
        if check_age:
            print(f"Last modified: {data.age} days ago")
        return data

    # Define columns to load
    usecols = _INDEX_COLUMNS[source] if only_index else None

    # Aux message
    if verbose:  # Print message if verbose
        if only_index:
            print(" Loading only index columns...")
        elif only_rows:
            print(f" Loading rows {not_stored_rows}...")
        else:
            print(" Loading the entire dataset...")

    # Load the dataset from the ZIP archive
    if from_zip:
        zip_path = dir_path / from_zip
        file_name = from_file if from_file else _DATASET_FILENAMES[source]
        data = load_from_zip(
            zip_path=zip_path,
            file_name=file_name,
            source=source,
            skip_rows=skip_rows,
            usecols=usecols,
            verbose=verbose,
        )
        age = _check_file_age(
            file_path=from_file,
            zip_path=zip_path,
            verbose=check_age,
        )
        origin.append("zip")
    # Load the dataset from the file
    elif from_file:
        file_path = dir_path / from_file
        data = pd.read_csv(
            file_path,
            header=0,
            skiprows=skip_rows,
            usecols=usecols,
            dtype=DATASET_DTYPES[source],
        )
        age = _check_file_age(
            file_path=from_file,
            zip_path=None,
            verbose=check_age,
        )
        origin.append("file")
    else:
        raise ValueError(
            "Data not found in memory, and no file or ZIP provided."
        )

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

    # Define origin
    origin = "mixed" if len(set(origin)) > 1 else origin[0]

    # Define is_full
    is_full = not only_rows

    # Define index_only
    index_only = not store

    # Check storing
    if store_index or store:  # If not storing, return the data
        _update_stored_dataset(
            data,
            source,
            age=age,
            origin=origin,
            is_full=is_full,
            verbose=verbose,
            index_only=index_only,
            sort=True,
            overwrite=overwrite,
        )

    return __aux_load_full(
        df=data,
        source=source,
        age=age,
        origin=origin,
        is_full=is_full,
        to_resokit=to_resokit,
        to_df=to_df,
    )


# -------------------------- BINARY SYSTEMS -----------------------------------


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
    circumbinary: bool,
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
            The dataset as a pandas DataFrame.
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
            # Clean if requested
            if clean:
                df.loc[df[7] > 98, 7] = pd.NA
                df.loc[df[13] > 998, 13] = pd.NA
            # Rename columns if requested
            if rename_columns:
                df.columns = _BINARIES_COLUMNS
            return df

    # Load the dataset from the ZIP archive
    if from_zip:
        if verbose:
            print(
                f"Loading the type-{letter} dataset "
                + f"from ZIP archive {from_zip}."
            )
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
            print(f"Loading the type-{letter} dataset from file {from_file}")
        file_path = dir_path / from_file
        with open(file_path, "r") as f:
            lines = f.readlines()

    # Extract header and data from lines
    header, data = _extract_header_and_data(
        lines=lines, circumbinary=circumbinary, inferr=inferr
    )

    # Store the data and header in memory
    _IN_MEMORY_BINARIES_HEADERS[letter] = str(header)
    _IN_MEMORY_BINARIES[letter] = data.copy(deep=True)
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


# -------------------------- MEMORY MANAGEMENT --------------------------------


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
        for key in _IN_MEMORY_DATASETS:
            # Clear the memory addresses
            _IN_MEMORY_INDEXES[key] = _mk_empty_dataset(key)
            _IN_MEMORY_DATASETS[key] = _mk_empty_dataset(key)
            _IS_FULLY_STORED[key] = False
            _IN_MEMORY_PARSED_INDEXES[source] = None
            if verbose:
                print(f" Cleared memory for source: {key}")
        return

    if source not in _IN_MEMORY_DATASETS:
        raise ValueError(f"Invalid source: {source}. Must be 'eu' or 'nasa'.")

    # Clear the memory addresses
    _IN_MEMORY_INDEXES[source] = _mk_empty_dataset(source)
    _IN_MEMORY_DATASETS[source] = _mk_empty_dataset(source)
    _IS_FULLY_STORED[source] = False  # Reset the fully stored flag
    _IN_MEMORY_PARSED_INDEXES[source] = None

    if verbose:
        print(f" Cleared memory for source: {source}")

    return
