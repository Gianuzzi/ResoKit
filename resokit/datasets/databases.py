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

"""Module to manage provided exoplanet datasets from exoplanet.eu and NASA."""

# =============================================================================
# IMPORTS
# =============================================================================

import datetime
import warnings
from io import BytesIO, StringIO, TextIOWrapper
from pathlib import Path
from typing import BinaryIO, List, Tuple, Union
from zipfile import ZIP_DEFLATED, ZipFile

import attrs

import pandas as pd

from resokit.core import MetaData, ResokitDataFrame, df_to_resokit
from resokit.datasets.utils import (
    BINARIES_COLUMNS,
    BINARIES_FILENAMES,
    BINARIES_URLS,
    DATASETS_DIR,
    DATASET_DTYPES,
    DATASET_FILENAMES,
    DATASET_URLS,
    DATASET_ZIPNAMES,
    INDEX_COLUMNS,
    check_file_age,
    check_online_binary,
    check_online_dataset,
    load_from_zip,
    remove_from_zip,
    request_dataset,
)
from resokit.utils.parser import DEFAULT_METADATA, parse_name, parse_to_iter

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
        if isinstance(key, ResoKitDataset):
            # Attempt to get a slice from sliced
            sliced = self.dataset.__getitem__(key.dataset)
        else:
            sliced = self.dataset.__getitem__(key)
        is_full = self.is_full and len(sliced) == len(self.dataset)
        # Transform to df if possible
        if isinstance(sliced, pd.Series):
            sliced = sliced.to_frame()
        return attrs.evolve(self, dataset=sliced, is_full=is_full)

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

    def __eq__(self, value):
        """X == Y <==> X.__eq__(Y)."""
        if isinstance(value, ResoKitDataset):
            return (
                self.dataset.equals(value.dataset)
                and self.source == value.source
                and self.age == value.age
                and self.origin == value.origin
                and self.is_full == value.is_full
                and self.metadata == value.metadata
            )
        elif isinstance(value, pd.DataFrame):
            return self.dataset.equals(value)
        elif isinstance(value, (str, int, float)):
            return self.dataset == value
        return False

    def __and__(self, other):
        """X & Y <==> X.__and__(Y)."""
        if isinstance(other, ResoKitDataset):
            return attrs.evolve(
                self,
                dataset=self.dataset.__and__(other.dataset),
                is_full=self.is_full and other.is_full,
            )
        return attrs.evolve(self, dataset=self.dataset.__and__(other))

    def __or__(self, other):
        """X | Y <==> X.__or__(Y)."""
        if isinstance(other, ResoKitDataset):
            return attrs.evolve(
                self,
                dataset=self.dataset.__or__(other.dataset),
                is_full=self.is_full and other.is_full,
            )
        return attrs.evolve(self, dataset=self.dataset.__or__(other))

    def to_dataframe(
        self,
        columns: Union[list, None] = None,
        copy: bool = True,
        sort: bool = False,
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
        return attrs.evolve(self, dataset=self.dataset.copy())

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
            rename_index=False,
            metadata=None,
        )

        return attrs.evolve(self, dataset=df)

    def to_file(
        self,
        path_or_buf: Union[str, Path, BinaryIO, TextIOWrapper],
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        """Save the dataset to a file.

        This method saves the dataset to a file in CSV format.

        Parameters
        ----------
        path_or_buf : str or Path or BinaryIO or TextIOWrapper
            File path or buffer to save the dataset.
        overwrite : bool, optional. Default: False.
            Whether to overwrite the file if it already exists.
        verbose : bool, optional. Default: True.
            Whether to print informational messages.
        """
        if not isinstance(path_or_buf, (BinaryIO, TextIOWrapper)):
            file_path = Path(path_or_buf)
            if file_path.exists() and not overwrite:
                raise FileExistsError(
                    f"File {file_path} already exists.\n"
                    + "  Set overwrite=True to force the save."
                )
        else:
            # If a buffer is provided, we assume it is a writable f-like object
            file_path = "provided buffer"

        # Save the dataset to a file
        if not overwrite:
            self.dataset.to_csv(path_or_buf, mode="x")
        else:
            self.dataset.to_csv(path_or_buf)

        if verbose:
            print(f"Dataset saved to {file_path}.")
        return


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


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


def _df_to_dataset(
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
        my_df = df_to_resokit(
            df,
            source=source,
            drop=True,
            copy=copy,
            sort_by=False,
            return_df=True,
            rename_index=False,
            metadata=None,
        )
    else:
        my_df = df.copy(deep=copy)

    assert isinstance(my_df, pd.DataFrame), (
        "Expected df to be a DataFrame, " + f"got {type(my_df)} instead."
    )

    if metadata is None:
        metadata = dict(DEFAULT_METADATA)

    return ResoKitDataset(
        dataset=my_df,
        source=source,
        age=age,
        origin=origin,
        is_full=is_full,
        metadata=metadata,
    )


# =============================================================================
# MANAGERS
# =============================================================================


class DatasetManager:
    """Manager for the ResoKit datasets.

    This class manages the datasets in memory and disk, allowing to load,
    update, and check if they are outdated. It also provides methods to
    download and store the datasets.
    """

    def __init__(self):
        # ---------------------- EU and NASA DATASETS ----------------------
        self._datasets = {
            "eu": _mk_empty_dataset("eu"),
            "nasa": _mk_empty_dataset("nasa"),
        }

        self._indexes = {
            "eu": _mk_empty_dataset("eu"),
            "nasa": _mk_empty_dataset("nasa"),
        }

        self._parsed_indexes = {"eu": None, "nasa": None}

        self._is_fully_stored = {"eu": False, "nasa": False}

    # ------------------------------------------------------------------------
    # Functions to manage datasets in memory
    # ------------------------------------------------------------------------

    def update(
        self,
        new_df: pd.DataFrame,
        source: str,
        age: int,
        origin: str,
        is_full: bool,
        verbose: bool = True,
        index_only: bool = False,
        sort: bool = True,
        metadata: Union[dict, None] = None,
        overwrite: bool = False,
    ) -> None:
        """Update the stored dataset in memory."""
        if self._is_fully_stored[source] and not overwrite:
            return

        if new_df.empty:
            if verbose:
                print(" No rows to store in memory.")
            return

        if is_full and (
            self._indexes[source].dataset.empty
            or (not self._indexes[source].dataset.empty and overwrite)
        ):
            new_index = new_df[INDEX_COLUMNS[source]].copy()

            self._indexes[source] = _df_to_dataset(
                new_index,
                source=source,
                age=age,
                origin=origin,
                is_full=is_full,
                metadata=metadata,
                copy=True,
                as_resokit=False,
            )
            parsed = new_index.astype(str)
            parsed[INDEX_COLUMNS[source][0]] = parsed[
                INDEX_COLUMNS[source][0]
            ].apply(parse_name, force=True)
            parsed[INDEX_COLUMNS[source][1]] = parsed[
                INDEX_COLUMNS[source][1]
            ].apply(parse_name, force=True)
            self._parsed_indexes[source] = parsed

            if verbose:
                print(" Updated stored index in memory.")

        if index_only:
            return

        if is_full or self._datasets[source].dataset.empty:
            self._datasets[source] = _df_to_dataset(
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
                self._is_fully_stored[source] = True
                if verbose:
                    print(" Stored dataset in memory.")
            else:
                new_to_store = new_df.index.to_list()
                if verbose:
                    print(f" Stored rows {new_to_store} in memory...")
            return

        new_to_store = [
            x
            for x in new_df.index
            if x not in self._datasets[source].dataset.index
        ]

        # Pre-define values
        updated_df = new_df
        meta_old = dict(self._datasets[source].metadata)
        age_old = self._datasets[source].age
        origin_old = self._datasets[source].origin

        if not new_to_store and not overwrite:
            return

        elif new_to_store and not overwrite:
            new_df = new_df.loc[new_to_store]
            updated_df = pd.concat([self._datasets[source].dataset, new_df])
            # age_old = self._datasets[source].age
            # origin_old = self._datasets[source].origin
            # meta_old = dict(self._datasets[source].metadata)

        elif overwrite:
            repeated = [
                x
                for x in new_df.index
                if x in self._datasets[source].dataset.index
            ]
            if not repeated:
                return
            elif len(repeated) == len(self._datasets[source].dataset):
                # updated_df = new_df
                age_old = age
                origin_old = origin
                meta_old = metadata
            else:
                to_keep = [
                    x
                    for x in self._datasets[source].dataset.index
                    if x not in new_df.index
                ]
                keep_df = self._datasets[source].dataset.loc[to_keep]
                updated_df = pd.concat([keep_df, new_df])
                # age_old = self._datasets[source].age
                # origin_old = self._datasets[source].origin
                # meta_old = dict(self._datasets[source].metadata)

            new_to_store = new_df.index.to_list()

        assert isinstance(meta_old, (dict, MetaData)), (
            "Expected metadata to be a dictionary, "
            + f"got {type(meta_old)} instead."
        )

        if sort:
            updated_df.sort_index(inplace=True)

        if metadata is not None:
            meta_old.update(metadata)

        self._datasets[source] = _df_to_dataset(
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

    def download(
        self,
        source: str,
        to_memory: bool = True,
        to_file: Union[str, Path, bool, None] = False,
        to_zip: Union[str, Path, bool, None] = True,
        dir_path: Union[str, Path, None] = None,
        overwrite: bool = False,
        check_online: bool = True,
        to_resokit: Union[bool, None] = None,
        verbose: bool = True,
        chunk_size: int = 1024,
        print_size: float = 0.15,
    ) -> Union[Path, pd.DataFrame, ResoKitDataset, None]:
        """Download a dataset from the internet."""
        source = source.lower()
        if source not in DATASET_FILENAMES:
            raise ValueError(
                f"Invalid source: {source}. Must be 'eu' or 'nasa'."
            )

        if not to_file and not to_zip and not to_memory and to_resokit is None:
            raise ValueError(
                "Nothing to do. Set at least one of to_file, to_zip, "
                + "to_memory, or to_resokit."
            )

        url = DATASET_URLS[source]

        if dir_path is None:
            dir_path = DATASETS_DIR
        else:
            dir_path = Path(dir_path)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory {dir_path} not found.")

        if to_file:
            file_name = (
                DATASET_FILENAMES[source] if to_file is True else to_file
            )
            file_path = dir_path / file_name
            if file_path.exists() and not overwrite:
                raise FileExistsError(
                    f"File {file_path} already exists. "
                    + "Set overwrite=True to force the download."
                )
        else:
            file_path = None
            file_name = DATASET_FILENAMES[source]

        if to_zip:
            zip_name = DATASET_ZIPNAMES[source] if to_zip is True else to_zip
            zip_path = dir_path / zip_name
            if zip_path.exists() and not overwrite:
                raise FileExistsError(
                    f"ZIP archive {zip_path} already exists. "
                    + " Set overwrite=True to force the download."
                )
        else:
            zip_path = None
            zip_name = DATASET_ZIPNAMES[source]

        if (
            self._is_fully_stored[source]
            and not overwrite
            and not to_file
            and not to_zip
        ):
            if verbose:
                print(
                    "Dataset is already fully stored. "
                    + "Set overwrite=True to force the download."
                )
            if to_resokit is not None:
                return (
                    self._datasets[source]
                    if to_resokit
                    else self._datasets[source].to_dataframe()
                )
            return None

        if check_online:
            outdated = check_outdated(source, verbose=verbose)
            if not outdated:
                if (to_file or to_zip) and verbose:
                    print(
                        "To store the dataset in a file, load it and "
                        + "invoke the `to_file` method."
                    )
                elif verbose:
                    print(
                        "No need to download the dataset. "
                        + "Set check_online=False to really force it."
                    )
                if to_resokit is not None:
                    df = self.load(source, verbose=False, to_df=not to_resokit)
                    assert isinstance(df, (pd.DataFrame, ResoKitDataset)), (
                        "Expected df to be a pandas DataFrame or "
                        + f"ResoKitDataset, got {type(df)} instead."
                    )
                    return df
                return None

        data = request_dataset(
            url, verbose=verbose, chunk_size=chunk_size, print_size=print_size
        )
        if not data or len(data) == 0:
            raise ValueError(f"Empty dataset downloaded from {url}.")
        elif verbose:
            print(f" Data downloaded successfully. ({len(data)/1e6:.2f} MB)")

        if zip_path is not None:
            file_name = str(file_name)
            if not zip_path.exists() and verbose:
                print(f" Creating the ZIP archive {zip_path}...")
            else:
                remove_from_zip(str(zip_path), file_name, verbose=verbose)
            with ZipFile(zip_path, "a", compression=ZIP_DEFLATED) as zipf:
                zipf.writestr(file_name, data)
            if verbose:
                print(f" Written {file_name} to {zip_path}.")

        if file_path is not None:
            if not file_path.exists() and verbose:
                print(f" Creating the file {file_path}...")
            with open(file_path, "wb") as f:
                f.write(data)
            if verbose:
                print(f" Written {file_path}.")

        # Set default df
        df = _mk_empty_dataset(source).dataset

        if to_memory or to_resokit is not None:
            df = pd.read_csv(BytesIO(data), dtype=DATASET_DTYPES[source])

        metadata = dict(DEFAULT_METADATA)
        metadata.update(
            {
                "downloaded": datetime.datetime.now().isoformat(),
                "url": url,
            }
        )

        if to_memory:
            self.update(
                df,
                source=source,
                age=0,
                origin="internet",
                is_full=True,
                verbose=verbose,
                index_only=False,
                sort=True,
                metadata=metadata,
                overwrite=True,
            )

        if to_resokit is not None:
            return _df_to_dataset(
                df,
                source=source,
                age=0,
                origin="internet",
                is_full=True,
                copy=True,
                as_resokit=to_resokit,
            )

        if file_path and zip_path:
            return file_path, zip_path
        if file_path:
            return file_path
        if zip_path:
            return zip_path

        return None

    def load_full(
        self,
        source: str,
        to_resokit: bool = True,
        sort: bool = True,
    ) -> ResoKitDataset:
        """Load the full dataset from memory or disk."""
        if not self._is_fully_stored[source]:
            raise ValueError(f"Source {source} is not fully stored.")

        ds = self._datasets[source]
        if sort:
            sorted_df = ds.dataset.sort_index()
            return _df_to_dataset(
                sorted_df,
                source=source,
                age=ds.age,
                origin=ds.origin,
                is_full=ds.is_full,
                metadata=dict(ds.metadata),
                copy=False,
                as_resokit=to_resokit,
            )
        return ds.to_resokit() if to_resokit else ds.copy()

    def load_rows(
        self,
        source: str,
        rows: Union[list, None] = None,
        full: bool = False,
    ) -> Union[Tuple[pd.DataFrame, list, int, str], ResoKitDataset]:
        """Load specific rows from the dataset."""
        if full:
            return self.load_full(source, to_resokit=True, sort=True)

        if rows is not None:
            stored = [
                x for x in rows if x in self._datasets[source].dataset.index
            ]
            not_stored = [x for x in rows if x not in stored]
            df = self._datasets[source].dataset.loc[stored].copy()
            age = self._datasets[source].age
            origin = self._datasets[source].origin
            return df, not_stored, age, origin

        raise ValueError("No rows provided.")

    def load_index(
        self,
        source: str,
        to_df: bool = False,
        to_resokit: bool = True,
        parsed: bool = False,
    ) -> Union[pd.DataFrame, ResoKitDataset, None]:
        """Load the index of a given source dataset."""
        if parsed:
            return self._parsed_indexes[source]

        index_ds = self._indexes[source]
        if index_ds.dataset.empty:
            return index_ds
        if not to_df:
            return index_ds.to_resokit() if to_resokit else index_ds
        return index_ds.to_dataframe()

    @staticmethod
    def _aux_load_full(
        df: pd.DataFrame,
        source: str,
        age: int,
        origin: str,
        is_full: bool,
        to_resokit: bool,
        to_df: bool,
        metadata: Union[dict, None] = None,
    ) -> Union[pd.DataFrame, ResokitDataFrame, ResoKitDataset]:
        """Auxiliary function to load a full dataset."""
        if not to_df:
            return _df_to_dataset(
                df,
                source=source,
                age=age,
                origin=origin,
                is_full=is_full,
                metadata=metadata,
                copy=False,
                as_resokit=to_resokit,
            )
        if to_resokit:
            return df_to_resokit(
                df,
                source=source,
                drop=True,
                copy=False,
                sort_by=False,
                metadata=metadata,
                return_df=True,
            )
        return df

    def load(
        self,
        source: str,
        from_memory: bool = True,
        from_zip: Union[str, bool] = True,
        from_file: Union[str, bool] = False,
        dir_path: Union[str, Path, bool] = True,
        to_resokit: bool = True,
        to_df: bool = False,
        check_age: bool = False,
        only_index: bool = False,
        only_rows: Union[list, int] = False,
        verbose: bool = True,
        store: Union[bool, str] = False,
        store_index: Union[bool, str] = True,
    ) -> Union[pd.DataFrame, ResokitDataFrame, ResoKitDataset, None]:
        """Load a dataset from memory, ZIP, or file."""
        source = source.lower()
        if source not in DATASET_FILENAMES:
            raise ValueError(
                f"Invalid source: {source}. Must be 'eu' or 'nasa'."
            )

        # Check if something to do
        if not from_memory and not from_zip and not from_file:
            raise ValueError(
                "Nothing to do. Set at least one of "
                + "from_memory, from_zip, or from_file."
            )

        # Redefine dir path
        if dir_path is None or dir_path is True:
            # Default directory
            dir_path = DATASETS_DIR
        elif not dir_path:
            # Assuming no file or ZIP required
            from_zip = False
            from_file = False
        else:
            # Convert to Path
            dir_path = Path(dir_path)

        # Flag for default path
        default = False

        # Define paths and ZIP extraction flag
        if from_zip is None or from_zip is True:
            from_zip = DATASET_ZIPNAMES[source]
            default = True  # Default ZIP

        # Define file name
        if from_file is None or from_file is True:
            from_file = DATASET_FILENAMES[source]
            default = True  # Default file

        # Check store_index
        if store and only_index:
            store_index = True

        # Define initials: origin, age, ...
        origin = []
        age = -1
        not_stored_rows = []
        stored_rows = []
        requested_rows = []
        data_stored = pd.DataFrame()

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
        ) or only_rows:  # If only_rows is provided, set up the skip_rows func

            if isinstance(only_rows, bool):
                raise ValueError("only_rows must be a list or an integer.")

            iter_only_rows = parse_to_iter(only_rows)  # Convert to iterable

            # Remove duplicates
            seen = set()
            seen_add = seen.add
            requested_rows = [
                x for x in iter_only_rows if not (x in seen or seen_add(x))
            ]

            # Check no negative values
            if any(x < 0 for x in requested_rows):
                raise ValueError("only_rows must be positive integers.")

            # Load stored rows if available
            if from_memory:
                data_stored, not_stored_rows, xage, xorigin = self.load_rows(
                    source,
                    rows=requested_rows,
                    full=False,
                )
                assert isinstance(data_stored, pd.DataFrame), (
                    "Expected data to be a pandas DataFrame, "
                    + f"got {type(data_stored)} instead."
                )
                assert isinstance(xage, int), (
                    "Expected age to be an integer, "
                    + f"got {type(xage)} instead."
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
                    len(data_stored) == len(self._datasets[source])
                ) and self._is_fully_stored[source]

                # No need to load the dataset or store the rows
                # (because they are already stored)
                return self._aux_load_full(
                    df=data_stored,
                    source=source,
                    age=age,
                    origin=origin[0],
                    is_full=is_full,
                    to_resokit=to_resokit,
                    to_df=to_df,
                    metadata=dict(self._datasets[source].metadata),
                )

            elif (not from_zip) and (
                not from_file
            ):  # If no file or ZIP provided
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
            parsed = (
                isinstance(only_index, str) and only_index.lower()[0] == "p"
            )
            data = self.load_index(
                source, to_df=False, to_resokit=to_resokit, parsed=parsed
            )
            if data is None:
                extra = "parsed " if parsed else ""
                if verbose:
                    print(
                        f" No {extra}index columns stored "
                        + f"in memory for {source}."
                    )
                return None
            elif parsed:
                if verbose:
                    print(
                        " Loaded parsed index columns from "
                        + "memory stored datasets."
                    )
                return data
            assert isinstance(data, ResoKitDataset), (
                "Expected data to be a ResoKitDataset, "
                + f"got {type(data)} instead."
            )
            if check_age and int(data.age) >= 0:
                print(f" Last modified: {data.age} days ago.")
            if to_df:
                data = data.to_dataframe()
            if not data.empty:
                if verbose:
                    print(
                        " Loaded index columns from "
                        + "memory stored datasets."
                    )
                return data

        # Check if the dataset is already stored in memory
        if (
            not (
                only_index or only_rows
            )  # Check if loading the entire dataset
            and self._is_fully_stored[source]  # Check if fully stored
            and from_memory  # Check if loading from memory
        ):
            data = self.load_full(source, to_resokit=to_resokit, sort=True)
            if verbose:
                print(" Loaded full dataset from memory stored datasets.")
            if check_age and data.age >= 0:
                print(f" Last modified: {data.age} days ago.")
            # Check if to df
            if to_df:
                return data.to_dataframe()
            return data

        # Define columns to load
        usecols = INDEX_COLUMNS[source] if only_index else None

        # Aux message
        if verbose:  # Print message if verbose
            if only_index:
                print(" Loading only index columns...")
            elif only_rows:
                print(f" Loading rows {not_stored_rows}...")
            else:
                print(" Loading the entire dataset...")

        # Check dir_path
        assert isinstance(dir_path, Path), (
            "Expected dir_path to be a Path, "
            + f"got {type(dir_path)} instead."
        )

        # Load the dataset from the ZIP archive
        if from_zip:
            zip_path = dir_path / Path(from_zip)
            file_name = from_file if from_file else DATASET_FILENAMES[source]
            try:
                data = load_from_zip(
                    zip_path=zip_path,
                    file_name=file_name,
                    source=source,
                    skip_rows=skip_rows,
                    usecols=usecols,
                    verbose=verbose,
                )
            except FileNotFoundError as err:
                msg = ""
                # Check if it is the default path
                if default:
                    msg = (
                        "\n Try running "
                        + f"`resokit.datasets.download_dataset('{source}',"
                        + "to_zip=True)` first to download the dataset."
                    )
                raise FileNotFoundError(err.args[0] + msg)
            age = check_file_age(
                file_path=file_name,
                zip_path=zip_path,
                verbose=check_age,
            )
            origin.append("zip")

        # Load the dataset from the file
        elif from_file:
            file_path = dir_path / Path(from_file)
            try:
                data = pd.read_csv(
                    file_path,
                    header=0,
                    skiprows=skip_rows,
                    usecols=usecols,
                    dtype=DATASET_DTYPES[source],
                )
            except FileNotFoundError:
                msg = ""
                if default:
                    msg = (
                        "\n Try running "
                        + f"`resokit.datasets.download_dataset('{source}',"
                        + " to_file=True)` first to download the dataset."
                    )
                raise FileNotFoundError(
                    f"File {file_path} not found at {dir_path}." + msg
                )
            age = check_file_age(
                file_path=file_path,
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
            assert isinstance(not_stored_rows, list), (
                "Expected not_stored_rows to be a list, "
                + f"got {type(not_stored_rows)} instead."
            )

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
            new_index = [
                x for x in requested_rows if x in used_rows + stored_rows
            ]

            data = data.reindex(new_index, copy=False)

        # Define origin
        origin = "mixed" if len(set(origin)) > 1 else origin[0]

        # Define is_full
        is_full = not only_rows

        # Define index_only
        index_only = bool(only_index or (store_index and not store))

        # Check storing
        if store_index or store:
            self.update(
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

        return self._aux_load_full(
            df=data,
            source=source,
            age=age,
            origin=origin,
            is_full=is_full,
            to_resokit=to_resokit,
            to_df=to_df,
        )

    def clear_memory(
        self, source: str, verbose: bool = True, files: bool = False
    ):
        """Clear the memory for the specified dataset."""
        source = source.lower()
        if files:
            if source in ["eu", "nasa"]:
                for file in DATASET_FILENAMES[source]:
                    file_path = DATASETS_DIR / file
                    if file_path.exists():
                        file_path.unlink()
                        if verbose:
                            print(f" Removed {file_path} from disk.")
            elif source == "both":
                for key in ["eu", "nasa"]:
                    self.clear_memory(key, verbose, files=True)
            else:
                raise ValueError("Invalid EU/NASA source.")
        else:
            if source in self._datasets:
                self._indexes[source] = _mk_empty_dataset(source)
                self._datasets[source] = _mk_empty_dataset(source)
                self._is_fully_stored[source] = False
                self._parsed_indexes[source] = None
                if verbose:
                    print(f" Cleared memory for source: {source}")
            elif source == "both":
                for key in self._datasets:
                    self.clear_memory(key, verbose=verbose)
            else:
                raise ValueError("Invalid EU/NASA source.")


class BinaryDatasetManager:
    """Manager for the ResoKit binaries datasets.

    This class manages the binaries datasets in memory and disk, allowing to
    load, update, and check if they are outdated. It also provides methods to
    download and store the datasets.
    """

    def __init__(self):
        # -------------------- BINARY SYSTEMS DATASETS ----------------------
        self._data = {"s": pd.DataFrame(), "p": pd.DataFrame()}
        self._headers = {"s": "", "p": ""}

    @staticmethod
    def _extract_header_and_data(
        lines: List[str], circumbinary: bool, inferr: bool
    ) -> Tuple[str, pd.DataFrame]:
        """Extract header and data from lines of the dataset.

        Parameters
        ----------
        lines : List[str]
            Lines of the dataset.
        circumbinary : bool
            Whether the dataset is circumbinary.
        inferr : bool
            Whether the width of the columns is inferred.
            If False, the width of the columns is fixed.

        Returns
        -------
        Tuple[str, pd.DataFrame]
            header : str
                The header of the dataset.
            data : pd.DataFrame
                The dataset as a pandas DataFrame.
        """
        # Find the index of the last line that starts with "Note:"
        # (or any number of hyphens)
        separator_index = len(lines)
        for i, line in enumerate(reversed(lines)):
            stripped = line.strip()
            if stripped.startswith("Note:") or stripped.startswith("-"):
                separator_index = len(lines) - i
                break

        # Check if the separator was found
        if separator_index == len(lines):
            raise ValueError("Separator line not found.")

        # The header is everything before the separator line
        header = "".join(lines[:separator_index]).strip()

        # The data starts after the last "Note:" line, so we extract the data
        data_lines = [
            line.replace("\t", " ") for line in lines[separator_index:]
        ]

        # Define widths for fixed-width formatted data
        kwargs = {}
        if inferr:
            kwargs["colspecs"] = "infer"
        elif circumbinary:
            kwargs["widths"] = [15, 10, 6, 6, 8, 2, 7, 7, 2, 10, 6, 9, 7, 8]
        else:
            kwargs["widths"] = [15, 10, 6, 6, 8, 2, 8, 7, 2, 8, 6, 9, 7, 8]

        # Use pandas to read the fixed-width formatted data
        # starting after the header
        data = pd.read_fwf(
            StringIO("".join(data_lines)), header=None, **kwargs
        )

        return header, data

    def load(
        self,
        source: str,
        from_memory: bool = True,
        from_file: Union[str, bool] = True,
        dir_path: Union[str, Path, bool] = True,
        rename_columns: bool = True,
        ret_header: bool = False,
        inferr: bool = False,
        clean: bool = True,
        verbose: bool = True,
    ) -> Union[pd.DataFrame, str]:
        """Read the provided multi-star system dataset."""
        # Check the source
        source = source.lower()
        if source in ["circumbinary", "c", "p"]:
            circumbinary = True
        elif source in ["simple", "s"]:
            circumbinary = False
        else:
            raise ValueError(
                "Invalid source. "
                + "Must be 'circumbinary', 'c', 'p', 'simple', or 's'."
            )

        # Define the filename based on the circumbinary parameter
        letter = "p" if circumbinary else "s"

        # Check if something to do
        if not from_memory and not from_file:
            raise ValueError(
                "Nothing to do. Set at least one of "
                + "from_memory, or from_file."
            )

        # Redefine dir path
        if dir_path is None or dir_path is True:
            dir_path = DATASETS_DIR
        elif not dir_path:
            # Assuming no file required
            from_file = False
        elif dir_path:
            # Convert to Path
            dir_path = Path(dir_path)

        # Define file name
        if from_file is None or from_file is True:
            from_file = BINARIES_FILENAMES[letter]

        # Default lines
        lines = []

        # Load the dataset from memory
        if from_memory:
            if ret_header and self._headers[letter] != "":
                if verbose:
                    print(f"Loading the type-{letter} header from memory.")
                return str(self._headers[letter])  # Return a copy
            elif not self._data[letter].empty:
                if verbose:
                    print(f"Loading the type-{letter} dataset from memory.")
                df = self._data[letter].copy()
                # Clean if requested
                if clean:
                    df.loc[df[7] > 98, 7] = pd.NA  # eccentricity
                    df.loc[df[13] > 998, 13] = pd.NA  # imutual
                # Rename columns if requested
                if rename_columns:
                    df.columns = BINARIES_COLUMNS
                return df

        # Load the dataset from the file
        if from_file:
            if verbose:
                print(
                    f"Loading the type-{letter} dataset from file {from_file}"
                )
            assert isinstance(dir_path, Path), (
                "Expected dir_path to be a Path, "
                + f"got {type(dir_path)} instead."
            )
            file_path = dir_path / Path(from_file)
            with open(file_path, "r") as f:
                lines = f.readlines()

        # Extract header and data from lines
        header, data = self._extract_header_and_data(
            lines=lines, circumbinary=circumbinary, inferr=inferr
        )

        # Store the data and header in memory
        self._headers[letter] = str(header)
        self._data[letter] = data.copy(deep=True)
        if verbose:
            print(f"Stored the type-{letter} dataset and header into memory.")

        # Clean data
        if clean:
            data.loc[data[7] > 98, 7] = pd.NA  # eccentricity
            data.loc[data[13] > 998, 13] = pd.NA  # imutual

        # Rename columns
        if rename_columns:
            data.columns = BINARIES_COLUMNS

        # Return the header if requested
        if ret_header:
            return header

        return data

    def download(
        self,
        source: str,
        to_file: Union[str, Path, bool, None] = False,
        dir_path: Union[str, Path, None] = None,
        to_memory: bool = True,
        return_data: bool = False,
        overwrite: bool = False,
        verbose: bool = True,
        chunk_size: int = 1024,
        print_size: float = 0.00001,
    ) -> Union[Path, pd.DataFrame, None]:
        """Download a dataset from a specified source and save it locally."""
        # Check the source
        source = source.lower()
        if source in ["circumbinary", "c", "p"]:
            circumbinary = True
        elif source in ["simple", "s"]:
            circumbinary = False
        else:
            raise ValueError(
                "Invalid source. "
                + "Must be 'circumbinary', 'c', 'p', 'simple', or 's'."
            )

        # Define the filename based on the circumbinary parameter
        letter = "p" if circumbinary else "s"

        # Check if something to do
        if not to_file and not to_memory and not return_data:
            raise ValueError(
                "Nothing to do. Set at least one of "
                + "to_file, to_zip, to_memory, or return_data."
            )
        if (
            not to_file
            and to_memory
            and not return_data
            and not self._data[letter].empty
            and not overwrite
        ):
            raise ValueError(
                "Nothing to do. Dataset is already stored in memory and "
                + "overwrite is False."
            )

        # Define URS
        url = BINARIES_URLS[letter]

        # Redefine dir path
        if dir_path is None or dir_path is True:
            dir_path = DATASETS_DIR
            # Default directory
            dir_path = DATASETS_DIR
        elif not dir_path:
            # Assuming no file required
            to_file = False
        elif dir_path:
            # Convert to Path
            dir_path = Path(dir_path)
            # Check if directory exists
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory {dir_path} not found.")

        # Check if file exists
        if to_file:
            if to_file is True:
                file_name = BINARIES_FILENAMES[letter]
            else:
                file_name = to_file
            file_path = dir_path / Path(file_name)
            if file_path.exists() and not overwrite:
                raise FileExistsError(
                    f"File {file_path} already exists.\n"
                    + " Set overwrite=True to overwrite it."
                )
        else:
            file_path = None
            file_name = BINARIES_FILENAMES[letter]

        # Download the dataset
        data = request_dataset(
            url, verbose=verbose, chunk_size=chunk_size, print_size=print_size
        )

        # Check if the data is valid. If not, raise an error. Check length > 0
        if not data or len(data) == 0:
            raise ValueError(f"Empty dataset downloaded from {url}.")
        elif verbose:
            if len(data) < 1e6:
                print(
                    f" Data downloaded successfully. ({len(data)/1e3:.3f} KB)"
                )
            else:
                print(
                    f" Data downloaded successfully. ({len(data)/1e6:.3f} MB)"
                )

        # Default df
        df = pd.DataFrame()

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
            header, df = self._extract_header_and_data(
                lines=StringIO(data.decode(encoding="utf-8")).readlines(),
                circumbinary=circumbinary,
                inferr=False,
            )
            if to_memory:
                # Store the data in memory
                self._headers[letter] = header
                self._data[letter] = df
                if verbose:
                    print(f" Stored the type-{letter} dataset in memory.")

        # Return the data
        if return_data:
            # Try to rename the columns
            try:
                df.columns = BINARIES_COLUMNS
            except ValueError:
                if verbose:
                    print("Columns could not be renamed.")
            return df

        # Return the path
        if file_path:
            return file_path

        return

    def clear_memory(
        self, source: str, verbose: bool = True, files: bool = False
    ):
        """Clear stored binary data from memory and/or disk."""
        source = source.lower()
        if files:
            if source in BINARIES_FILENAMES:
                file_path = DATASETS_DIR / BINARIES_FILENAMES[source]
                if file_path.exists():
                    file_path.unlink()
                    if verbose:
                        print(f" Removed {file_path} from disk.")
            elif source in ["both", "all"]:
                for key in BINARIES_FILENAMES:
                    self.clear_memory(key, verbose=verbose, files=True)
            else:
                raise ValueError("Invalid binary source.")
        else:
            if source in self._data:
                self._data[source] = pd.DataFrame()
                self._headers[source] = ""
                if verbose:
                    print(f" Cleared memory for binaries type-{source}")
            elif source == "binary":
                for key in self._data:
                    self.clear_memory(key, verbose=verbose)
            else:
                raise ValueError("Invalid binary source.")


# -------------------------  INITIALIZATION --------------------------

_full_manager = DatasetManager()
_binary_manager = BinaryDatasetManager()


# =============================================================================
# FUNCTIONS
# =============================================================================

# --------------------------- EU AND NASA DATASETS ----------------------------


def load_dataset(
    source: str,
    from_memory: bool = True,
    from_zip: Union[str, bool] = True,
    from_file: Union[str, bool] = False,
    dir_path: Union[str, Path, bool] = True,
    to_resokit: bool = True,
    to_df: bool = False,
    check_age: bool = False,
    only_index: bool = False,
    only_rows: Union[list, int] = False,
    verbose: bool = True,
    store: Union[bool, str] = False,
    store_index: Union[bool, str] = True,
) -> Union[pd.DataFrame, ResokitDataFrame, ResoKitDataset, None]:
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
    from_file : str or Path or bool, optional. Default: False.
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
    return _full_manager.load(
        source=source,
        from_memory=from_memory,
        from_zip=from_zip,
        from_file=from_file,
        dir_path=dir_path,
        to_resokit=to_resokit,
        to_df=to_df,
        check_age=check_age,
        only_index=only_index,
        only_rows=only_rows,
        verbose=verbose,
        store=store,
        store_index=store_index,
    )


def download_dataset(
    source: str,
    to_memory: bool = True,
    to_file: Union[str, Path, bool, None] = False,
    to_zip: Union[str, Path, bool, None] = False,
    dir_path: Union[str, Path, None] = None,
    overwrite: bool = False,
    check_online: bool = True,
    to_resokit: Union[bool, None] = None,
    verbose: bool = True,
    chunk_size: int = 1024,
    print_size: float = 0.15,
) -> Union[Path, pd.DataFrame, ResoKitDataset, None, dict]:
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
        If "all" or "both", downloads both datasets.
    to_memory : bool, optional. Default: False.
        If `True`, stores the dataset in memory.
    to_file : str or Path or bool, optional. Default: False.
        Path or str to the file to store the dataset.
        If `True`, default filename is used.
        If `False`, the file is not saved nor created.
    to_zip : str or Path or bool, optional. Default: False.
        Path or str to the ZIP archive to store the dataset.
        If `True`, default ZIP filename is used. (datasets.zip)
        If `False`, the file is not saved nor created in the ZIP archive.
    dir_path : str or Path
        Directory path to save the dataset, or path to the ZIP archive.
        If `None`, the default directory is used (resokit.datasets).
    overwrite : bool, optional. Default: False.
        If `True`, overwrites the file if it already exists.
        The memory stored Dataset and Index are always overwritten,
        independently of this parameter.
    check_online : bool, optional. Default: True.
        Whether to check if the dataset is already up-to-date.
    to_resokit : bool, dict, optional. Default: None.
        If `True`, returns the dataset as a ResoKitDataset.
        If `False`, returns the dataset as a pandas DataFrame.
        If `None`, returns the path to the downloaded file.
    verbose : bool, optional. Default: True.
        If `True`, displays messages about the download process.
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
    if source.lower() in ["all", "both"]:
        # Download both datasets
        eu = download_dataset(
            source="eu",
            to_memory=to_memory,
            to_file=to_file,
            to_zip=to_zip,
            dir_path=dir_path,
            overwrite=overwrite,
            check_online=check_online,
            to_resokit=to_resokit,
            verbose=verbose,
            chunk_size=chunk_size,
            print_size=print_size,
        )
        nasa = download_dataset(
            source="nasa",
            to_memory=to_memory,
            to_file=to_file,
            to_zip=to_zip,
            dir_path=dir_path,
            overwrite=overwrite,
            check_online=check_online,
            to_resokit=to_resokit,
            verbose=verbose,
            chunk_size=chunk_size,
            print_size=print_size,
        )
        return {"eu": eu, "nasa": nasa}
    return _full_manager.download(
        source=source,
        to_memory=to_memory,
        to_file=to_file,
        to_zip=to_zip,
        dir_path=dir_path,
        overwrite=overwrite,
        check_online=check_online,
        to_resokit=to_resokit,
        verbose=verbose,
        chunk_size=chunk_size,
        print_size=print_size,
    )


# --------------------------- BINARY SYSTEMS DATASETS -------------------------


def load_binary_dataset(
    which: Union[str, bool],
    from_memory: bool = True,
    from_file: Union[str, bool] = True,
    dir_path: Union[str, Path, bool] = True,
    rename_columns: bool = True,
    ret_header: bool = False,
    inferr: bool = False,
    clean: bool = True,
    verbose: bool = True,
) -> Union[pd.DataFrame, str]:
    """Load a binary dataset.

    Parameters
    ----------
    which : str, bool
        Which dataset to load:
        'circumbinary' or 'c' or 'p' for the p-type circumbinaries dataset,
        'simple' or 's' for the s-type binaries dataset.
        If `True`, loads the default dataset (circumbinary).
        If `False`, loads the simple binary dataset.
    from_memory : bool, optional. Default: True.
        If `True`, loads the dataset from memory if available.
    from_file : str or bool, optional. Default: True.
        If `True`, default filename is used.
        If `False`, the file is not loaded.
    dir_path : str, Path or bool, optional. Default: True.
        Directory path to load the dataset from.
        If `True` or `None` the default directory is used. (resokit.datasets)
    rename_columns : bool, optional. Default: True.
        If True, rename the columns for human readability.
    ret_header : bool, optional. Default: False.
        If True, return the header.
        If False, return the data.
    inferr : bool, optional. Default: False.
        If False, the width of the columns is fixed. (Recommended)
        If True, the parsed width of the columns is inferred. Use in case
        the dataset cannot be parsed with fixed-width columns.
    clean : bool, optional. Default: True.
        If True, replace the unknown values with NaN.
    verbose : bool, optional. Default: True.
        If True, print the header and messages.

    Returns
    -------
    Union[pd.DataFrame, str]
        header : str if ret_header is True.
            The header of the dataset.
        data : pd.DataFrame if ret_header is False.
            The dataset as a pandas DataFrame.
    """
    # Check the which parameter
    if isinstance(which, bool):
        if which:
            which = "circumbinary"
        else:
            which = "simple"
    return _binary_manager.load(
        source=which,
        from_memory=from_memory,
        from_file=from_file,
        dir_path=dir_path,
        rename_columns=rename_columns,
        ret_header=ret_header,
        inferr=inferr,
        clean=clean,
        verbose=verbose,
    )


def download_binary_dataset(
    which: str,
    to_file: Union[str, Path, bool, None] = False,
    dir_path: Union[str, Path, None] = None,
    to_memory: bool = True,
    return_data: bool = False,
    overwrite: bool = False,
    verbose: bool = True,
    chunk_size: int = 1024,
    print_size: float = 0.00001,
) -> Union[Path, pd.DataFrame, None, dict]:
    """Download a binary dataset from a specified source and save it locally.

    The dataset is downloaded from the internet and can be stored in a file,
    in memory, and/or simply returned.

    Note
    ----
    Requires the requests library.

    Parameters
    ----------
    which : str
        Which dataset to download:
        'circumbinary' or 'c' or 'p' for the p-type circumbinaries dataset,
        'simple' or 's' for the s-type binaries dataset.
        If "all" or "both", downloads both datasets.
    dir_path : str or Path
        Directory path to save the dataset.
        If `None`, the default directory is used (resokit.datasets).
    to_file : str or Path or bool, optional. Default: False.
        Path or str to the file to store the dataset.
        If `True`, default filename is used.
        If `False`, the file is not saved nor created.
    to_memory : bool, optional. Default: True.
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
    if which.lower() in ["all", "both"]:
        # Download both datasets
        s = download_binary_dataset(
            which="simple",
            to_file=to_file,
            dir_path=dir_path,
            to_memory=to_memory,
            return_data=return_data,
            overwrite=overwrite,
            verbose=verbose,
            chunk_size=chunk_size,
            print_size=print_size,
        )
        p = download_binary_dataset(
            which="circumbinary",
            to_file=to_file,
            dir_path=dir_path,
            to_memory=to_memory,
            return_data=return_data,
            overwrite=overwrite,
            verbose=verbose,
            chunk_size=chunk_size,
            print_size=print_size,
        )
        return {"s": s, "p": p}
    return _binary_manager.download(
        source=which,
        to_file=to_file,
        dir_path=dir_path,
        to_memory=to_memory,
        return_data=return_data,
        overwrite=overwrite,
        verbose=verbose,
        chunk_size=chunk_size,
        print_size=print_size,
    )


# --------------------------- AUXILIAR FUNCTIONS ----------------------------


def clear_memory(
    which: str, verbose: bool = True, files: bool = False
) -> None:
    """Clear the memory for the specified dataset.

    Parameters
    ----------
    which : str
        Which dataset ('eu', 'nasa', 'datasets',
        'p', 's', 'binary', 'all').
    verbose : bool, optional. Default: True.
        Whether to print informational messages.
    files : bool, optional. Default: False.
        If `True`, also removes the files from disk.
    """
    which = which.lower()  # Ensure lowercase
    if which in DATASET_FILENAMES:
        _full_manager.clear_memory(source=which, verbose=verbose)
        _full_manager.clear_memory(source=which, verbose=verbose, files=files)
    elif which in BINARIES_FILENAMES:

        _binary_manager.clear_memory(source=which, verbose=verbose)
        _binary_manager.clear_memory(
            source=which, verbose=verbose, files=files
        )
    elif which == "datasets":

        _full_manager.clear_memory(source="both", verbose=verbose)
        _full_manager.clear_memory(source="both", verbose=verbose, files=files)
    elif which == "binary":

        _binary_manager.clear_memory(source="both", verbose=verbose)
        _binary_manager.clear_memory(
            source="both", verbose=verbose, files=files
        )
    elif which == "all":
        _full_manager.clear_memory(source="both", verbose=verbose)
        _binary_manager.clear_memory(source="both", verbose=verbose)
        _full_manager.clear_memory(source="both", verbose=verbose, files=files)
        _binary_manager.clear_memory(
            source="both", verbose=verbose, files=files
        )
    else:
        raise ValueError(
            f"Invalid source: {which}. Must be 'eu', 'nasa', 'p', 's', "
            + "'binary', 'datasets', or 'all'."
        )


def check_outdated(which: str, verbose: bool = True, soft=True) -> bool:
    """Check if the specified stored dataset is outdated.

    Parameters
    ----------
    which : str
        Which dataset ('eu' or 'nasa').
    verbose : bool, optional. Default: True.
        Whether to print informational messages.

    Returns
    -------
    outdated : bool
        Whether the dataset is outdated.
    """
    # Check if which is valid
    which = which.lower()  # Ensure lowercase
    if which not in DATASET_FILENAMES:
        if which in BINARIES_FILENAMES:
            if verbose:
                print(
                    f"Use `check_binary_outdated('{which}') to check if"
                    + "binary dataset is outdated."
                )
        raise ValueError(f"Invalid which: {which}. Must be 'eu' or 'nasa'.")

    if verbose:
        print(f"Checking local dataset from {which}...")

    # Check if the dataset is stored
    try:
        if which == "eu":
            df_stored = _full_manager.load(
                "eu",
                verbose=False,
                to_df=True,
                only_index=True,
                check_age=True,
                only_rows=False,
                store=False,
                store_index=True,
            )
        else:
            df_stored = _full_manager.load(
                "nasa",
                verbose=False,
                to_df=True,
                to_resokit=False,
                check_age=True,
                only_index=False,
                only_rows=False,
                store=False,
                store_index=True,
            )
            # Keep only non controversial and default_flag
            df_stored = df_stored[df_stored["default_flag"] == 1]
    except FileNotFoundError as error:
        if verbose:
            print(f"File for source {which} to check if outdated not found.")
        if soft:
            return True
        raise error
    assert isinstance(df_stored, pd.DataFrame), (
        "Expected df_stored to be a pd.DataFrame, "
        + f"got {type(df_stored)} instead."
    )
    n_local = len(df_stored)
    if n_local > 0 and verbose:
        print(f" Number of planets in stored dataset: {n_local}")
        if which == "nasa":
            print("  (Including also non-default parameters set.)")
    elif verbose:
        print("Could not load the stored dataset. ")

    # Check if the dataset is outdated
    n_online, _ = check_online_dataset(source=which, verbose=verbose)

    if n_online == n_local:
        if verbose:
            print("Dataset is already up-to-date.")
        return False
    elif n_online <= 0:
        if verbose:
            print("Cannot check if the dataset is up-to-date. ")
            print("The dataset could be outdated.")
        return True
    elif n_online < n_local:
        if verbose:
            print("The online dataset has less rows than the stored dataset. ")
            print("This is unexpected.")
        return False
    # n_online > n_local
    if verbose:
        print("The online dataset has more rows than the stored dataset. ")
        print("The dataset is outdated.")

    return True


def check_binary_outdated(
    which: Union[str, bool], verbose: bool = True, soft=True
) -> bool:
    """Check if the specified stored bianry dataset is outdated.

    Parameters
    ----------
    which : str, bool
        Which dataset: 'p' (circumbinary) or 's' (single binary).
        If True, circumbinary; if False, single binary.
    verbose : bool, optional. Default: True.
        Whether to print informational messages.

    Returns
    -------
    outdated : bool
        Whether the dataset is outdated.
    """
    # Check if which is valid
    if isinstance(which, bool):
        which = "p" if which is True else "s"
    which = which.lower()  # Ensure lowercase
    if which not in BINARIES_FILENAMES:
        if which in DATASET_FILENAMES:
            if verbose:
                print(
                    f"Use `check_outdated('{which}') to check if "
                    + f"'{which}' dataset is outdated."
                )
        raise ValueError(f"Invalid which: {which}. Must be 'p' or 's'.")

    # Check if the dataset is stored
    try:
        header = _binary_manager.load(source=which, ret_header=True)
        df = _binary_manager.load(source=which, ret_header=False)
    except FileNotFoundError as error:
        if verbose:
            print(f"File for source {which} to check if outdated not found.")
        if soft:
            return True
        raise error

    assert isinstance(header, str), (
        "Expected header to be a str, " + f"got {type(header)} instead."
    )
    n_local = len(df) + len(header.splitlines())
    if n_local > 0 and verbose:
        print(f" Number of lines in stored dataset: {n_local}")
    elif verbose:
        print("Could not load the stored dataset. ")

    # Check if the dataset is outdated
    n_online = check_online_binary(source=which, verbose=verbose)

    if n_online == n_local:
        if verbose:
            print("Dataset is already up-to-date.")
        return False
    elif n_online <= 0:
        if verbose:
            print("Cannot check if the dataset is up-to-date. ")
            print("The dataset could be outdated.")
        return True
    elif n_online < n_local:
        if verbose:
            print("The online dataset has less rows than the stored dataset. ")
            print("This is unexpected.")
        return False
    # n_online > n_local
    if verbose:
        print("The online dataset has more rows than the stored dataset. ")
        print("The dataset is outdated.")

    return True
