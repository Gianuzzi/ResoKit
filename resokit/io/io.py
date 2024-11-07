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

"""Module with input/output functions for the ResoKit package."""

# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd
from resokit.io.utils import _EU_MAPPING, _NASA_MAPPING, __n_close, __similar
from resokit.datasets import load_dataset
from resokit.datasets.databases import (
    _load_dataset_expanded,
    IN_MEMORY_DATASETS,
    IN_MEMORY_INDEXES,
)

# =============================================================================
# CONSTANTS
# =============================================================================

MAPPINGS = {"eu": _EU_MAPPING, "nasa": _NASA_MAPPING}
RATIOS_THRESHOLD = 0.94

# =============================================================================
# FUNCTIONS
# =============================================================================


def _search_system_index(
    source: str,
    name: str,
    is_planet: bool = False,
    store_index: bool = True,
    verbose: bool = False,
    raw_df: pd.DataFrame = None,
) -> tuple[pd.Index, pd.Series, float]:
    """
    Search for the index of the system in the dataset.

    Parameters
    ----------
    source : str
        Source of the dataset. Either 'eu' or 'nasa'.
    name : str
        Name of the system or planet.
    is_planet : bool, optional
        Whether to search for a planet or a star.
    store_index : bool, optional
        Whether to store the index in memory.
    verbose : bool, optional
        Whether to print information.
    raw_df : pd.DataFrame, optional
        Raw dataset.

    Returns
    -------
    tuple[pd.Index, pd.Series, float]
        Index, values, and similarity ratio.
    """
    # Define the column to search
    column = (
        "pl_name"
        if is_planet and source == "nasa"
        else (
            "hostname"
            if source == "nasa"
            else "name" if is_planet else "star_name"
        )
    )

    # Load the dataset if not in memory
    raw_series = (
        IN_MEMORY_INDEXES[source]
        if IN_MEMORY_INDEXES[source] is not None
        else (
            raw_df
            if raw_df is not None
            else _load_dataset_expanded(
                source=source,
                only_index=True,
                verbose=verbose,
                store=store_index,
            )
        )
    )
    raw_series = raw_series[column]  # Get the column

    # Search for the system
    exact_matches = raw_series[raw_series == name]
    if not exact_matches.empty:
        return exact_matches.index, exact_matches.values, 1

    # If no exact matches, search for 1 space-close names
    length = len(name)
    close_matches = raw_series.apply(lambda x: __n_close(x, name, length, 1))
    if close_matches.any():
        return raw_series[close_matches].index, raw_series[close_matches], 0.9

    # If no 1 space-close names, search for 2 space-close names
    close_matches = raw_series.apply(lambda x: __n_close(x, name, length, 2))
    if close_matches.any():
        return raw_series[close_matches].index, raw_series[close_matches], 0.8

    # If no 2 space-close names, search for similar names
    similarity_ratios = raw_series.apply(lambda x: __similar(x, name))
    good_matches = similarity_ratios >= RATIOS_THRESHOLD
    similarity_ratios = similarity_ratios[good_matches]

    if similarity_ratios.empty:  # No similar names found
        return pd.Index([]), pd.Series([]), 0

    # Return the index, values, and the minimum similarity ratio
    return (
        similarity_ratios.index,
        raw_series[good_matches],
        similarity_ratios.values.min(),
    )


def load_system_from_eu(
    name: str,
    is_planet: bool = False,
    load_dataset_kwargs: dict = {},
    drop: bool = True,
    store: bool = False,
    store_index: bool = True,
    verbose: bool = True,
    low_memory: bool = False,
) -> pd.DataFrame:
    """
    Load system from ExoplanetEU.

    Parameters
    ----------
    name : str
        System/planet name.
        (Remember case sensitivity)
    is_planet : bool, optional
        Whether to search for a planet or a star.
    load_dataset_kwargs : dict, optional
        Keyword arguments for the load_dataset function.
    drop : bool, optional
        Whether to drop extra columns.
    store : bool, optional
        Whether to store the whole dataset in memory.
    store_index : bool, optional
        Whether to store the whole dataset index in memory.
        Automatically set to True if store is True.
    verbose : bool, optional
        Whether to print information.
    low_memory : bool, optional
        Whether to avoid loading the whole dataset into memory.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the system data.
    """
    return _load_system_from_db(
        name=name,
        is_planet=is_planet,
        source="eu",
        load_dataset_kwargs=load_dataset_kwargs,
        drop=drop,
        store=store,
        store_index=store_index,
        verbose=verbose,
        low_memory=low_memory,
    )


def load_system_from_nasa(
    name: str,
    is_planet: bool = False,
    load_dataset_kwargs: dict = {},
    drop: bool = True,
    store: bool = False,
    store_index: bool = True,
    verbose: bool = True,
    low_memory: bool = False,
    controversial_set: bool = False,
    default_set: bool = True,
) -> pd.DataFrame:
    """
    Load system from NASA.

    Parameters
    ----------
    name : str
        System/planet name.
        (Remember case sensitivity)
    is_planet : bool, optional
        Whether to search for a planet or a star.
    load_dataset_kwargs : dict, optional
        Keyword arguments for the load_dataset function.
    drop : bool, optional
        Whether to drop extra columns.
    store : bool, optional
        Whether to store the whole dataset in memory.
    store_index : bool, optional
        Whether to store the whole dataset index in memory.
        Automatically set to True if store is True.
    verbose : bool, optional
        Whether to print information.
    low_memory : bool, optional
        Whether to avoid loading the whole dataset into memory.
    controversial_set : bool, optional
        Whether to include controversial data.
        None to include all data.
    default_set : bool, optional
        Whether to include default data.
        None to include all data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the system data.
    """
    df = _load_system_from_db(
        name=name,
        is_planet=is_planet,
        source="nasa",
        load_dataset_kwargs=load_dataset_kwargs,
        drop=drop,
        store=store,
        store_index=store_index,
        verbose=verbose,
        low_memory=low_memory,
    )

    if df.empty:
        return df

    if controversial_set is not None:  # Filter controversial data
        df = df[df["controversial"] == int(controversial_set)]
    if default_set is not None:  # Filter default data
        df = df[df["default_set"] == int(default_set)]
    return df


def _load_system_from_db(
    name: str,
    is_planet: bool = False,
    source: str = None,
    load_dataset_kwargs: dict = {},
    drop: bool = True,
    store: bool = False,
    store_index: bool = True,
    verbose: bool = True,
    low_memory: bool = False,
) -> pd.DataFrame:
    """
    Load system from ExoplanetEU or NASA.

    Parameters
    ----------
    name : str
        System/planet name.
    is_planet : bool, optional
        Whether to search for a planet or a star.
    source : str, optional
        Source of the dataset. Either 'eu' or 'nasa'.
    load_dataset_kwargs : dict, optional
        Keyword arguments for the load_dataset function.
    drop : bool, optional
        Whether to drop extra columns.
    store : bool, optional
        Whether to store the whole dataset in memory.
    store_index : bool, optional
        Whether to store the whole dataset index in memory.
        Automatically set to True if store is True.
    verbose : bool, optional
        Whether to print information.
    low_memory : bool, optional
        Whether to avoid loading the whole dataset into memory.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the system data.
    """
    if store:
        store_index = True

    raw_df = IN_MEMORY_DATASETS[source]

    # Load the dataset if not in memory
    if store or not low_memory:
        load_dataset_kwargs.update({"store": store, "verbose": verbose})
        raw_df = load_dataset(source=source, **load_dataset_kwargs)

    if store and verbose:
        print(f"Run load_reso_dataset(source='{source}') to load it.")

    # Search for the system
    idx, values, ratio = _search_system_index(
        source=source,
        name=name,
        is_planet=is_planet,
        store_index=store_index,
        verbose=verbose,
        raw_df=raw_df,
    )

    # Check if the system was found
    if ratio < 1:
        if is_planet:
            print(f"Planet {name} not found in {source} dataset.")
        else:
            print(f"Star {name} not found in {source} dataset.")
        if ratio == 0:  # No similar names found
            return pd.DataFrame()
        # Similar names found
        print(f"Similar names found in {source} dataset:")
        print(set(values.values))
        return pd.DataFrame()

    # Load the system
    if raw_df is None:
        raw_df = _load_dataset_expanded(
            source=source, only_rows=idx, verbose=verbose
        )
        return _convert_to_resokit_format(df=raw_df, source=source, drop=drop)
    else:
        return _convert_to_resokit_format(
            df=raw_df.loc[idx], source=source, drop=drop
        )


def _convert_to_resokit_format(
    df: pd.DataFrame,
    source: str,
    drop: bool = True,
    copy: bool = False,
) -> pd.DataFrame:
    """
    Convert ExoplanetEU or NASA dataset to ResoKit format.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    source : str
        Source of the dataset. Either 'eu' or 'nasa'.
    drop : bool
        Whether to drop columns not in the mapping.
    copy : bool
        Whether to return a copy of the DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame in ResoKit format.
    """
    # Get the new columns dictionary
    new_cols_dict = MAPPINGS[source]

    # Copy the DataFrame
    if copy:
        df = df.copy()
    # Rename columns
    df = df.rename(columns=new_cols_dict)
    # Drop columns not in the mapping
    if drop:
        df = df.drop(columns=set(df.columns) - set(new_cols_dict.values()))
    return df


def load_reso_dataset(
    source: str = "eu",
    check_age: bool = False,
    download_if_missing: bool = False,
    extract: bool = False,
    verbose: bool = True,
    store: bool = False,
    drop: bool = True,
) -> pd.DataFrame:
    """
    Load the ExoplanetEU or NASA dataset.

    Parameters
    ----------
    source : str, optional
        Source of the dataset. Either 'eu' or 'nasa'.
    check_age : bool, optional
        Whether to check the age of the dataset.
    download_if_missing : bool, optional
        Whether to download the dataset if missing.
    extract : bool, optional
        Whether to extract the dataset.
    verbose : bool, optional
        Whether to print information.
    store : bool, optional
        Whether to store the whole dataset in memory.
    drop : bool, optional
        Whether to drop extra columns.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the dataset.
    """
    source = source.lower()

    df_raw = load_dataset(
        source=source,
        check_age=check_age,
        download_if_missing=download_if_missing,
        extract=extract,
        verbose=verbose,
        store=store,
    )
    df = _convert_to_resokit_format(df=df_raw, source=source, drop=drop)

    return df
