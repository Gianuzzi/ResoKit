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

from difflib import SequenceMatcher
from typing import Tuple, Union

import pandas as pd

from resokit.core import (
    ResokitDataFrame,
    StaticSystem,
    df_to_resokit,
    resokit_to_system,
)
from resokit.datasets.databases import load_full
from resokit.utils.utils import DEFAULT_METADATA

# =============================================================================
# CONSTANTS
# =============================================================================

RATIOS_THRESHOLD = 0.94

# =============================================================================
# FUNCTIONS
# =============================================================================


def _similar(a: str, b: str) -> float:
    """Calculate the similarity ratio between two strings."""
    return SequenceMatcher(None, str(a), b).ratio()


def _n_close(a: any, b: str, length: int, n=0) -> bool:
    """Check if two strings are n spaces-close."""
    stra = str(a)  # Convert to string

    return (stra[:length] == str(b)) and (
        (len(stra) == length + n) or stra[length] == " "
    )


def _search_system_index(
    source: str,
    name: str,
    is_planet: bool = False,
    raw_df: pd.DataFrame = None,
    **load_extra_kwargs,
) -> Tuple[pd.Index, pd.Series, float]:
    """Search for the index of the system in the dataset.

    Parameters
    ----------
    source : str
        Source of the dataset. Either 'eu' or 'nasa'.
    name : str
        Name of the system or planet.
    is_planet : bool, optional. Default: False.
        Whether to search for a planet or a star.
    raw_df : pd.DataFrame, optional. Default: None.
        Raw dataset used for the search, instead of loading it.
    load_extra_kwargs : dict
        Extra keyword arguments for the load function.

    Returns
    -------
    index : pd.Index
        Index of the system.
    values : pd.Series
        Values of the system.
    ratio : float
        Similarity ratio.
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
        raw_df
        if raw_df is not None
        else load_full(
            source=source,
            **load_extra_kwargs,
        )
    )
    raw_series = raw_series[column]  # Get the column

    # Search for the system
    exact_matches = raw_series[raw_series == name]
    if not exact_matches.empty:
        return exact_matches.index, exact_matches.values, 1

    # If no exact matches, search for 1 space-close names
    length = len(name)
    close_matches = raw_series.apply(lambda x: _n_close(x, name, length, 1))

    if close_matches.any():  # If 1 space-close names found
        return raw_series[close_matches].index, raw_series[close_matches], 0.9

    # If no 1 space-close names, search for 2 space-close names
    close_matches = raw_series.apply(lambda x: _n_close(x, name, length, 2))

    if close_matches.any():  # If 2 space-close names found
        return raw_series[close_matches].index, raw_series[close_matches], 0.8

    # If no 2 space-close names, search for similar names
    similarity_ratios = raw_series.apply(lambda x: _similar(x, name))
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


def _load_system_from_db(
    name: str,
    is_planet: bool = False,
    source: str = None,
    store: bool = False,
    store_index: bool = True,
    load_kwargs: dict = None,
    verbose: bool = True,
    low_memory: bool = False,
) -> pd.DataFrame:
    """Load system from ExoplanetEU or NASA.

    Parameters
    ----------
    name : str
        System/planet name.
    is_planet : bool, optional. Default: False.
        Whether to search for a planet or a star.
    source : str, optional. Default: None.
        Source of the dataset. Either 'eu' or 'nasa'.
    store : bool, optional. Default: False.
        Whether to store the whole dataset in memory.
    store_index : bool, optional. Default: True.
        Whether to store the whole dataset index in memory.
        Automatically set to True if store is True.
    load_kwargs : dict, optional. Default: {}.
        Extra keyword arguments for the load function.
    verbose : bool, optional. Default: True.
        Whether to print information.
    low_memory : bool, optional. Default: False.
        Whether to avoid loading the whole dataset into memory.
        Instead, first loads only the index,
        and then only the system data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the system data.
    """
    # Print information
    if verbose:
        print(
            f"Loading {'planet' if is_planet else 'star system'} {name} "
            + f"from {source}."
        )

    # If storing, then load the whole dataset
    if store:
        store_index = True  # Store the index if the dataset will be stored
        low_memory = False  # Load the whole dataset if it will be stored

    # Update the keyword arguments
    if load_kwargs is None:
        load_kwargs = {}
    load_kwargs.update(
        {
            "store": store,
            "verbose": verbose,
            "store_index": store_index,
            "to_resokit": False,
            "only_rows": None,
            "only_index": False,
        }
    )

    # Load the dataset
    if not low_memory:  # Load the whole dataset
        raw_df = load_full(source=source, **load_kwargs)
    else:  # Will load only the index if possible
        raw_df = None

    # Define the keyword arguments for the system loading
    load_extra_kwargs = {**load_kwargs, "only_index": True}

    # Search for the system
    idx, values, ratio = _search_system_index(
        source=source,
        name=name,
        is_planet=is_planet,
        raw_df=raw_df,
        **load_extra_kwargs,
    )

    # Check if the system was found
    if ratio < 1:
        if is_planet:
            print(f"Planet {name} not found in {source} dataset.")
        else:
            print(f"Star {name} not found in {source} dataset.")
        if ratio == 0:  # No similar names found
            return pd.DataFrame()

        # Note: get most probable by whitespace separation
        most_prob = list(set(val for val in values if name + " " in val))
        others = list(set(val for val in values if val not in most_prob))

        most_prob.sort()  # Sort the most probable
        others.sort()  # Sort the others

        # Forced to print the most probable and others
        print(f" Similar names found in {source} dataset:")
        print(f" - {most_prob + others}")

        return pd.DataFrame()  # Return an empty DataFrame

    # Load the system
    if raw_df is None:  # Load only the system data
        # Update the keyword arguments
        load_kwargs.update({"only_rows": idx.to_list()})
        return load_full(source=source, **load_kwargs)

    return raw_df.loc[idx]  # Load the system data from the raw dataset


def load_system_from_eu(
    name: str,
    is_planet: bool = False,
    load_kwargs: dict = None,
    drop: bool = True,
    store: bool = False,
    store_index: bool = True,
    verbose: bool = True,
    low_memory: bool = True,
    as_resokit: bool = False,
) -> Union[ResokitDataFrame, StaticSystem]:
    """Load system from ExoplanetEU.

    Parameters
    ----------
    name : str
        System/planet name.
        (Remember case sensitivity)
    is_planet : bool, optional. Default: False.
        Whether to search for a planet or a star.
    load_kwargs : dict, optional. Default: {}.
        Keyword arguments for the load function.
    drop : bool, optional. Default: True.
        Whether to drop extra columns.
    store : bool, optional. Default: False.
        Whether to store the whole dataset in memory.
    store_index : bool, optional. Default: True.
        Whether to store the whole dataset index in memory.
        Automatically set to True if store is True.
    verbose : bool, optional. Default: True.
        Whether to print information.
    low_memory : bool, optional. Default: True.
        Whether to avoid loading the whole dataset into memory.
    as_resokit : bool, optional. Default: False.
        Whether to return the dataset in ResoKit format.

    Returns
    -------
    system : ResokitDataFrame or StaticSystem
        Loaded system as :py:class:`ResokitDataFrame` (if `as_resokit=True`),
        or :py:class:`StaticSystem`.
    """
    if load_kwargs is None:
        load_kwargs = {}

    # Load the system from the database
    df = _load_system_from_db(
        name=name,
        is_planet=is_planet,
        source="eu",
        store=store,
        store_index=store_index,
        load_kwargs=load_kwargs,
        verbose=verbose,
        low_memory=low_memory,
    )

    # Can't work with empty DataFrame
    if df.empty:
        return df

    # Convert the DataFrame to ResoKit format
    # Note: Metadata is set from default values
    meta = dict(DEFAULT_METADATA)
    meta.update({f"load_{'planet' if is_planet else 'system'}": name})
    meta.update({"eu_index": int(df.index[0])})

    reso = df_to_resokit(  # Convert to ResoKit format
        df=df,
        source="eu",
        drop=drop,
        copy=False,
        metadata=meta,
    )

    if not as_resokit:  # Return StaticSystem
        return resokit_to_system(reso)

    return reso  # Return ResoKit DataFrame


def load_system_from_nasa(
    name: str,
    is_planet: bool = False,
    load_kwargs: dict = None,
    drop: bool = True,
    store: bool = False,
    store_index: bool = True,
    verbose: bool = True,
    low_memory: bool = True,
    controversial_set: bool = False,
    default_set: bool = True,
    as_resokit: bool = False,
) -> Union[ResokitDataFrame, StaticSystem]:
    """Load system from NASA.

    Parameters
    ----------
    name : str
        System/planet name.
        (Remember case sensitivity)
    is_planet : bool, optional. Default: False.
        Whether to search for a planet or a star.
    load_kwargs : dict, optional. Default: {}.
        Keyword arguments for the load function.
    drop : bool, optional. Default: True.
        Whether to drop extra columns.
    store : bool, optional. Default: False.
        Whether to store the whole dataset in memory.
    store_index : bool, optional. Default: True.
        Whether to store the whole dataset index in memory.
        Automatically set to True if store is True.
    verbose : bool, optional. Default: True.
        Whether to print information.
    low_memory : bool, optional. Default: True.
        Whether to avoid loading the whole dataset into memory.
    controversial_set : bool, optional. Default: False.
        Whether to include controversial data.
        None to include all data.
    default_set : bool, optional. Default: True.
        Whether to include default data.
        None to include all data.
    as_resokit : bool, optional. Default: False.
        Whether to return the dataset in ResoKit format.

    Returns
    -------
    system : ResokitDataFrame or StaticSystem
        Loaded system as :py:class:`ResokitDataFrame` (if `as_resokit=True`),
        or :py:class:`StaticSystem`.
    """
    if load_kwargs is None:
        load_kwargs = {}

    # Load the system from the database
    df = _load_system_from_db(
        name=name,
        is_planet=is_planet,
        source="nasa",
        load_kwargs=load_kwargs,
        store=store,
        store_index=store_index,
        verbose=verbose,
        low_memory=low_memory,
    )

    # Check if the dataset is empty
    if df.empty:
        return df  # Can't work with empty DataFrame

    # Filter controversial data
    if controversial_set is not None:
        df = df[df["pl_controv_flag"] == int(controversial_set)]

    # Filter default data
    if default_set is not None:
        df = df[df["default_flag"] == int(default_set)]

    # Convert the DataFrame to ResoKit format
    # Note: Metadata is set from default values
    meta = dict(DEFAULT_METADATA)
    meta.update({f"load_{'planet' if is_planet else 'system'}": name})
    meta.update({"nasa_index": int(df.index[0])})

    reso = df_to_resokit(  # Convert to ResoKit format
        df=df,
        source="nasa",
        drop=drop,
        copy=False,
        metadata=meta,
    )

    if not as_resokit:  # Return StaticSystem
        return resokit_to_system(reso)

    return reso  # Return ResoKit DataFrame
