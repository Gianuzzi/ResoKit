#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2024, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Module with input/output functions for the ResoKit package."""

# =============================================================================
# IMPORTS
# =============================================================================

from itertools import product
from typing import List, Tuple, Union

import pandas as pd

from resokit.core import (
    ResokitDataFrame,
    StaticBinaryStar,
    StaticSystem,
    binary_row_to_binary_star,
    df_to_resokit,
    resokit_to_system,
)
from resokit.datasets.databases import load_binary, load_full
from resokit.utils.parser import DEFAULT_METADATA, find_best_match

# =============================================================================
# FUNCTIONS
# =============================================================================


# --------------------------- EU and NASA -------------------------------------


def _search_system_index(
    source: str,
    name: str,
    is_planet: bool = False,
    raw_df: pd.DataFrame = None,
    alternative_names: bool = False,
    **load_kwargs,
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
    alternative_names : bool, optional. Default: False.
        Whether to search for alternative names.
    load_kwargs : dict
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
    # Check not to search for alternative names in NASA
    if alternative_names and source == "nasa":
        raise ValueError("Alternative names not available in NASA dataset.")

    # Define the column to search
    if not alternative_names:
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
    else:
        column = "alternate_names" if is_planet else "star_alternate_names"

    # Update the necessary keyword arguments
    load_kwargs.update(
        {
            "to_df": True,
            "to_resokit": False,
        }
    )

    # Define parsing
    parse = True
    # Load the dataset if not in memory
    if not alternative_names:
        # Update the keyword arguments
        parsed = load_full(
            source=source,
            **{**load_kwargs, "only_index": "parsed", "verbose": False},
        )
        if raw_df is not None:
            raw_series = raw_df
        elif parsed is not None:
            parse = None
            raw_series = parsed
        else:
            raw_series = load_full(
                source=source,
                **load_kwargs,
            )  # Will be stored and parsed next time
        raw_series = raw_series[column]  # Get the column
    else:
        raw_series = load_full(
            source=source,
            **{**load_kwargs, "only_index": False, "verbose": False},
        )
        raw_series = raw_series[column].str.split(", ").explode()

    # Use the new function
    index, values, ratio = find_best_match(
        raw_series, name=name, parse=parse, force=is_planet
    )

    # If parse, return originals
    if parse is not None:
        return index, values, ratio

    # If parse is None, then we have to get back the original values
    original_values = (
        load_full(
            source=source,
            **{**load_kwargs, "only_index": True, "verbose": False},
        )[column]
        .loc[index]
        .tolist()
    )

    # Redefine ratio if exact match
    if original_values[0] == name:
        ratio = 1

    return index, original_values, ratio


def _load_system_from_db(
    name: str,
    is_planet: bool = False,
    source: str = None,
    store: bool = False,
    store_index: bool = True,
    verbose: bool = True,
    low_memory: bool = False,
    alternative_names: bool = False,
    exact_match: bool = False,
    check_binary: bool = True,
) -> Tuple[pd.DataFrame, Tuple[str, int]]:
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
    verbose : bool, optional. Default: True.
        Whether to print information.
    low_memory : bool, optional. Default: False.
        Whether to avoid loading the whole dataset into memory.
        Instead, first loads only the index,
        and then only the system data.
    alternative_names : bool, optional. Default: False.
        Whether to search for alternative names. Only available in ExoplanetEU.
    exact_match : bool, optional. Default: False.
        Whether to return only an exact match.
    check_binary : bool, optional. Default: True.
        Whether to check if the system is a binary system.

    Returns
    -------
    Tuple[pd.DataFrame, Tuple[str,int] : data, binary
        data : Loaded system as a DataFrame.
        binary : Tuple with the binary information. If the system is a binary
            system, then the tuple is (cb_letter, dataset_index).
            If it is circumbinary, cb_letter is "p"; if it is circumstellar,
            cb_letter is "s". If the system is not a binary system, then
            cb_letter is "f" (for "false"); and if no binary information
            was found, then the cb_letter is "n" (for "none").
            The dataset_index is the index of the system in the dataset.
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

    # Check if alternative names are available
    if alternative_names:
        if source != "eu":
            raise ValueError(
                "Alternative names only available in ExoplanetEU dataset."
            )
        if verbose:
            print("Searching for alternative names.")

    # Define the keyword arguments
    load_kwargs = {
        "store": store,
        "verbose": verbose,
        "store_index": store_index,
        "to_resokit": False,
        "only_rows": None,
        "only_index": False,
        "to_df": True,
    }

    # Load the dataset
    if not low_memory:  # Load the whole dataset
        raw_df = load_full(source=source, **load_kwargs)
    else:  # Will load only the index if possible
        raw_df = None

    # Search for the system
    idx, values, ratio = _search_system_index(
        source=source,
        name=name,
        is_planet=is_planet,
        raw_df=raw_df,
        alternative_names=alternative_names if source == "eu" else False,
        **{**load_kwargs, "only_index": True},
    )

    auxmsg = "alternate names column of " if alternative_names else ""
    # Check if the system was found
    if ratio < 0.99999:  # To take into account the almost 1 ratio
        if is_planet:
            print(f"Planet {name} not found in {auxmsg}{source} dataset.")
        else:
            print(f"Star {name} not found in {auxmsg}{source} dataset.")
        if ratio == 0:  # No similar names found
            return pd.DataFrame(), "n", -1  # Return an empty DataFrame

        # Note: get most probable by whitespace separation
        most_prob = list(set(val for val in values if name + " " in val))
        others = list(set(val for val in values if val not in most_prob))

        most_prob.sort()  # Sort the most probable
        others.sort()  # Sort the others

        # Message for the most probable and others
        if verbose:
            print(f" Similar names found in {auxmsg}{source} dataset:")
            print(f" - {most_prob + others}")

            if source == "eu" and not alternative_names:
                print(
                    "Note: ExoplanetEU has alternative names "
                    + "for some systems. "
                )
                print(
                    "      If no similar names found, try searching with: "
                    + "alternative_names=True."
                )

        return pd.DataFrame(), "n", -1  # Return an empty DataFrame
    elif ratio < 1:  # Only spaces or hyphens differences
        # Note: get most probable by whitespace separation
        pl = "planet" if is_planet else "star"
        if verbose:
            print(
                f"Found almost exact match {pl} {values[0]} "
                + f"in {auxmsg}{source} dataset."
            )
        if exact_match:
            return pd.DataFrame(), "n", -1  # Return an empty DataFrame

    # In case duplicated entries (due to alternate nemes used), we use the
    # list of the set of idx.
    idx = list(set(idx))

    # Load the system
    if raw_df is None:  # Load only the system data
        data = load_full(source=source, **{**load_kwargs, "only_rows": idx})
    else:
        data = raw_df.loc[idx]  # Load the system data from the raw dataset

    # Check if the system is a binary system?
    is_binary = False  # Default: not a binary system
    binary_type = "f"  # Default: not a binary system
    if check_binary:  # Check if binary
        star_name_col = "star_name" if source == "eu" else "hostname"
        star_name = data[star_name_col].iloc[0]  # Get the (first) star name
        if verbose:
            print(f"Checking if {star_name} is a binary system...")
        is_binary, circumbinary, idxbin, values, _ = check_if_binary(
            star_name, exact_match=exact_match, verbose=verbose
        )
        # Confirm that if multiple solutions, they are the same index
        if len(values) > 1:
            if len(set(idxbin)) != 1:
                raise ValueError(
                    "Multiple values found, but different indexes."
                )
            idxbin = idxbin[0]  # Get the index
    else:  # Not checking if binary
        binary_type = "n"  # No binary information

    # Change is_bina
    if is_binary:
        binary_type = (
            "p" if circumbinary else "s"
        )  # Circumbinary or circumstellar

    # Return the system data and binary information
    return data, binary_type, idxbin


def load_system_from_eu(
    name: str,
    is_planet: bool = False,
    drop: bool = True,
    store: bool = False,
    store_index: bool = True,
    verbose: bool = True,
    low_memory: bool = True,
    as_resokit: bool = False,
    alternative_names: bool = False,
    exact_match: bool = False,
    check_binary: Union[bool, None] = None,
) -> Union[ResokitDataFrame, StaticSystem]:
    """Load system from ExoplanetEU.

    Parameters
    ----------
    name : str
        System/planet name.
        (Remember case sensitivity)
    is_planet : bool, optional. Default: False.
        Whether to search for a planet or a star.
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
    alternative_names : bool, optional. Default: False.
        Whether to search for alternative names.
    exact_match : bool, optional. Default: False.
        Whether to search for an exact match.
        If `False`, the search will be more flexible, and a
        very (very) similar name will be accepted. Useful for
        names with different characters (e.g., hyphens), or
        for names with extra information (e.g., "A" or "B").
    check_binary : bool, optional. Default: True.
        Whether to check if the system is a binary system.
        If it is a binary system indeed, then the final system
        created is a `StaticBinarySystem` instead of a `StaticSystem`.
        If `None`, the check will be performed only to print
        information (if `verbose=True`).

    Returns
    -------
    system : ResokitDataFrame or StaticSystem
        Loaded system as :py:class:`ResokitDataFrame` (if `as_resokit=True`),
        or :py:class:`StaticSystem`.
    """
    # Load the system from the database
    df, binary, bindx = _load_system_from_db(
        name=name,
        is_planet=is_planet,
        source="eu",
        store=store,
        store_index=store_index,
        verbose=verbose,
        low_memory=low_memory,
        alternative_names=alternative_names,
        exact_match=exact_match,
        check_binary=check_binary or check_binary is None,
    )

    # Can't work with empty DataFrame
    if df.empty:
        return df

    # Convert the DataFrame to ResoKit format
    # Note: Metadata is set from default values
    meta = dict(DEFAULT_METADATA)
    meta.update({f"load_{'planet' if is_planet else 'system'}": name})
    meta.update({"eu_indexes": [int(idx) for idx in df.index]})

    # Convert to ResoKit format
    reso = df_to_resokit(
        df=df,
        source="eu",
        drop=drop,
        copy=False,
        metadata=meta,
    )

    if not as_resokit:  # Return XSystem
        if binary:
            pass
            # return resokit_to_binary_system(reso, binary=True)  # TBD!!!
        return resokit_to_system(reso)  # Return StaticSystem

    return reso  # Return ResoKit DataFrame


def load_system_from_nasa(
    name: str,
    is_planet: bool = False,
    drop: bool = True,
    store: bool = False,
    store_index: bool = True,
    verbose: bool = True,
    low_memory: bool = True,
    controversial_set: bool = False,
    default_set: bool = True,
    as_resokit: bool = False,
    exact_match: bool = False,
    check_binary: Union[bool, None] = None,
) -> Union[ResokitDataFrame, StaticSystem]:
    """Load system from NASA.

    Parameters
    ----------
    name : str
        System/planet name.
        (Remember case sensitivity)
    is_planet : bool, optional. Default: False.
        Whether to search for a planet or a star.
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
    exact_match : bool, optional. Default: False.
        Whether to search for an exact match.
        If `False`, the search will be more flexible, and a
        very (very) similar name will be accepted. Useful for
        names with different characters (e.g., hyphens), or
        for names with extra information (e.g., "A" or "B").
    check_binary : bool, optional. Default: True.
        Whether to check if the system is a binary system.
        If it is a binary system indeed, then the final system
        created is a `StaticBinarySystem` instead of a `StaticSystem`.
        If `None`, the check will be performed only to print
        information (if `verbose=True`).

    Returns
    -------
    system : ResokitDataFrame or StaticSystem
        Loaded system as :py:class:`ResokitDataFrame` (if `as_resokit=True`),
        or :py:class:`StaticSystem`.
    """
    # Load the system from the database
    df, binary, bindx = _load_system_from_db(
        name=name,
        is_planet=is_planet,
        source="nasa",
        store=store,
        store_index=store_index,
        verbose=verbose,
        low_memory=low_memory,
        exact_match=exact_match,
        check_binary=check_binary or check_binary is None,
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


# --------------------------- Binary Stars ------------------------------------


def load_from_binary(
    name: str,
    exact_match: bool = False,
    as_pandas: bool = False,
    soft: bool = True,
    verbose: bool = True,
) -> StaticBinaryStar:
    """Load a binary star system from the dataset.

    Parameters
    ----------
    name : str
        Name of the binary star system to load.
    exact_match : bool, optional. Default is False.
        If True, return the exact match only.
        If False, return the best match.
    as_pandas : bool, optional. Default is False.
        If True, return the data as a pandas DataFrame.
    soft : bool, optional. Default is True.
        If True, return None if the star is not found.
        If False, raise an error if the star is not found.
    verbose : bool, optional. Default is True.
        If True, print messages.

    Returns
    -------
    StaticBinaryStar
        The loaded binary star system.
    """
    # Check if the star is part of a binary system
    is_binary, circumbinary, idx, _, _ = check_if_binary(
        star_name=name, exact_match=exact_match, verbose=verbose
    )

    if not is_binary:
        if soft:
            return None
        raise ValueError(f"Star {name} is not part of a binary system.")

    # Extract the data
    row = load_binary(
        circumbinary=circumbinary,
        from_memory=True,
        rename_columns=True,
        verbose=False,
    ).loc[idx]

    # Return as a pandas DataFrame if requested
    if as_pandas:
        return row

    # Add metadata
    metadata = dict(DEFAULT_METADATA)
    metadata["circumbinary"] = circumbinary

    # Define the star system
    binary = binary_row_to_binary_star(row, source="binary", metadata=metadata)

    return binary


# =============================================================================
# AUXILIARY FUNCTIONS
# =============================================================================


def check_if_binary(
    star_name: str, exact_match: bool = False, verbose: bool = True
) -> Tuple[bool, bool, str, List[str], float]:
    """Check if a star is part of a binary system.

    Parameters
    ----------
    star_name : str
        Name of the star to check.
    exact_match : bool, optional. Default is False.
        If True, return `True` only if an exact match.
        If False, return `True` if a very (99%) close match is found.
    verbose : bool, optional. Default is True.
        If True, print messages.

    Returns
    -------
    Tuple[bool, str, List[str], float]
        is_binary : bool
            True if the star is part of a binary system.
        circumbinary : bool
            True if the binary system is circumbinary.
        idx : str
            Index of the found binary system.
        values : List[str]
            List of the values found.
        ratio : float
            Ratio of the match.
    """
    for circumbinary, col in product([True, False], [0, 1]):
        # 0: star1_name, 1: alternate_name
        series = load_binary(
            circumbinary=circumbinary,
            from_memory=True,
            rename_columns=False,
            clean=False,
            verbose=False,
        )[col]
        idx, values, ratio = find_best_match(
            series, name=star_name, parse=True
        )
        if ratio > 0.99:  # Found a binary system
            if exact_match and ratio < 1:
                if verbose:
                    print(f"Found a very close binary match in [{values}]")
                continue
            if verbose:
                print(f"Binary system found in {values}")
            # Check if multiple values
            if len(values) > 1:
                # In this case, it is probable we looked in
                # the alternate names and found that one of the alternate names
                # is the exact match. Nevertheless, we will check they all have
                # the same idx in index.
                if len(set(idx)) != 1:
                    raise ValueError(
                        "Multiple values found, but different indexes."
                    )
                return True, circumbinary, idx[0], values, ratio

            return True, circumbinary, idx, values, ratio
    if verbose:
        print(f"Star {star_name} is not part of a binary system.")
    return False, False, "", [], 0.0
