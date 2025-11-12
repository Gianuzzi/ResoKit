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

"""Module with planetary systems load functions for the ResoKit package."""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
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
from resokit.datasets.databases import load, load_binary
from resokit.utils.parser import DEFAULT_METADATA, find_best_match
from resokit.utils.utils import calc_period

# =============================================================================
# FUNCTIONS
# =============================================================================


# --------------------------- EU and NASA -------------------------------------


def _search_system_index(
    source: str,
    name: str,
    is_planet: bool = False,
    raw_df: Union[pd.DataFrame, None] = None,
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
        Whether to search in the alternative names column.
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
    parse = True  # True means name and raw_series are parsed
    not_parsed = None  # Will be stored and parsed next time
    # Load the dataset if not in memory
    # Search in the main column?
    if not alternative_names and raw_df is not None:  # Use the raw dataset
        raw_series = raw_df[column]  # Get the column
    elif not alternative_names:
        # Update the keyword arguments
        parsed = load(
            source=source,
            **{**load_kwargs, "only_index": "parsed", "verbose": False},
        )  # Load the parsed dataset (if it is in memory)
        if parsed is not None:  # Use the parsed dataset
            parse = None  # None mean parse only the name
            raw_series = parsed  # Because raw_series is already parsed
        else:  # Load the whole dataset
            raw_series = load(
                source=source,
                **load_kwargs,
            )  # Will be stored and parsed next time
        raw_series = raw_series[column]  # Get the column
    else:  # Search in the alternate names
        not_parsed = load(
            source=source,
            **{**load_kwargs, "only_index": False, "verbose": False},
        )  # Load the whole dataset (worst scenario)
        raw_series = not_parsed[column].str.split(", ").explode()

    # Use the new function
    index, _, ratio = find_best_match(
        raw_series=raw_series, name=name, parse=parse, force=is_planet
    )

    # We have to get back the original values
    # If parse is None, then we have to compute the non parsed
    if not_parsed is None:
        not_parsed = load(
            source=source,
            **{**load_kwargs, "only_index": False, "verbose": False},
        )
    # Get the original values
    original_values = not_parsed[column].loc[index].tolist()

    # Downgrade ratio to account for exact matches
    ratio = ratio * 0.99

    # Redefine ratio for possible exact match
    if original_values[0] == name:
        ratio = 1

    return index, original_values, ratio


def _from_db(
    name: str,
    source: str,
    is_planet: bool = False,
    file_path: Union[str, Path, bool] = True,
    store: bool = False,
    store_index: bool = True,
    verbose: bool = True,
    low_memory: bool = True,
    alternative_names: bool = False,
    exact_match: bool = False,
    check_binary: bool = True,
) -> Tuple[pd.DataFrame, str, int]:
    """Load system from ExoplanetEU or NASA.

    Parameters
    ----------
    name : str
        System/planet name.
    source : str
        Source of the dataset. Either 'eu' or 'nasa'.
    is_planet : bool, optional. Default: False.
        Whether to search for a planet or a star.
    file_path : str, Path, bool, optional. Default: True
        Path to the file to load the dataset.
        If `True`, default filename is used.
        If `False`, the file is not loaded.
    store : bool, optional. Default: False.
        Whether to store the whole dataset in memory.
    store_index : bool, optional. Default: True.
        Whether to store the whole dataset index in memory.
        Automatically set to True if store is True.
    verbose : bool, optional. Default: True.
        Whether to print information.
    low_memory : bool, optional. Default: True.
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
        data: Loaded system as a DataFrame.
        binary: Tuple with the binary information. If the system is a binary
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
            f"Looking for {'planet' if is_planet else 'star system'} '{name}' "
            + f"in {source} database."
        )

    # If storing, then load the whole dataset
    if store:
        store_index = True  # Store the index if the dataset will be stored
        # low_memory = False  # Load the whole dataset if it will be stored

    # Check if alternative names are available
    if alternative_names:
        if source != "eu":
            raise ValueError(
                "Alternative names only available in ExoplanetEU dataset."
            )
        if verbose:
            print(" Searching for alternative names.")

    # Hard work: Define from_zip and from_file
    from_file = file_path
    from_zip = True
    if isinstance(file_path, (str, Path)):
        file_path = Path(file_path)
        if file_path.name.endswith("zip"):
            from_zip = file_path
            from_file = True
        elif file_path.resolve().parent.name.endswith("zip"):
            from_zip = False

    # Define the keyword arguments
    load_kwargs = {
        "store": store,
        "verbose": verbose,
        "store_index": store_index,
        "to_resokit": False,
        "only_rows": None,
        "only_index": False,
        "to_df": True,
        "from_file": from_file,
        "from_zip": from_zip,
    }

    # Load the dataset
    if not low_memory:  # Load the whole dataset
        raw_df = load(source=source, **load_kwargs)
    else:  # Will load only the index if possible
        raw_df = None

    # Search for the system
    idx, values, ratio = _search_system_index(
        source=source,
        name=name,
        is_planet=is_planet,
        raw_df=raw_df,
        alternative_names=alternative_names if source == "eu" else False,
        **{**load_kwargs, "only_index": True, "store": False},  # Not store yet
    )

    auxmsg = "alternate names column of " if alternative_names else ""
    # Check if the system was found
    if ratio < 0.98:  # To take into account the almost 1 ratio
        if verbose and is_planet:
            print(f"Planet {name} not found in {auxmsg}{source} dataset.")
        elif verbose:
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
            if ratio > 0.5:  # Only if ratio is high enough
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
                f" Found a very close {pl} match: '{values[0]}' "
                + f"in {auxmsg}{source} dataset."
            )
        if exact_match:  # Return an empty DataFrame
            if verbose:
                print(
                    " Execute with exact_match=False to load it, "
                    + "or rewrite the name."
                )
            return pd.DataFrame(), "n", -1  # Return an empty DataFrame
        # We will load the system with the almost exact match
        if verbose:
            print(" Loading the almost exact match...")

    # In case duplicated entries (due to alternate nemes used), we use the
    # list of the set of idx.
    idx = list(set(idx))

    # Load the system
    if raw_df is None:  # Load only the system data
        data = load(source=source, **{**load_kwargs, "only_rows": idx})
    else:
        data = raw_df.loc[idx]  # Load the system data from the raw dataset

    # Check if the system is a binary system?
    is_binary = False  # Default: not a binary system
    circumbinary = False  # Default: not a circumbinary system
    binary_type = "f"  # Default: not a binary system
    idxbin = -1  # Default: not a binary system
    if check_binary:  # Check if binary
        star_name_col = "star_name" if source == "eu" else "hostname"
        star_name = data[star_name_col].iloc[0]  # Get the (first) star name
        if verbose:
            print(f"Checking if '{star_name}' is a binary system...")
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


def from_eu(
    name: str,
    is_planet: bool = False,
    file_path: Union[str, Path, bool] = True,
    drop: bool = True,
    store: bool = True,
    store_index: bool = True,
    verbose: bool = True,
    low_memory: bool = False,
    as_resokit: bool = False,
    alternative_names: bool = False,
    exact_match: bool = True,
    check_binary: Union[bool, None] = True,
    soft: bool = False,
) -> Union[ResokitDataFrame, StaticSystem, None]:
    """Load system from ExoplanetEU.

    Parameters
    ----------
    name : str
        System/planet name.
        (Remember case sensitivity)
    is_planet : bool, optional. Default: False.
        Whether to search for a planet or a star.
    file_path : str, Path, bool, optional. Default: True
        Path to the file to load the dataset.
        If `True`, default filename is used.
        If `False`, the file is not loaded.
    drop : bool, optional. Default: True.
        Whether to drop extra columns.
    store : bool, optional. Default: True.
        Whether to store the whole dataset in memory.
    store_index : bool, optional. Default: True.
        Whether to store the whole dataset index in memory.
        Automatically set to True if store is True.
    verbose : bool, optional. Default: True.
        Whether to print information.
    low_memory : bool, optional. Default: False.
        Whether to avoid loading the whole dataset into memory.
    as_resokit : bool, optional. Default: False.
        Whether to return the dataset in ResoKit format.
    alternative_names : bool, optional. Default: False.
        Whether to search for alternative names.
    exact_match : bool, optional. Default: True.
        Whether to search for an exact match. If `True`
        `verbose=True`, suggestions will be printed in case
        of no exact match. If `False`, the search will be more
        flexible, and a very (very) similar name will be accepted.
        Useful for names with different characters (e.g., hyphens), or
        for names with extra information (e.g., "A" or "B").
    check_binary : bool, optional. Default: True.
        Whether to check if the system is a binary system.
        If it is a binary system indeed, then the final system
        created is a `StaticBinarySystem` instead of a `StaticSystem`.
        If `None`, the check will be performed only to print
        information (if `verbose=True`).
    soft : bool, optional. Default: False.
        If True, return None if the system is not found.
        If False, raise an error if the system is not found.

    Returns
    -------
    system : ResokitDataFrame or StaticSystem
        Loaded system as :py:class:`ResokitDataFrame` (if `as_resokit=True`),
        or :py:class:`StaticSystem`.
    """
    # Load the system from the database
    df, bin_type, _ = _from_db(
        name=name,
        source="eu",
        is_planet=is_planet,
        file_path=file_path,
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
        if soft:
            return None
        obj = "Planet" if is_planet else "Star"
        raise ValueError(f"{obj} {name} not found in ExoplanetEU database.")

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
        return_df=False,
        metadata=meta,
    )

    if as_resokit:  # Return ResoKit DataFrame
        return reso

    # Return StaticSystem
    if bin_type in ["p", "s"]:  # We have to create StaticBinaryStar
        binary = from_binary(
            name=name,
            exact_match=exact_match,
            as_pandas=False,
            soft=False,
            add_period=True,
            verbose=False,
        )
        return resokit_to_system(
            reso,
            binary_star=binary,
            circumbinary=bin_type == "p",
            verbose=verbose,
        )

    return resokit_to_system(reso, verbose=verbose)  # Return StaticSystem


def from_nasa(
    name: str,
    is_planet: bool = False,
    file_path: Union[str, Path, bool] = True,
    drop: bool = True,
    store: bool = True,
    store_index: bool = True,
    verbose: bool = True,
    low_memory: bool = False,
    controversial_set: Union[bool, None] = False,
    default_set: Union[bool, None] = True,
    as_resokit: bool = False,
    exact_match: bool = True,
    check_binary: Union[bool, None] = True,
    soft: bool = False,
) -> Union[ResokitDataFrame, StaticSystem, None]:
    """Load system from NASA.

    Parameters
    ----------
    name : str
        System/planet name.
        (Remember case sensitivity)
    is_planet : bool, optional. Default: False.
        Whether to search for a planet or a star.
    file_path : str, Path, bool, optional. Default: True
        Path to the file to load the dataset.
        If `True`, default filename is used.
        If `False`, the file is not loaded.
    drop : bool, optional. Default: True.
        Whether to drop extra columns.
    store : bool, optional. Default: True.
        Whether to store the whole dataset in memory.
    store_index : bool, optional. Default: True.
        Whether to store the whole dataset index in memory.
        Automatically set to True if store is True.
    verbose : bool, optional. Default: True.
        Whether to print information.
    low_memory : bool, optional. Default: False.
        Whether to avoid loading the whole dataset into memory.
    controversial_set : bool, None, optional. Default: False.
        Whether to include controversial data.
        None to include all data.
    default_set : bool, None, optional. Default: True.
        Whether to include default data.
        None to include all data.
    as_resokit : bool, optional. Default: False.
        Whether to return the dataset in ResoKit format.
        If the output is not a single system, a ResoKitDataframe
        will be returned.
    exact_match : bool, optional. Default: True.
        Whether to search for an exact match. If `True`
        `verbose=True`, suggestions will be printed in case
        of no exact match. If `False`, the search will be more
        flexible, and a very (very) similar name will be accepted.
        Useful for names with different characters (e.g., hyphens), or
        for names with extra information (e.g., "A" or "B").
    check_binary : bool, optional. Default: True.
        Whether to check if the system is a binary system.
        If it is a binary system indeed, then the final system
        created is a `StaticBinarySystem` instead of a `StaticSystem`.
        If `None`, the check will be performed only to print
        information (if `verbose=True`).
    soft : bool, optional. Default: False.
        If True, return None if the system is not found.
        If False, raise an error if the system is not found.

    Returns
    -------
    system : ResokitDataFrame or StaticSystem
        Loaded system as :py:class:`ResokitDataFrame` (if `as_resokit=True`),
        or :py:class:`StaticSystem`.
    """
    df, bin_type, _ = _from_db(
        name=name,
        source="nasa",
        is_planet=is_planet,
        file_path=file_path,
        store=store,
        store_index=store_index,
        verbose=verbose,
        low_memory=low_memory,
        exact_match=exact_match,
        check_binary=check_binary or check_binary is None,
    )

    # Check if the dataset is empty
    if df.empty:
        if soft:
            return None
        obj = "Planet" if is_planet else "Star"
        raise ValueError(f"{obj} {name} not found in NASA database.")

    # Filter controversial and/ or defalut data
    single_syst = controversial_set is False and default_set is True
    if controversial_set is not None or default_set is not None:
        if controversial_set is not None:
            df = df[df["pl_controv_flag"] == int(controversial_set)]
        if default_set is not None:
            df = df[df["default_flag"] == int(default_set)]
        # Check if empty after filtering
        if df.empty:
            if soft:
                return None
            obj = "Planet" if is_planet else "Star"
            raise ValueError(
                f"{obj} {name} not found in NASA database, "
                + "after filtering with "
                + f"{controversial_set=} and {default_set=}."
            )
        # In this case, there is no such thing as a "system", because
        # each planet solution may be independant from other. So, we just
        # return all solutions as a DataFrame.
        if verbose and not single_syst:
            print(
                "Multiple solutions found for the search."
                + " Returning all solutions.\n"
                + " Binary systems are not supported in this case."
            )

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

    if as_resokit or not single_syst:  # Return ResoKit DataFrame
        # Add system set in the case of multiple solutions
        if not single_syst and not is_planet:
            values = pd.factorize(reso["reference"])[0]
            reso.set_column("solution_set", values, silent=True, inplace=True)
        return reso

    # Return StaticSystem
    if bin_type in ["p", "s"]:  # We have to create StaticBinaryStar
        binary = from_binary(
            name=name,
            exact_match=exact_match,
            as_pandas=False,
            soft=False,
            add_period=True,
            verbose=False,
        )
        return resokit_to_system(
            reso,
            binary_star=binary,
            circumbinary=bin_type == "p",
            verbose=verbose,
        )

    return resokit_to_system(reso, verbose=verbose)  # Return StaticSystem


# --------------------------- Binary Stars ------------------------------------


def from_binary(
    name: str,
    exact_match: bool = True,
    as_pandas: bool = False,
    soft: bool = False,
    add_period: bool = True,
    verbose: bool = True,
    rename: Union[str, None] = None,
) -> StaticBinaryStar:
    """Load a binary star system from the dataset.

    Parameters
    ----------
    name : str
        Name of the binary star system to load.
    exact_match : bool, optional. Default is True.
        If True, return the exact match only.
        If False, return the best match.
    as_pandas : bool, optional. Default is False.
        If True, return the data as a pandas DataFrame.
    soft : bool, optional. Default is False.
        If True, return None if the star is not found.
        If False, raise an error if the star is not found.
    add_period : bool, optional. Default is True.
        If True, add the period of the binary system.
    verbose : bool, optional. Default is True.
        If True, print messages.
    rename:  Union[str, None], optional. Default is None.
        If not None, set this value as the name of the stars.

    Returns
    -------
    StaticBinaryStar
        The loaded binary star system.
    """
    # Print information
    if verbose:
        print(f"Looking for star system '{name}' in binary datasets.")

    # Check if the star is part of a binary system
    is_binary, circumbinary, idx, _, _ = check_if_binary(
        star_name=name, exact_match=exact_match, verbose=verbose
    )

    if not is_binary:
        if soft:
            return None
        raise ValueError(f"Star '{name}' not found in binary datasets.")

    # Extract the data
    row = load_binary(
        which=circumbinary,
        from_memory=True,
        rename_columns=True,
        verbose=False,
    )
    assert isinstance(row, pd.DataFrame), (
        "Expected row to be a DataFrame, " + f"got {type(row)} instead."
    )
    row = row.loc[idx]  # Get the row with the index

    # Add the period
    if add_period:
        row["P"] = calc_period(
            row["a"], row["star0_mass"] + row["star1_mass"], 0.0
        )

    # Rename the stars if requested
    if rename is not None:
        row["star0_name"] = str(rename)
        # Just the first, because then binary_row... will rename the second

    # Return as a pandas DataFrame if requested
    if as_pandas:
        return row

    # Add metadata
    metadata = dict(DEFAULT_METADATA)
    metadata["circumbinary"] = circumbinary

    # To create the binary star system, we need a Series
    row = row.squeeze()

    # Be sure that is a pandas Series
    if not isinstance(row, pd.Series):
        raise ValueError("A problem occurred while loading the binary system.")

    # Define the star system
    binary = binary_row_to_binary_star(row, source="binary", metadata=metadata)

    return binary


# =============================================================================
# AUXILIARY FUNCTIONS
# =============================================================================


def check_if_binary(
    star_name: str,
    exact_match: bool = True,
    verbose: bool = True,
    soft: bool = True,
) -> Tuple[bool, bool, str, List[str], float]:
    """Check if a star is part of a binary system.

    Parameters
    ----------
    star_name : str
        Name of the star to check.
    exact_match : bool, optional. Default is True.
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
    maybe = False
    df = pd.DataFrame()
    for circumbinary in [True, False]:
        which = "p" if circumbinary else "s"
        try:
            df = load_binary(
                which=which,
                from_memory=True,
                rename_columns=False,
                clean=False,
                verbose=False,
            )
        except FileNotFoundError as error:
            maybe = True
            if verbose:
                print(
                    f" Unable to check for {which}-tpye binary orbits."
                    + " Txt file not found.\n"
                    + " Try downloading with "
                    + "resokit.datasets.download_binary"
                    + f"_dataset({which=}, to_file=True)"
                )
            if circumbinary:  # Try both...
                continue
            if soft:
                return False, False, "", [], 0.0
            else:
                raise error
        # 0: star0_name, 1: alternate_name
        for col in [0, 1]:
            series = df[col]
            assert isinstance(series, pd.Series), (
                "Expected series to be a pd.Series, "
                + f"got {type(series)} instead."
            )
            idx, values, ratio = find_best_match(
                series, name=star_name, parse=True
            )
            if ratio > 0.99:  # Found a binary system
                maybe = True
                if exact_match and ratio < 1:
                    if verbose:
                        print(f" Found a very close binary match in: {values}")
                        print(" Execute with exact_match=False to load it.")
                    break
                if verbose:
                    print(
                        f" Binary system found in {values}, "
                        + f"in {which}-type binary orbit."
                    )
                # Check if multiple values
                if len(values) > 1:
                    # In this case, it is probable we looked in
                    # the alternate names and found that one of
                    # the alternate names is the exact match.
                    # Nevertheless, we will check they all have
                    # the same idx in index.
                    if len(set(idx)) != 1:
                        raise ValueError(
                            "Multiple values found, but different indexes."
                        )
                    return True, circumbinary, idx[0], values, ratio

                return True, circumbinary, idx, values, ratio

    if verbose:
        aux = "could be" if maybe else "is not"
        print(f"Star {star_name} {aux} part of a binary system.")

    return False, False, "", [], 0.0
