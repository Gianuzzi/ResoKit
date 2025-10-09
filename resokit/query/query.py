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

"""Module to query exoplanet.eu and NASA datasets with optimized structure."""

# =============================================================================
# IMPORTS
# =============================================================================

from io import BytesIO
from typing import Union

import pandas as pd

import requests

from resokit.core import (
    ResokitDataFrame,
    StaticSystem,
    df_to_resokit,
    resokit_to_system,
)
from resokit.utils.parser import DEFAULT_METADATA, assert_module_imported


try:
    from astropy.table import Table

    astropy_imported = True
except ImportError:
    astropy_imported = False

# =============================================================================
# CONSTANTS
# =============================================================================

# Query URLs for the two datasets
QUERY_URL = {
    "eu": "http://voparis-tap-planeto.obspm.fr/tap/sync",
    "nasa": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
}


# =============================================================================
# DYNAMIC
# =============================================================================

_session_queries = {}  # This dict will store queries for the session

# =============================================================================
# FUNCTIONS
# =============================================================================


def build_query(
    source: str,
    select: str = "*",
    alias: str = "",
    conditions: str = "",
    order_by: str = "",
) -> str:
    """Construct a very simple query for the specified dataset source.

    For more information on how to write the conditions, visit:
    - EU information: http://voparis-tap-planeto.obspm.fr/__system__/dc_tables/show/tableinfo/exoplanet.epn_core
    - NASA information: https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html#sync-query

    Parameters
    ----------
    source : str
        Data source identifier ('eu' or 'nasa').
    select : str, optional. Default: '*'.
        Columns to select in the query (default is '*').
    alias : str, optional. Default: ''.
        Optional alias for the table or columns.
    conditions : list of str, optional. Default: ''.
        Multiple (list) conditions are grouped with 'AND' clause.
        If 'OR' is needed, it must be written explicitly.
        List of conditions for WHERE clause.
    order_by : str, optional. Default: ''.
        Column name for ORDER BY clause.

    Returns
    -------
    str
        Constructed query string.
    """  # noqa: E501
    source = source.lower()  # Ensure lowercase

    # SELECT clause
    if not isinstance(select, str):
        raise ValueError("Select must be a string.")

    # Construct the query
    query = f"SELECT {select} "

    # FROM clause
    if alias:
        if not isinstance(alias, str):
            raise ValueError("Alias must be a string.")
        query += f"AS {alias} "

    # Add the source table
    query += "FROM ps" if source == "nasa" else "FROM exoplanet.epn_core"

    # WHERE clause
    if conditions:
        if isinstance(conditions, str):
            query += f" WHERE {conditions}"
        else:
            where_conditions = [f"({condition})" for condition in conditions]
            query += f" WHERE {' AND '.join(where_conditions)}"

    # ORDER BY clause
    if order_by:
        if not isinstance(order_by, str):
            raise ValueError("Order by must be a string.")
        query += f" ORDER BY {order_by}"

    return query


def execute_query(
    source: str,
    query: str,
    cache: bool = True,
    to_bytes: bool = False,
    verbose: bool = True,
    soft: bool = False,
) -> Union[bytes, pd.DataFrame]:
    """Execute a query on the specified dataset source.

    This function attempts to follow all TAPs.
    For more information on how to write the conditions, visit:
    - EU information: http://voparis-tap-planeto.obspm.fr/__system__/dc_tables/show/tableinfo/exoplanet.epn_core
    - NASA information: https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html#sync-query

    Parameters
    ----------
    source : str
        Data source identifier ('eu' or 'nasa').
    query : str
        Query string to execute.
    cache : bool, optional. Default: True.
        Cache the query and result (dataframe) in case of repetition
        during this session.
    to_bytes: bool, optional. Default: False
        Return bytes instead of pandas dataframe.
    verbose : bool, optional. Default: True.
        Print messages
    soft : bool, optional. Default = False.
        Whether to perform a query that do not includes a
        WHERE statement. This is not recommended, as it would download
        the full databases. Use `resokit.datasets.download(...)` for this.

    Returns
    -------
    Union[bytes, pd.DataFrame]
        Resulting dataset as a pandas DataFrame, or bytes if requested.
    """  # noqa: E501
    source = source.lower()  # Ensure lowercase

    if source not in QUERY_URL:
        raise ValueError(f"Unknown data source: {source}")

    # Define the query URL
    url = QUERY_URL[source]

    # Check if WHERE is included
    if "WHERE" not in query.upper():
        if not soft:
            raise ValueError("Expected a WHERE clause in the query.")
        if verbose:
            print("Warning: No WHERE clause in the query.")

    # Define the query parameters
    params = {
        "query": query,
    }
    if source == "nasa":
        params["format"] = "csv"
    else:
        global astropy_imported
        astropy_imported = assert_module_imported(
            astropy_imported, "astropy", "(Not needed for NASA)"
        )
        params["lang"] = "ADQL"

    # Construct the full query URL (used as the cache key)
    req = requests.Request("GET", url, params=params).prepare()
    query_url = req.url  # This is the full URL with encoded params

    if query_url in _session_queries:
        if verbose:
            print("Using cached previous identic query.")
        if not isinstance(_session_queries[query_url], pd.DataFrame):
            print("Error. Last query result was not a dataframe.")
            print(" Deleting cached query. Retry if necessary.")
            raise ValueError(
                "Expected previous query result as dataframe, but"
                + f" got {type(_session_queries[query_url])} instead."
            )
        if to_bytes:
            buffer = BytesIO()
            _session_queries[query_url].to_csv(buffer)
            buffer.seek(0)
            return buffer.getvalue()
        return _session_queries[query_url]

    # Default result
    result = pd.DataFrame()

    try:  # Execute the query
        if verbose:
            print("Executing the query...")
        response = requests.get(url, params=params)
        response.raise_for_status()

        if source == "nasa":  # Parse CSV response
            result = pd.read_csv(BytesIO(response.content))
        else:  # Parse VOTable response
            result = Table.read(BytesIO(response.content)).to_pandas()

        if verbose:
            aux = " , but returned no data" if result.empty else ""
            print(f"Query executed successfully{aux}.")

    except requests.RequestException as e:
        print(f" Error querying {source} database: {e}")

    if cache:
        _session_queries.update({query_url: result})

    return result


def query_system(
    source: str,
    star_name: str = "",
    planet_name: str = "",
    default_flag: int = 1,
    controversial_flag: int = 0,
    cache: bool = True,
    verbose: bool = True,
    as_frame: bool = False,
    raw: bool = False,
) -> Union[pd.DataFrame, ResokitDataFrame, StaticSystem]:
    """Query the online dataset based on specified filters.

    Parameters
    ----------
    source : str
        Data source identifier ('eu' or 'nasa').
    star_name : str, optional. Default: ''.
        Host star or system name.
    planet_name : str, optional. Default: ''.
        Planet name.
    default_flag : int, optional. Default: 1.
        Restrict to default values in NASA dataset.
        If equal to 0, only returns non default parameter set for each planet.
        If equal to 1, only returns default parameter set for each planet.
        If None, all planet sets (default and not) are returned.
    controversial_flag : int, optional. Default: 0.
        Restrict to controversial planets in NASA dataset.
        If equal to 0, only returns confirmed planets.
        If equal to 1, only returns controversial planets.
        If None, all planets (confirmed and not) are returned.
    cache : bool, optional. Default: True.
        Cache the query and result in case of repetition
        during this session.
    verbose : bool, optional. Default: True.
        Print query information.
    as_frame : bool, optional. Default: False.
        Whether to return the dataset in a `ResokitDataFrame`
        format (True), or a `StaticSystem` format (False).
    raw : bool, optional. Default = False.
        Whether to return a pandas DataFrame of the query result.
        Overrides `as_frame` parameter.

    Returns
    -------
    data : DataFrame, ResokitDataFrame or StaticSystem
        Results of the query in a :py:class:`ResokitDataFrame`
        (if `as_resokit=True`), or :py:class:`StaticSystem`,
        or :py:class:`pandas.DataFrame` (if `raw=True`).
    """
    if not planet_name and not star_name:
        raise ValueError(
            "Either 'planet_name' or 'star_name' must be provided."
        )

    if planet_name and star_name:
        raise ValueError(
            "Only one of 'planet_name' or 'star_name' can be provided."
        )

    # Define the target or star field based on the source
    if source not in ["eu", "nasa"]:
        raise ValueError("Invalid source. Must be 'eu' or 'nasa'.")

    if source == "eu":
        field_name = "target_name" if planet_name else "star_name"
        global astropy_imported
        astropy_imported = assert_module_imported(
            astropy_imported, "astropy", "(Not needed for NASA)"
        )
    else:
        field_name = "pl_name" if planet_name else "hostname"

    filter_value = star_name or planet_name  # Get the filter value

    # Build the query
    query = build_query(source, conditions=[f"{field_name}='{filter_value}'"])

    # Add default_flag condition for NASA source
    if default_flag and source == "nasa":
        query += " AND default_flag=1"

    # Add controversial_flag condition for NASA source
    if controversial_flag is not None and source == "nasa":
        query += f" AND pl_controv_flag={controversial_flag}"

    # Print query information
    if verbose:
        print(f" Querying {source} database with query: {query}")

    # Execute query and get results
    df = execute_query(source=source, query=query)

    # Cache result
    if cache:
        _session_queries.update({query: df})

    # Get raw?
    if raw:
        return df

    # Convert to ResoKit format
    # Note: Metadata is set from default values
    meta = DEFAULT_METADATA.copy()
    meta.update({"query": query})

    reso = df_to_resokit(  # Convert to ResoKit DataFrame
        df=df,
        source=source,
        drop=False,
        copy=False,
        metadata=meta,
    )

    if not as_frame:  # Return StaticSystem
        return resokit_to_system(reso)

    return reso  # Return ResoKit DataFrame
