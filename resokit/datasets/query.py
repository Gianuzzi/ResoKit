# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2024, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Module to query exoplanet.eu and NASA datasets with optimized structure."""

# =============================================================================
# IMPORTS
# =============================================================================

from io import BytesIO
import pandas as pd
from resokit.utils import __assert_module_imported
from resokit.io.io import _convert_to_resokit_format

try:
    import requests

    requests_imported = True
except ImportError:
    requests_imported = False

try:
    from astropy.table import Table
    from astropy.io.votable import parse_single_table

    astropy_imported = True
except ImportError:
    astropy_imported = False

# =============================================================================
# CONSTANTS
# =============================================================================

QUERY_URL = {
    "eu": "http://voparis-tap-planeto.obspm.fr/tap/sync?lang=ADQL&",
    "nasa": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?",
}

# =============================================================================
# FUNCTIONS
# =============================================================================


def _build_query(
    source: str,
    select: str = "*",
    alias: str = "",
    conditions: str = "",
    order_by: str = "",
):
    """
    Construct a query for the specified dataset source.

    Parameters
    ----------
    source : str
        Data source identifier ('eu' or 'nasa').
    select : str, optional
        Columns to select in the query (default is '*').
    alias : str, optional
        Optional alias for the table or columns.
    conditions : list of str, optional
        List of conditions for WHERE clause.
    order_by : str, optional
        Column name for ORDER BY clause.

    Returns
    -------
    str
        Constructed query string.
    """
    source = source.lower()

    if not isinstance(select, str):
        raise ValueError("Select must be a string.")

    query = f"SELECT {select} "
    if alias:
        if not isinstance(alias, str):
            raise ValueError("Alias must be a string.")
        query += f"AS {alias} "

    query += "FROM ps" if source == "nasa" else "FROM exoplanet.epn_core"

    # WHERE clause
    if conditions:
        where_conditions = [f"({condition})" for condition in conditions]
        query += f" WHERE {' AND '.join(where_conditions)}"

    # ORDER BY clause
    if order_by:
        if not isinstance(order_by, str):
            raise ValueError("Order by must be a string.")
        query += f" ORDER BY {order_by}"

    return query


def _execute_query(query: str, source: str):
    """
    Execute a query on the specified dataset source.

    Parameters
    ----------
    query : str
        Query string to execute.
    source : str
        Data source identifier ('eu' or 'nasa').

    Returns
    -------
    pd.DataFrame
        Resulting dataset as a pandas DataFrame.
    """
    __assert_module_imported(requests_imported, "requests")
    source = source.lower()
    url = QUERY_URL[source]
    query_url = (
        "query="
        + query.replace(" ", "+")
        + ("&format=csv" if source == "nasa" else "")
    )

    try:
        response = requests.get(url + query_url)
        response.raise_for_status()

        if source == "nasa":
            return pd.read_csv(BytesIO(response.content))
        else:
            return Table.read(BytesIO(response.content)).to_pandas()

    except requests.RequestException as e:
        print(f"Error querying {source.upper()} database: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


def query_exoplanet_data(
    source: str,
    star_name: str = "",
    planet_name: str = "",
    default_flag: int = 1,
    controversial_flag: int = 0,
    to_resokit: bool = True,
):
    """
    Query the exoplanet dataset based on specified filters.

    Parameters
    ----------
    source : str
        Data source identifier ('eu' or 'nasa').
    star_name : str, optional
        Host star or system name.
    planet_name : str, optional
        Planet name.
    default_flag : int, optional
        Restrict to default values in NASA dataset (default is 1).
    controversial_flag : int, optional
        Restrict to controversial planets in NASA dataset (default is 0).
    to_resokit : bool, optional
        Convert the dataset to ResoKit format (default is True).

    Returns
    -------
    pd.DataFrame
        Resulting dataset as a pandas DataFrame.
    """
    __assert_module_imported(requests_imported, "requests")
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
        __assert_module_imported(
            astropy_imported, "astropy", "Not needed for NASA."
        )
    else:
        field_name = "pl_name" if planet_name else "hostname"
    filter_value = star_name or planet_name
    query = _build_query(source, conditions=[f"{field_name}='{filter_value}'"])

    # Add default_flag condition for NASA source
    if default_flag and source == "nasa":
        query += " AND default_flag=1"
    if controversial_flag is not None and source == "nasa":
        query += f" AND pl_controv_flag={controversial_flag}"

    print(f"Querying {source.upper()} database with query: {query}")

    # Execute query and get results
    df = _execute_query(query, source)

    if not to_resokit or df.empty:
        return df

    # Convert to ResoKit format
    return _convert_to_resokit_format(df, source=source, drop=True)


def get_dataset_length(source: str):
    """
    Query the length (count) of the dataset from the specified source.

    Parameters
    ----------
    source : str
        Data source identifier ('eu' or 'nasa').

    Returns
    -------
    int
        Number of entries in the dataset.
    """
    __assert_module_imported(requests_imported, "requests")
    source = source.lower()
    if source == "nasa":
        query = "query=SELECT+COUNT(*)+FROM+ps&format=csv"
    elif source == "eu":
        query = "query=SELECT+COUNT(*)+FROM+exoplanet.epn_core"
        __assert_module_imported(
            astropy_imported, "astropy", "Not needed for NASA."
        )
    else:
        raise ValueError("Invalid source. Must be 'eu' or 'nasa'.")

    url = QUERY_URL[source]

    try:
        response = requests.get(url + query)
        response.raise_for_status()

        # For NASA, parse CSV response
        if source == "nasa":
            return int(response.text.splitlines()[1])

        # For EU, parse as VOTable
        votable = parse_single_table(BytesIO(response.content))
        return int(votable.array[0][0])

    except requests.RequestException as e:
        print(f"Error querying {source.upper()} dataset length: {e}")
        return 0  # Return 0 on error
