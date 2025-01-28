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

import pandas as pd
from io import StringIO

# =============================================================================
# CONSTANTS
# =============================================================================

# Define the sources
_DATASET_URLS = {
    "p": "https://lesia.obspm.fr/perso/philippe-thebault/plan_circ.txt",
    "s": "https://lesia.obspm.fr/perso/philippe-thebault/plan_bin500au.txt",
}

# =============================================================================
# FUNCTIONS
# =============================================================================


def read_multi_star(
    circumbinary: bool = False, inferr: bool = False
) -> pd.DataFrame:
    """Read the provided multi-star system dataset.

    Parameters
    ----------
    circumbinary : bool, optional
        If True, read the circumbinary dataset.
        If False, read the binary dataset.
        Default is False.
    inferr : bool, optional
        If True, read the inferred dataset. If False, read the observed dataset.
        Default is False.

    Returns
    -------
    pd.DataFrame
        DataFrame with the multi-star system data.
    """
    # Define the filename based on the circumbinary parameter
    if circumbinary:
        filename = "plan_circ.txt"
    else:
        filename = "plan_bin500au.txt"

    # Open the file to read the header and find where data starts
    with open(filename, "r") as f:
        # Read all lines
        lines = f.readlines()

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

    # Print the header and DataFrame for inspection
    print(f"Header Information:\n{header}")
    print("\nData:\n", data)

    # Return the DataFrame
    return data
