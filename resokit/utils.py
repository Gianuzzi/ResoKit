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

"""Module with internal utility functions for the ResoKit package."""

# =============================================================================
# FUNCTIONS
# =============================================================================


def __assert_module_imported(
    imported: bool, module_name: str, message: str = ""
):
    """
    Assert that the specified module is imported.

    Parameters
    ----------
    imported : bool
        Boolean indicating whether the module is imported.
    module_name : str
        Name of the module to check.
    message : str
        Error message to display if the module is not imported.
    """
    if not imported:
        raise ImportError(
            f"{module_name} is required for this function. {message}"
        )
