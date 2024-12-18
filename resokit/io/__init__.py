#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2024, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# This file indicates that the directory should be treated as a package.

# ============================================================================
# DOCS
# ============================================================================

"""The ResoKit.io module includes tools for input/output operations."""

# =============================================================================
# IMPORTS
# =============================================================================

from .io import load_system_from_eu  # noqa
from .io import load_system_from_nasa  # noqa
from .query import query_online  # noqa

# Make the functions available at the package level

__all__ = [
    "load_system_from_eu",
    "load_system_from_nasa",
    "query_online",
]
