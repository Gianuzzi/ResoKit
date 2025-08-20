#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2025, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# This file indicates that the directory should be treated as a package.

# ============================================================================
# DOCS
# ============================================================================

"""The ResoKit.query package includes tools to make online queries."""

# =============================================================================
# IMPORTS
# =============================================================================


from .query import build_query, execute_query, query_system  # noqa

# Make the functions available at the package level

__all__ = [
    "build_query",
    "execute_query",
    "query_system",
]
