#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2025, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# This file indicates that the directory should be treated as a package.

# =============================================================================
# DOCS
# =============================================================================

"""The ResoKit.units package includes tools for unit manipulation."""

# =============================================================================
# IMPORTS
# =============================================================================

from .units import MKS, UNITS, convert  # noqa

__all__ = ["MKS", "UNITS", "convert"]
