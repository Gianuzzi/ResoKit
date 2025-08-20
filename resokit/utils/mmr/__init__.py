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

"""The ResoKit.utils.mmr package includes diverse tools for MMRs analysis."""

# =============================================================================
# IMPORTS
# =============================================================================

from .mmrs import closest_mmr3b  # noqa
from .mmrs import label_mmr3b  # noqa
from .mmrs import mindist_mmr3b  # noqa
from .mmrs import mmr3b  # noqa
from .mmrs import mmrs_in_area  # noqa
from .mmrs import plot_mmrs  # noqa

# Make the functions available at the package level

__all__ = [
    "closest_mmr3b",
    "label_mmr3b",
    "mindist_mmr3b",
    "mmr3b",
    "mmrs_in_area",
    "plot_mmrs",
]
