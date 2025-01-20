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

"""The ResoKit.datasets module includes tools for loading datasets."""

# =============================================================================
# IMPORTS
# =============================================================================

from .databases import clear_memory  # noqa
from .databases import download  # noqa
from .databases import load_eu  # noqa
from .databases import load_nasa  # noqa
from .databases import update_eu  # noqa
from .databases import update_nasa  # noqa

# Make the functions available at the package level

__all__ = [
    "clear_memory",
    "download",
    "load_eu",
    "load_nasa",
    "update_eu",
    "update_nasa",
]
