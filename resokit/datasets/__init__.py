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

"""The ResoKit.datasets module includes tools for loading datasets."""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path

from .databases import clear_memory  # noqa
from .databases import download_binary_dataset  # noqa
from .databases import download_dataset  # noqa
from .databases import load_binary_dataset  # noqa
from .databases import load_dataset  # noqa
from .databases import check_binary_outdated  # noqa
from .databases import check_outdated  # noqa
from .utils import DATASETS_DIR  # noqa

# =============================================================================
# DIRECTORY SETUP
# =============================================================================

Path(DATASETS_DIR).mkdir(parents=True, exist_ok=True)


# Make the functions available at the package level

__all__ = [
    "clear_memory",
    "download_binary_dataset",
    "download_dataset",
    "load_binary_dataset",
    "load_dataset",
    "check_binary_outdated",
    "check_outdated",
]
