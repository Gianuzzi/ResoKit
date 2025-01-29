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
"""ResoKit.

ResoKit addresses the need for diagnosing and analyzing  mean motion
resonances (MMR) in coplanar planetary systems.
"""

__version__ = "0.0.1"


# =============================================================================
# IMPORTS
# =============================================================================

from . import core  # noqa
from . import datasets  # noqa
from . import io  # noqa
from . import utils  # noqa
from . import units  # noqa
from .utils import mmr  # noqa
from .utils import mass_radius  # noqa

# Make the core classes available directly from the package.

__all__ = ["core", "datasets", "io", "units", "utils", "mmr", "mass_radius"]
