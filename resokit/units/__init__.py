#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2024, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# This file indicates that the directory should be treated as a package.

# =============================================================================
# DOCS
# =============================================================================

"""The ResoKit.units module includes tools for unit manipulation (m|Kg|seg)."""

# =============================================================================
# IMPORTS
# =============================================================================

from .units import *  # noqa

import sys  # noqa

__all__ = []

# Add everything from .units dynamically, if available
if hasattr(sys.modules[__name__ + ".units"], "__all__"):
    __all__.extend(sys.modules[__name__ + ".units"].__all__)
else:
    __all__.extend(
        [
            name
            for name in dir(sys.modules[__name__ + ".units"])
            if not name.startswith("_")
        ]
    )
