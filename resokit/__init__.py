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
resonances (MMR) in coplanar planetary systems."
"""

__version__ = "0.1.0"


# =============================================================================
# IMPORTS
# =============================================================================

# from . import tools  # noqa
from . import core  # noqa
from . import datasets  # noqa
from . import io  # noqa
from . import utils  # noqa
from .utils import mmr  # noqa

# =============================================================================
# FUNCTIONS
# =============================================================================


def hello():
    """Print the hello message."""
    print("Hello from the resokit package!")
