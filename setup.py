#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2024, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE


# =============================================================================
# DOCS
# =============================================================================

"""This file is for distribute and install ResoKit."""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import pathlib

from setuptools import setup  # noqa

# =============================================================================
# CONSTANTS
# =============================================================================

REQUIREMENTS = [
    "attrs",
    "numpy",
    "matplotlib",
    "pandas",
    "scipy",
]

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

with open(PATH / "README.md") as fp:
    LONG_DESCRIPTION = fp.read()

with open(PATH / "resokit" / "__init__.py") as fp:
    for line in fp.readlines():
        if line.startswith("__version__ = "):
            VERSION = line.split("=", 1)[-1].replace('"', "").strip()
            break


DESCRIPTION = "Miscellaneous tools for the analysis of planetary systems."


# =============================================================================
# FUNCTIONS
# =============================================================================


def do_setup():
    setup(
        name="resokit",
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author=[
            "Emmanuel Gianuzzi",
            "Matías Cerioni",
        ],
        author_email="egianuzzi@unc.edu.ar",
        url="https://github.com/Gianuzzi/resokit",
        packages=[
            "resokit",
            "resokit.datasets",
            "resokit.io",
            "resokit.utils",
        ],
        include_package_data=True,
        license="MIT",
        install_requires=REQUIREMENTS,
        extras_require={
            "query": ["requests", "astropy", "beautifulsoup4"],
            "reb": ["rebound"],
        },
        keywords=["resokit", "planetary systems", "resonances"],
        classifiers=[
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering",
        ],
    )


if __name__ == "__main__":
    do_setup()
