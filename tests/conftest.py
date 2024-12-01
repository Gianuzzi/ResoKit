#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2024, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================

import pathlib

import numpy as np

import pytest

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def zip_path():
    return (
        pathlib.Path(__file__).parents[1]
        / "resokit"
        / "datasets"
        / "datasets.zip"
    )


@pytest.fixture(scope="session")
def skip_rows():
    return {"nasa": 291, "eu": 0}


@pytest.fixture(scope="session")
def index_cols():
    return {"nasa": ["pl_name", "hostname"], "eu": ["name", "star_name"]}


@pytest.fixture(scope="session")
def random_int():
    rng = np.random.default_rng(seed=42)
    return rng.integers(low=1)
