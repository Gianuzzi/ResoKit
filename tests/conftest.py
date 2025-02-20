#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2025, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================

import pathlib

import numpy as np

from pandas import DataFrame, Series

import pytest

# ============================================================================
# CONSTANTS
# ============================================================================

default_ss = Series(data=[1, 2, 3])
default_df = DataFrame(data=[[1, 2, 3], [4, 5, 6]])

# ============================================================================
# CONFIG
# ============================================================================


def pytest_configure(config):
    pytest.default_ss = default_ss
    pytest.default_df = default_df


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
def random_int_gen():
    def _random_int(size=1):
        rng = np.random.default_rng(seed=42)
        return rng.integers(1, 999999999, size=size)

    return _random_int


@pytest.fixture(scope="session")
def db_temp_path(tmp_path_factory):
    def make_temp_path(source="data"):
        fn = tmp_path_factory.mktemp("tmp") / f"{source}.csv"
        return fn

    return make_temp_path


@pytest.fixture(scope="session")
def mock_in_mem(session_mocker):
    # Define def ss and df
    def _mock_in_mem(which="all"):
        if which == "index":
            # Mock the variables
            session_mocker.patch(
                "resokit.datasets.databases._IN_MEMORY_INDEXES",
                {"eu": default_ss, "nasa": default_ss},
            )
        elif which == "parsed":
            # Mock the variables
            session_mocker.patch(
                "resokit.datasets.databases._IN_MEMORY_PARSED_INDEXES",
                {"eu": default_ss, "nasa": default_ss},
            )
        elif which == "data":
            session_mocker.patch(
                "resokit.datasets.databases._IN_MEMORY_DATASETS",
                {"eu": default_df, "nasa": default_df},
            )
        elif which == "fully":
            session_mocker.patch(
                "resokit.datasets.databases._IS_FULLY_STORED",
                {"eu": True, "nasa": True},
            )
        elif which == "all":
            _mock_in_mem(which="index")
            _mock_in_mem(which="data")
            _mock_in_mem(which="fully")
        else:
            raise ValueError(f"which={which} not recognized.")

    return _mock_in_mem
