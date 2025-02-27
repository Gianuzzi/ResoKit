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

import pytest

import resokit.io as rio
from resokit.core import StaticBinaryStar, StaticSystem
from resokit.datasets import databases

# ============================================================================
# TESTS
# ============================================================================


class TestLoadSystem:
    load_function = {
        "eu": rio.load_system_from_eu,
        "nasa": rio.load_system_from_nasa,
    }

    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_system_wrong(self, source: str, capfd):
        """Test load_system with wrong system."""
        # Test load_system with wrong system
        with pytest.raises(ValueError, match="Star wrong_system not found in"):
            self.load_function[source](name="wrong_system", verbose=False)

        # Ensure the verbose output is correct
        out, _ = capfd.readouterr()
        assert out == ""

        # Now with verbose=True
        with pytest.raises(ValueError, match="Star wrong_system not found in"):
            self.load_function[source](name="wrong_system", verbose=True)

        # Ensure the verbose output is correct
        out, _ = capfd.readouterr()
        assert "Star wrong_system not found" in out
        if source == "eu":
            assert "Note: ExoplanetEU has alternative" in out

    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_system_wrong_soft(self, source: str, capfd):
        """Test load_system with wrong system, soft=True."""
        # Test load_system with wrong system
        syst = self.load_function[source](
            name="wrong_system", verbose=True, soft=True
        )

        # syst is a DataFrame
        assert syst is None

        # Ensure the verbose output is correct
        out, err = capfd.readouterr()
        assert "Star wrong_system not found" in out
        assert err == ""
        if source == "eu":
            assert "Note: ExoplanetEU has alternative" in out

    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_system_almost(self, source: str, capfd):
        """Test load_system with almost correct system."""
        syst = self.load_function[source](
            name="kepler11", verbose=False, soft=True
        )

        assert syst is None

        # Ensure the verbose output is correct
        out, err = capfd.readouterr()
        assert out == ""
        assert err == ""

        # Now with verbose=True
        syst = self.load_function[source](
            name="kepler11", verbose=True, soft=True
        )

        assert syst is None

        # Ensure the verbose output is correct
        out, err = capfd.readouterr()
        assert "Looking for star system 'kepler11'" in out
        assert "Found a very close star match:" in out
        assert "Execute with exact_match=False to load it" in out
        assert err == ""

    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_system_almost_not_exact(self, source: str):
        """Test load_system with almost correct system."""
        # Clear the memory
        databases.clear_memory("all", verbose=False)

        syst = self.load_function[source](name="kepler11", exact_match=False)

        assert isinstance(syst, StaticSystem)
        assert syst.n_planets_ == 6

    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_store_system(self, source: str, mocker):
        """Test load_system and store_system."""
        # Assert nothing is pre-stored
        databases.clear_memory("all", verbose=False)

        # -----------------------------------
        # Store nothing
        # -----------------------------------

        # Load the system
        syst = self.load_function[source](
            name="kepler11",
            verbose=False,
            exact_match=False,
            store_index=False,
            store=False,
        )

        # Assert the system is loaded
        assert isinstance(syst, StaticSystem)
        assert syst.n_planets_ == 6

        # Assert nothing is stored
        assert databases._IN_MEMORY_DATASETS[source].empty
        assert databases._IN_MEMORY_INDEXES[source].empty
        assert not databases._IS_FULLY_STORED[source]

        # -----------------------------------
        # Store the index
        # -----------------------------------

        # Load the system and store the index
        syst = self.load_function[source](
            name="kepler11",
            verbose=False,
            exact_match=False,
            store_index=True,
            store=False,
        )

        # Assert the system is loaded
        assert isinstance(syst, StaticSystem)
        assert syst.n_planets_ == 6

        # Assert just the index is stored
        assert databases._IN_MEMORY_DATASETS[source].empty
        assert not databases._IN_MEMORY_INDEXES[source].empty
        assert not databases._IS_FULLY_STORED[source]

        # Clear the memory
        databases.clear_memory("both", verbose=False)

        # -----------------------------------
        # Store the whole dataset
        # -----------------------------------

        # Load the system and store just this system
        syst = self.load_function[source](
            name="kepler11",
            verbose=False,
            exact_match=False,
            store_index=False,
            store=True,
        )

        # Assert the system is loaded
        assert isinstance(syst, StaticSystem)
        assert syst.n_planets_ == 6

        # Assert a partial dataset is stored (The index too)
        assert not databases._IN_MEMORY_DATASETS[source].empty
        assert not databases._IN_MEMORY_INDEXES[source].empty
        assert not databases._IS_FULLY_STORED[source]

        # Assert only this system was stored
        if source == "eu":
            assert databases._IN_MEMORY_DATASETS[source].shape[0] == 6
        else:
            # Nasa has 97 total solution rows for planets in this system
            assert databases._IN_MEMORY_DATASETS[source].shape[0] == 97

    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_from_stored(self, source: str, mocker):
        """Test load_system with stored datasets."""
        # Assert nothing is pre-stored
        databases.clear_memory("all", verbose=False)

        # Load the system and store the whole dataset
        syst = self.load_function[source](
            name="kepler11",
            verbose=False,
            exact_match=False,
            store_index=False,
            store=True,
        )

        # Assert the system is loaded
        assert isinstance(syst, StaticSystem)
        assert syst.n_planets_ == 6

        # Assert the whole dataset is stored (The index too)
        assert not databases._IN_MEMORY_DATASETS[source].empty
        assert not databases._IN_MEMORY_INDEXES[source].empty
        assert not databases._IS_FULLY_STORED[source]

        # -----------------------------------
        # Read from the stored dataset
        # -----------------------------------

        # Mock the zip_path
        mocker.patch("resokit.datasets.utils.ZIP_FILENAME", "wrong_name")

        # Load a second system
        syst = self.load_function[source](
            name="kepler47",
            verbose=False,
            exact_match=False,
            store_index=False,
            store=True,
            low_memory=True,
        )

        # Assert the system is loaded
        assert isinstance(syst, StaticSystem)
        assert syst.n_planets_ == 3

        # Clear the memory
        databases.clear_memory("both", verbose=False)


class TestLoadBinary:
    def test_load_binary_wrong(self, capfd):
        """Test load_binary with wrong system."""
        # Assert nothing is pre-stored
        databases.clear_memory("all", verbose=False)

        with pytest.raises(ValueError, match="Star wrong_system not found"):
            rio.load_from_binary(name="wrong_system", verbose=False)
        # Capture the output
        out, _ = capfd.readouterr()
        # Ensure the verbose output is correct
        assert out == ""

        # Now with verbose=True
        with pytest.raises(ValueError, match="Star wrong_system not found"):
            rio.load_from_binary(name="wrong_system", verbose=True)

        # Ensure the verbose output is correct
        out, _ = capfd.readouterr()
        assert "Star wrong_system is not part " in out

    def test_load_binary_wrong_soft(self, capfd):
        """Test load_binary with wrong system, soft=True."""
        # Assert nothing is pre-stored
        databases.clear_memory("all", verbose=False)

        syst = rio.load_from_binary(
            name="wrong_system", verbose=True, soft=True
        )

        # syst is a DataFrame
        assert syst is None

        # Ensure the verbose output is correct
        out, err = capfd.readouterr()
        assert "Star wrong_system is not part " in out
        assert err == ""

    def test_load_binary_almost(self, capfd):
        """Test load_binary with almost correct system."""
        # Assert nothing is pre-stored
        databases.clear_memory("all", verbose=False)

        syst = rio.load_from_binary(name="kepler47", verbose=False, soft=True)

        assert syst is None

        # Ensure the verbose output is correct
        out, err = capfd.readouterr()
        assert out == ""
        assert err == ""

        # Now with verbose=True
        syst = rio.load_from_binary(name="kepler47", verbose=True, soft=True)

        assert syst is None

        # Ensure the verbose output is correct
        out, err = capfd.readouterr()
        assert "Looking for star system 'kepler47'" in out
        assert "Found a very close binary match" in out
        assert "Execute with exact_match=False to load it" in out
        assert err == ""

    def test_load_binary_almost_not_exact(self):
        """Test load_binary with almost correct system."""
        # Assert nothing is pre-stored
        databases.clear_memory("all", verbose=False)

        syst = rio.load_from_binary(name="kepler47", exact_match=False)

        assert isinstance(syst, StaticBinaryStar)
        assert syst.name == "Kepler47"
