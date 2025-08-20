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

import pandas as pd

import pytest

import resokit.load as rio
from resokit.core import (
    StaticBinaryStar,
    StaticPlanet,
    StaticStar,
    StaticSystem,
)

# ============================================================================
# CONSTANTS
# ============================================================================


bin_syst = "kepler47"  # Binary system with 3 planets
simple_syst = "kepler11"  # Simple system with 6 planets


# # ============================================================================
# # TESTS
# # ============================================================================


@pytest.mark.usefixtures("load_eu_data", "load_binary_data")
class TestLoadSystem:

    load_function = {
        "eu": rio.from_eu,
        "nasa": rio.from_nasa,
    }

    @pytest.mark.parametrize("source", ["eu"])
    def test_load_binary_system(self, source: str):
        """Test load_system with a binary system."""
        syst = self.load_function[source](
            name=bin_syst, verbose=False, exact_match=False
        )

        # Assert types
        assert isinstance(syst, StaticSystem)
        assert isinstance(syst.star, StaticBinaryStar)
        for star in syst.star:
            assert isinstance(star, StaticStar)
        for planet in syst.planets:
            assert isinstance(planet, StaticPlanet)

        # Assert source
        assert syst.source_ == source

        # Assert binarity
        assert syst.is_binary_
        assert syst.is_circumbinary

        # Assert number of planets
        assert len(syst.planets) == 3

        # Assert names
        assert syst.star.name == "Kepler-47"
        assert syst.star.star0.name == "Kepler-47 A"
        assert syst.star.star1.name == "Kepler-47 B"

    @pytest.mark.parametrize("source", ["eu"])
    def test_load_binary_without_binary_system(self, source: str):
        """Test load_system with a binary system."""
        syst = self.load_function[source](
            name=bin_syst, verbose=False, exact_match=False, check_binary=False
        )

        # Assert types
        assert isinstance(syst, StaticSystem)
        assert isinstance(syst.star, StaticStar)
        for planet in syst.planets:
            assert isinstance(planet, StaticPlanet)

        # Assert source
        assert syst.source_ == source

        # Assert not binarity
        assert not syst.is_binary_

        # Assert number of planets
        assert len(syst.planets) == 3

        # Assert names
        assert syst.name == "Kepler-47 A"
        assert syst.star.name == "Kepler-47 A"

    @pytest.mark.parametrize("source", ["eu"])
    def test_load_simple_system(self, source: str, k11):
        """Test load_system with a simple system."""
        syst = self.load_function[source](
            name=simple_syst, verbose=False, exact_match=False
        )

        # Assert types
        assert isinstance(syst, StaticSystem)
        assert isinstance(syst.star, StaticStar)
        for planet in syst.planets:
            assert isinstance(planet, StaticPlanet)

        # Assert source
        assert syst.source_ == source

        # Assert binarity
        assert not syst.is_binary_

        # Assert number of planets
        assert len(syst.planets) == k11.npl

        # Assert names
        assert syst.star.name == "Kepler-11"


@pytest.mark.usefixtures("load_eu_data", "load_binary_data")
class TestStaticSystem:

    load_function = {
        "eu": rio.from_eu,
        "nasa": rio.from_nasa,
    }

    @pytest.mark.parametrize("name", ["Kepler-11 b", "b", 0])
    def test_static_system_get_planet(self, name):
        """Test StaticSystem class planet method."""
        syst = self.load_function["eu"](
            name=simple_syst, verbose=False, exact_match=False
        )

        # Test get planet
        pl1 = syst.planet("Kepler-11 b")

        # Assert types
        assert isinstance(pl1, StaticPlanet)

        # Assert is the same as the first planet
        assert pl1 == syst.planets[0]

    def test_static_system_get_item(self, k11):
        """Test StaticSystem class get_item method."""
        syst = self.load_function["eu"](
            name=simple_syst, verbose=False, exact_match=False
        )

        # Test get item
        per = syst.get_item("P")

        # Assert types
        assert isinstance(per, pd.Series)

        # Assert equal to base example
        pd.testing.assert_series_equal(
            per,
            k11.data["P"],
        )

    @pytest.mark.parametrize("source", ["eu"])
    def test_static_system_get_item_errors(self, source: str, k11):
        """Test StaticSystem class get_item with errors method."""
        syst = self.load_function[source](
            name=simple_syst, verbose=False, exact_match=False
        )

        # Test get item
        per = syst.get_item("P", error=True)

        # Assert types
        assert isinstance(per, pd.DataFrame)

        base = k11.data[["P", "P_err_min", "P_err_max"]]

        # Assert equal to base example
        if source == "eu":
            pd.testing.assert_frame_equal(
                per,
                base,
            )
        # else:
        #     base["P_err_min"] = k11.nasa_p_err_min
        #     base["P_err_max"] = k11.nasa_p_err_max
        #     pd.testing.assert_frame_equal(
        #         per,
        #         base,
        #     )

    @pytest.mark.parametrize("source", ["eu"])
    def test_static_system_get_wrong_item(self, source: str):
        """Test StaticSystem class get_item with errors method."""
        syst = self.load_function[source](
            name=simple_syst, verbose=False, exact_match=False
        )

        with pytest.raises(KeyError):
            syst.get_item("wrong_item", error=True)

    def test_static_system_period_ratios(self, k11):
        """Test StaticSystem class period_ratios method."""
        syst = self.load_function["eu"](
            name=simple_syst, verbose=False, exact_match=False
        )

        # Test get item
        perat = syst.period_ratios_

        # Assert types
        assert isinstance(perat, pd.DataFrame)

        # Assert equal to base example
        pd.testing.assert_frame_equal(
            perat,
            k11.perat,
        )

    def test_static_system_period_ratio(self, k11):
        """Test StaticSystem class period_ratios method."""
        syst = self.load_function["eu"](
            name=simple_syst, verbose=False, exact_match=False
        )

        # Test get item
        perat = syst.period_ratio()

        # Assert types
        assert isinstance(perat, pd.DataFrame)

        # Assert equal to base example
        pd.testing.assert_frame_equal(
            perat,
            k11.perat,
        )
