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
from resokit.core import (
    StaticBinaryStar,
    StaticPlanet,
    StaticStar,
    StaticSystem,
)

# ============================================================================
# TESTS
# ============================================================================


class TestLoadSystem:
    load_function = {
        "eu": rio.load_system_from_eu,
        "nasa": rio.load_system_from_nasa,
    }
    bin_syst = "kepler47"  # Binary system with 3 planets
    simple_syst = "kepler11"  # Simple system with 6 planets

    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_binary_system(self, source: str):
        """Test load_system with a binary system."""
        syst = self.load_function[source](
            name=self.bin_syst, verbose=False, exact_match=False
        )

        # Assert types
        assert isinstance(syst, StaticSystem)
        assert isinstance(syst.star, StaticBinaryStar)
        for star in syst.star:
            assert isinstance(star, StaticStar)
        for planet in syst.planets:
            assert isinstance(planet, StaticPlanet)

        # Assert source
        if source == "eu":
            assert syst.source_ == source
        else:
            assert syst.source_ == "eu_and_nasa"

        # Assert binarity
        assert syst.is_binary_
        assert syst.is_circumbinary

        # Assert number of planets
        assert len(syst.planets) == 3

        # Assert names
        assert syst.star.name == "Kepler47"
        if source == "eu":
            assert syst.star.star0.name == "Kepler-47 A"
        else:
            assert syst.star.star0.name == "Kepler-47"

        assert syst.star.star1.name == "Kepler47 B"

    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_simple_system(self, source: str):
        """Test load_system with a simple system."""
        syst = self.load_function[source](
            name=self.simple_syst, verbose=False, exact_match=False
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
        assert len(syst.planets) == 6

        # Assert names
        assert syst.star.name == "Kepler-11"
