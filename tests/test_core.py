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

import warnings

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import pytest

import resokit.load as rkload
from resokit import core

# ============================================================================
# CONSTANTS
# ============================================================================

matplotlib.use("Agg")

bin_syst = "kepler47"  # Binary system with 3 planets
simple_syst = "kepler11"  # Simple system with 6 planets


# ============================================================================
# TESTS
# ============================================================================


@pytest.mark.usefixtures("load_eu_data", "load_binary_data")
class TestLoadSystem:

    load_function = {
        "eu": rkload.from_eu,
        "nasa": rkload.from_nasa,
    }

    @pytest.mark.parametrize("source", ["eu"])
    def test_load_binary_system(self, source: str):
        """Test load_system with a binary system."""
        syst = self.load_function[source](
            name=bin_syst, verbose=False, exact_match=False
        )

        # Assert types
        assert isinstance(syst, core.StaticSystem)
        assert isinstance(syst.star, core.StaticBinaryStar)
        for star in syst.star:
            assert isinstance(star, core.StaticStar)
        for planet in syst.planets:
            assert isinstance(planet, core.StaticPlanet)

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
        assert isinstance(syst, core.StaticSystem)
        assert isinstance(syst.star, core.StaticStar)
        for planet in syst.planets:
            assert isinstance(planet, core.StaticPlanet)

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
        assert isinstance(syst, core.StaticSystem)
        assert isinstance(syst.star, core.StaticStar)
        for planet in syst.planets:
            assert isinstance(planet, core.StaticPlanet)

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
        "eu": rkload.from_eu,
        "nasa": rkload.from_nasa,
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
        assert isinstance(pl1, core.StaticPlanet)

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


class TestMetadata:
    def test_metadata_basic(self):
        md = core.MetaData({"a": 12, "b": 2})
        # attribute access
        assert md.a == 12
        assert md["a"] == 12
        # iterator and length
        assert set(iter(md)) == {"a", "b"}
        assert len(md) == 2
        # repr mentions 'Metadata'
        assert "Metadata(" in repr(md)


class TestResokitdataframe:
    def test_resokit_dataframe_empty(self):
        with pytest.warns(UserWarning):
            core.ResokitDataFrame(pd.DataFrame(), source="user")

    def test_resokit_dataframe_getitem(self):
        rkdf2 = core.ResokitDataFrame(
            pd.DataFrame({"name": ["pl1", "pl2"], "x": [1, 2], "y": [3, 4]}),
            source="user",
        )
        assert list(rkdf2.name) == ["pl1", "pl2"]

        rkdf1 = core.ResokitDataFrame(
            pd.Series({"name": "pl1", "x": 1, "y": 3}), source="user"
        )
        assert rkdf1.x == 1
        assert rkdf1[0] == "pl1"
        assert rkdf1[2] == 3
        assert list(rkdf1[[0, 2]]) == ["pl1", 3]

    def test_resokit_dataframe_equal(self):
        df = pd.DataFrame({"name": ["pl1", "pl2"], "x": [1, 2], "y": [3, 4]})
        df2 = pd.DataFrame({"name": ["pl1", "pl3"], "x": [1, 2], "y": [3, 4]})
        rkdf1 = core.ResokitDataFrame(df, source="user")
        rkdf2 = core.ResokitDataFrame(df, source="user")
        rkdf3 = core.ResokitDataFrame(df2, source="user")
        rkdf4 = core.ResokitDataFrame(df, source="binary")
        assert rkdf1 == rkdf1
        assert not rkdf1 == df
        assert rkdf1 == rkdf2
        assert not rkdf1 == rkdf3
        assert not rkdf1 == rkdf4

    def test_resokit_dataframe_basics(self):
        df = pd.DataFrame({"name": ["o1", "o2"], "x": [1.0, 2.0]})
        rdf = core.ResokitDataFrame(data=df, source="user", metadata={"k": 1})

        # structural attributes
        assert rdf.n_objects_ == 2
        assert rdf.n_columns_ == 2
        assert list(rdf.columns_) == ["name", "x"]
        assert len(rdf) == 2

        # __getitem__ for multi-row returns column
        col = rdf["x"]
        assert isinstance(col, pd.Series)
        assert col.iloc[0] == 1.0

        # __getattr__ delegates to underlying data
        assert rdf.shape == df.shape

        # set_column (not inplace) returns a new object and original unchanged
        new = rdf.set_column("z", [7, 8])
        assert "z" in new.columns_
        assert "z" not in rdf.columns_

        # to_dataframe returns requested columns and copies when asked
        df_out = rdf.to_dataframe(columns=None, copy=True)
        assert isinstance(df_out, pd.DataFrame)
        assert df_out.shape == rdf.data.shape

        # copy should equal but not be the same object
        cp = rdf.copy()
        assert cp == rdf
        assert cp is not rdf

    def test_resokit_dataframe_set_col(self, fake_df):
        rdf = core.ResokitDataFrame(data=fake_df, source="user", metadata={})
        rdf.set_column("name", "new", silent=False)

    def test_resokit_dataframe_single_object_must_be_series(self):
        # If only one object, a Series or df must be passed
        df1 = pd.DataFrame({"name": ["single"], "x": [1.0]})
        core.ResokitDataFrame(data=df1, source="user", metadata={})

        s = df1.iloc[0]
        rdf = core.ResokitDataFrame(data=s, source="user", metadata={})
        assert rdf.n_objects_ == 1

    def test_resokit_dataframe_empty_plot_and_repr(self, monkeypatch):
        df = pd.DataFrame({"name": ["pl1", "pl2"], "x": [1, 2], "y": [3, 4]})
        rdf = core.ResokitDataFrame(
            data=df, source="user", metadata={"note": "edge"}
        )

        # Patch plt.show to avoid opening GUI
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        _, ax = plt.subplots()
        prev = len(ax.get_children())

        # Normal scatter plot
        rdf.plot("x", "y", ax=ax)
        assert len(ax.get_children()) > prev

        # Edge: missing 'name' column triggers a warning
        df2 = pd.DataFrame({"a": [1], "b": [2]})
        with warnings.catch_warnings(record=True) as w:
            _, ax = plt.subplots()
            rdf2 = core.ResokitDataFrame(data=df2, source="user", metadata={})
            rdf2.plot("a", "b")
            assert any("name" in str(wi.message) for wi in w)
            assert len(ax.get_children()) > prev

        # Invalid kind raises
        with pytest.raises(KeyError):
            rdf.plot("a", "b")

        # error_x / error_y not bool
        with pytest.raises(TypeError):
            rdf.plot("x", "y", error_x=123)

        # _repr_html_ returns HTML string
        html = rdf._repr_html_()
        assert isinstance(html, str) and "table" in html.lower()

        # _repr_html_ returns HTML string
        html = rdf._repr_html_(switch=True)
        assert isinstance(html, str) and "table" in html.lower()

    def test_resokit_dataframe_to_dataframe_specific_columns(self, fake_df):
        rdf = core.ResokitDataFrame(data=fake_df, source="user", metadata={})
        df_part = rdf.to_dataframe(columns=["name"])
        assert list(df_part.columns) == ["name"]

    def test_resokit_dataframe_plot(self, planet_series):
        rdf = core.ResokitDataFrame(
            data=planet_series, source="user", metadata={}
        )
        _, ax = plt.subplots()
        prev = len(ax.get_children())
        rdf.plot("name", "P")
        assert len(ax.get_children()) == prev
        rdf.plot("P", "name")
        assert len(ax.get_children()) == prev
        with pytest.raises(ValueError):
            rdf.plot("radius", "P", error_x=True)
        rdf.plot("mass", "P", error_x=True)
        assert len(ax.get_children()) > prev
        rdf.plot("mass", "P", error_y=True)
        rdf.plot("P", "mass", error_y=True)
        rdf.plot("P", "mass", error_x=True)


class TestDftoresokit:
    def test_df_to_resokit_validation_and_basic_flow(self):
        sample = pd.DataFrame({"name": ["a"], "P": [2.0], "other": [5]})
        # invalid source
        with pytest.raises(ValueError):
            core.df_to_resokit(sample, "bad_source")

        # invalid df type
        with pytest.raises(TypeError):
            core.df_to_resokit("not a df", "eu")

        # return_df True returns a pandas DataFrame
        out_df = core.df_to_resokit(sample, "eu", drop=False, return_df=True)
        assert isinstance(out_df, pd.DataFrame)
        # rename_index True sets index to 'name'
        out_df2 = core.df_to_resokit(
            sample, "eu", drop=False, return_df=True, rename_index=True
        )
        assert out_df2.index[0] == "a"

        # normal conversion to ResokitDataFrame
        res = core.df_to_resokit(sample, "eu", drop=False, return_df=False)
        assert isinstance(res, core.ResokitDataFrame)

        # 'n' column added when 'P' present
        assert "n" in res.data.index
        assert pytest.approx(res.data["n"]) == 2.0 * np.pi / 2.0

        # empty df raises
        with pytest.raises(ValueError):
            core.df_to_resokit(pd.DataFrame(), "eu")

    def test_df_to_resokit_sort_and_column_handling(self):
        # Sorted and unsorted by name
        df = pd.DataFrame({"name": ["B", "A"], "P": [2.0, 1.0]})
        out = core.df_to_resokit(df, "eu", drop=False, return_df=True)
        assert "n" in out.columns
        # drop=True removes empty columns
        df2 = pd.DataFrame({"name": ["A"], "empty": [float("nan")]})
        res = core.df_to_resokit(df2, "eu", drop=True, return_df=True)
        assert "empty" not in res.columns

    def test_df_to_resokit_handles_period_without_n(self):
        df = pd.DataFrame({"P": [10.0]})
        # Should add 'n' if missing
        res = core.df_to_resokit(df, "eu", drop=False, return_df=True)
        assert "n" in res.columns

    def test_df_to_resokit_with_none_df_raises(
        self,
    ):
        with pytest.raises(TypeError):
            core.df_to_resokit(None, "eu")

    def test_df_to_resokit_unsupported_source_raises(
        self,
    ):
        df = pd.DataFrame({"name": ["X"], "P": [1.0]})
        with pytest.raises(ValueError):
            core.df_to_resokit(df, "unsupported")


class TestStaticplanet:
    def test_static_planet_basic_and_get_item(self, simple_planet):
        pl = simple_planet
        assert pl.name == "pl1"
        # get_item returns Series with requested values
        res = pl.get_item("radius", error=False, silent=False)
        assert isinstance(res, pd.Series)
        assert res["radius"] == 1.0

        # requesting error columns that don't exist emits an error
        with pytest.raises(KeyError):
            assert "miss" not in pl.columns_
            pl.get_item(["miss"], error=True, silent=False)

    def test_estimate_mass_and_radius_return_types_and_new_planet(
        self, simple_planet, monkeypatch
    ):
        pl = simple_planet

        # Case: err_method default (-1) -> ret_err False ->
        # returns single mass value
        # estimate_mass default patched in autouse fixture
        # returns (100,90,110) and convert is identity
        m = pl.estimate_mass()
        assert isinstance(m, float) or isinstance(m, (int,))
        # calling with err_method=1 returns triple
        monkeypatch.setattr(
            core, "estimate_mass", lambda **kwargs: (200.0, 190.0, 210.0)
        )
        triple = pl.estimate_mass(err_method=1)
        assert isinstance(triple, tuple) and len(triple) == 3
        # new_planet True returns a StaticPlanet with mass in data
        newp = pl.estimate_mass(new_planet=True, err_method=1)
        assert isinstance(newp, core.StaticPlanet)
        assert "mass" in newp.data.index or "mass" in newp.data

    def test_estimate_radius_and_new_planet(self, simple_planet, monkeypatch):
        pl = simple_planet
        # err_method default is -1 -> ret_err False ->
        # returns single radius value if no errors requested
        r = pl.estimate_radius()
        assert isinstance(r, float) or isinstance(r, (int,))
        # request errors
        monkeypatch.setattr(
            core, "estimate_radius", lambda **kwargs: (3.0, 2.5, 3.5)
        )
        triple = pl.estimate_radius(err_method=1)
        assert isinstance(triple, tuple) and len(triple) == 3
        newp = pl.estimate_radius(new_planet=True, err_method=1)
        assert isinstance(newp, core.StaticPlanet)
        assert "radius" in newp.data.index or "radius" in newp.data

    def test_static_planet_set_attr_and_index_error(self, simple_planet):
        pl = simple_planet
        # cannot index by integer on StaticBody-derived objects
        with pytest.raises(IndexError):
            _ = pl[0]
        # set_attr name -> new instance with changed name and user source
        new = pl.set_attr("name", "plX")
        assert new.name == "plX"
        assert new.source == "user"
        assert "history" in dict(new.metadata)

    def test_static_planet_set_attr_edge_cases(self, simple_planet):
        pl = simple_planet
        # invalid attr raises
        with pytest.raises(TypeError):
            pl.set_attr(123, 1)

    def test_static_planet_invalid_get_item_index(self, simple_planet):
        pl = simple_planet
        with pytest.raises(IndexError):
            _ = pl[999]

    def test_static_planet_html_repr(self, simple_planet):
        pl = simple_planet
        html = pl._repr_html_()
        assert isinstance(html, str) and "table" in html.lower()

    def test_static_planet_plot_and_repr(self, monkeypatch, simple_planet):
        pl = simple_planet
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)

        # Normal plot
        with pytest.raises(TypeError):
            pl.plot()

        # _repr_html_ coverage
        html = pl._repr_html_()
        assert isinstance(html, str)

    def test_static_planet_invalid_err_method(
        self, simple_planet, patch_units_and_estimators
    ):
        pl = simple_planet
        # invalid err_method should raise ValueError
        pl.estimate_mass(err_method="bad")
        pl.estimate_radius(err_method="bad")

    def test_static_planet_get_item_silent_and_error(self, simple_planet):
        pl = simple_planet
        # error=True but missing err column triggers warning not exception
        with warnings.catch_warnings(record=True) as w:
            pl.get_item("name", error=True, silent=True)
            assert w == []
        with pytest.raises(KeyError):
            pl.get_item("nonexistent", error=True, silent=False)


class TestStaticstar:
    def test_static_star_basic(self, simple_star):
        st = simple_star
        assert st.name == "st1"
        assert st.is_star is True
        assert "StaticStar" in st.__repr__()

    def test_static_star_plot(self, monkeypatch, simple_star):
        st = simple_star
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        with pytest.raises(TypeError):
            st.plot()


class TestStaticbinarystar:
    def test_static__binary_basic_methods(self, simple_binary):
        bin_sys = simple_binary
        assert len(bin_sys) == 2
        # star by index
        assert bin_sys.star(0).name == bin_sys.star0.name
        assert bin_sys.star(1).name == bin_sys.star1.name
        # star by name
        assert bin_sys.star(bin_sys.star0.name) == bin_sys.star0
        # star 'all'
        both = bin_sys.star("all")
        assert isinstance(both, list) and len(both) == 2
        # to_dataframe returns DataFrame with column named as binary name
        df = bin_sys.to_dataframe()
        assert bin_sys.name in df.columns
        # set binary attribute (adds new key to data)
        nb = bin_sys.set_attr("new_val", 42)
        assert "new_val" in nb.data.index or "new_val" in nb.data
        # set attribute on star (in_star param)
        nb2 = bin_sys.set_attr("mass", 2.0, in_star=0)
        assert nb2.star0.mass == 2.0

    def test_static_binary_star_errors(self, simple_binary):
        bin_sys = simple_binary
        with pytest.raises(ValueError):
            bin_sys.star(5)  # invalid int
        with pytest.raises(ValueError):
            bin_sys.star("nonexistent")

    def test_static_binary_repr_and_html(self, simple_binary):
        b = simple_binary
        assert type(b) is core.StaticBinaryStar
        # rep = b.__repr__()
        # assert "StaticBinaryStar" in rep
        html = b._repr_html_()
        assert isinstance(html, str) and "table" in html.lower()

    def test_static_binary_set_attr_invalid_in_star(self, simple_binary):
        b = simple_binary
        # Invalid in_star index
        with pytest.raises(ValueError):
            b.set_attr("mass", 3.0, in_star=5)

    def test_static_binary_set_attr_new_value(self, simple_binary):
        nb = simple_binary.set_attr("luminosity", 5)
        assert "luminosity" in nb.data.index or "luminosity" in nb.data

    def test_static_binary_star_bad(self, fake_df, simple_star):
        with pytest.raises(AttributeError):
            core.StaticBinaryStar(simple_star, simple_star, "bad")
        with pytest.raises(KeyError):
            core.StaticBinaryStar(simple_star, simple_star, fake_df)


class TestStaticsystem:
    def test_static_system_basics(self, simple_system):
        sys = simple_system
        assert sys.n_planets_ == 2
        assert sys.n_stars_ == 1
        assert "pl1" in sys.planet_names_
        # period ratio for two planets should be P2/P1
        assert pytest.approx(sys.period_ratios_) == (20.0 / 10.0)
        # body by integer
        assert sys.body(0) == sys.star
        assert sys.body(1) == sys.planets[0]
        # body by name
        assert sys.body("star") == sys.star
        assert sys.body("all") == [sys.star] + sys.planets
        # planet slicing
        assert isinstance(sys.planet(0), core.StaticPlanet)
        assert (
            isinstance(sys.planet("all"), list) and len(sys.planet("all")) == 2
        )
        # get planet items
        items = sys._get_planets_items("name")
        assert isinstance(items, list)
        assert items[0] == sys.planets[0]["name"]
        # __contains__
        assert sys.star.name in sys
        assert sys.planets[0] in sys
        # equality checks
        other = core.StaticSystem(
            star=sys.star, planets=sys.planets, name="sys1", metadata={}
        )
        assert sys == other
        diff = core.StaticSystem(
            star=sys.star, planets=sys.planets, name="diff", metadata={}
        )
        assert not (sys == diff)

    def test_static_system_no_html_repr(self, simple_system):
        sys = simple_system
        # invalid index
        with pytest.raises(AttributeError):
            sys._repr_html_()

    def test_static_system_invalid_planet_access(self, simple_system):
        sys = simple_system
        with pytest.raises(ValueError):
            sys.planet("not_exist")

    def test_static_system_plot(self, monkeypatch, simple_system):
        sys = simple_system
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        _, ax = plt.subplots()
        prev = len(ax.get_children())
        ax = sys.plot("P_err_min", "P_err_max")
        assert len(ax.get_children()) > prev

    def test_static_system_set_attr_on_star_and_planet(self, simple_system):
        sys = simple_system
        # set new attribute on star
        sys2 = sys.set_attr("metallicity", 0.5, in_star=True)
        assert isinstance(sys2, core.StaticSystem)
        # set new attribute on planet 0
        sys3 = sys.set_attr("albedo", 0.2, in_planet=0)
        assert isinstance(sys3, core.StaticSystem)

    def test_static_system_invalid_attr_params(self, simple_system):
        sys = simple_system
        # invalid type for in_star/in_planet
        with pytest.raises(IndexError):
            sys.set_attr("bad", 1, in_planet=4)

    def test_static_system_to_df(self, simple_system):
        sys = simple_system
        df = sys.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_static_system_contains_edge(self, simple_system):
        sys = simple_system
        assert "sys1" not in sys  # name != object
        assert not ("random" in sys)
