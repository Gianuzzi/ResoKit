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

from fractions import Fraction

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images

from numpy import isnan

import pandas as pd

import pytest

import resokit.utils as rutils
from resokit import core
from resokit.load import from_eu

# ============================================================================
# CONSTANTS
# ============================================================================

bin_syst = "kepler47"  # Binary system with 3 planets
simple_syst = "kepler11"  # Simple system with 6 planets

# ============================================================================
# TESTS
# ============================================================================


class TestFloatToFraction:

    golden = (1.0 + 5 ** (0.5)) * 0.5  # Golden number

    def test_no_arguments(self):
        """Test with missing arguments."""

        with pytest.raises(
            ValueError,
            match="At least one of max_iter or max_error "
            + "or stop_func must be set.",
        ):
            rutils.float_to_fraction(1)

    def test_golden_max_error0(self, capfd):
        """Test results with golden number."""

        result = rutils.float_to_fraction(self.golden, max_error=0.0)
        assert result == (377, 233)

        # Capture the output
        out, _ = capfd.readouterr()
        # Ensure the verbose output is correct
        assert "Maximum number of iterations reached: 12" in out

    def test_golden_max_iter10_max_error0(self, capfd):
        """Test results with golden number."""

        result = rutils.float_to_fraction(
            self.golden, max_iter=10, max_error=0.0
        )
        assert result == (89, 55)

        # Capture the output
        out, _ = capfd.readouterr()
        # Ensure the verbose output is correct
        assert out.endswith(
            " Iter 10:  89/55  -> 1.618182 (error: 9.14e-05)\n"
        )

    def test_golden_as_fraction(self, capfd):
        """Test results with golden number."""

        result = rutils.float_to_fraction(
            self.golden, max_iter=10, max_error=0.0, as_fraction=True
        )
        assert result == Fraction(89, 55)

        # Capture the output
        out, _ = capfd.readouterr()
        # Ensure the verbose output is correct
        assert out.endswith(
            " Iter 10:  89/55  -> 1.618182 (error: 9.14e-05)\n"
        )


class TestCalcA:

    def test_estimate_a_naive(self, k11):
        """Test StaticSystem estimate a."""
        syst = from_eu(name=simple_syst, verbose=False, exact_match=False)
        a_ss = syst.estimate_semi_major_axis()

        # Assert types
        assert isinstance(a_ss, pd.Series)

        # Assert equal to base example
        pd.testing.assert_series_equal(
            a_ss,
            k11.estimated_a,
            check_names=False,  # Names are different
            rtol=1e-3,
            atol=1e-4,
        )

    def test_estimate_a_force_noerr(self, k11):
        """Test StaticSystem force estimate a."""
        syst = from_eu(name=simple_syst, verbose=False, exact_match=False)
        a_ss = syst.estimate_semi_major_axis(force=True)

        # Assert types
        assert isinstance(a_ss, pd.Series)

        # Assert equal to base example
        pd.testing.assert_series_equal(
            a_ss,
            k11.estimated_a_forced,
            check_names=False,  # Names are different
            rtol=1e-3,
            atol=1e-4,
        )

    @pytest.mark.parametrize("err", [0, 1, 2, 3, 4])
    def test_estimate_a_noforce_err(self, err: int, k11):
        """Test StaticSystem estimate a with err."""
        syst = from_eu(name=simple_syst, verbose=False, exact_match=False)
        if err in [0, 1, 2, 3]:
            a_df = syst.estimate_semi_major_axis(force=False, err_method=err)

            # Assert types
            assert isinstance(a_df, pd.DataFrame)

            # Assert equal to base example
            pd.testing.assert_frame_equal(
                a_df,
                k11.a_with_err,
                check_names=False,  # Names are different
                rtol=1e-3,
                atol=1e-4,
            )

        else:
            with pytest.raises(
                ValueError,
                match="Invalid err_method=4.",
            ):
                syst.estimate_semi_major_axis(force=False, err_method=err)

    @pytest.mark.parametrize("err", [0, 1, 2, 3, 4])
    def test_estimate_a_force_err(self, err: int, k11):
        """Test StaticSystem estimate a with err."""
        syst = from_eu(name=simple_syst, verbose=False, exact_match=False)
        if err in [0, 1, 2, 3]:
            a_df = syst.estimate_semi_major_axis(force=True, err_method=err)

            # Assert types
            assert isinstance(a_df, pd.DataFrame)

            dicc = k11.get_estimated_a_erri_f()

            this = dicc[err]
            if err == 0:
                this.a_err_min = 0.0
                this.a_err_max = 0.0

            # Assert equal to base example
            pd.testing.assert_frame_equal(
                a_df,
                this,
                check_names=False,  # Names are different
                rtol=1e-1,
                atol=1e-2,
            )

        else:
            with pytest.raises(
                ValueError,
                match="Invalid err_method=4.",
            ):
                syst.estimate_semi_major_axis(force=False, err_method=err)


class TestCalcP:

    def test_estimate_p_naive(self, k11):
        """Test StaticSystem estimate P."""
        syst = from_eu(name=simple_syst, verbose=False, exact_match=False)
        p_ss = syst.estimate_period()

        # Assert types
        assert isinstance(p_ss, pd.Series)

        # Assert equal to base example
        pd.testing.assert_series_equal(
            p_ss,
            k11.estimated_p,
            check_names=False,  # Names are different
            rtol=1e-3,
            atol=1e-4,
        )

    def test_estimate_p_force_noerr(self, k11):
        """Test StaticSystem force estimate P."""
        syst = from_eu(name=simple_syst, verbose=False, exact_match=False)
        p_ss = syst.estimate_period(force=True)

        # Assert types
        assert isinstance(p_ss, pd.Series)

        # Assert equal to base example
        pd.testing.assert_series_equal(
            p_ss,
            k11.estimated_p_forced,
            check_names=False,  # Names are different
            rtol=1e-3,
            atol=1e-4,
        )

    @pytest.mark.parametrize("err", [0, 1, 2, 3, 4])
    def test_estimate_p_noforce_err(self, err: int, k11):
        """Test StaticSystem estimate P with err."""
        syst = from_eu(name=simple_syst, verbose=False, exact_match=False)
        if err in [0, 1, 2, 3]:
            p_df = syst.estimate_period(force=False, err_method=err)

            # Assert types
            assert isinstance(p_df, pd.DataFrame)

            # Assert equal to base example
            pd.testing.assert_frame_equal(
                p_df,
                k11.p_with_err,
                check_names=False,  # Names are different
                rtol=1e-3,
                atol=1e-4,
            )

        else:
            with pytest.raises(
                ValueError,
                match="Invalid err_method=4.",
            ):
                syst.estimate_period(force=False, err_method=err)

    @pytest.mark.parametrize("err", [0, 1, 2, 3, 4])
    def test_estimate_p_force_err(self, err: int, k11):
        """Test StaticSystem estimate P with err."""
        syst = from_eu(name=simple_syst, verbose=False, exact_match=False)
        if err in [0, 1, 2, 3]:
            p_df = syst.estimate_period(force=True, err_method=err)

            # Assert types
            assert isinstance(p_df, pd.DataFrame)

            dicc = k11.get_estimated_p_erri_f()

            this = dicc[err]
            if err == 0:
                this.p_err_min = 0.0
                this.p_err_max = 0.0

            # Assert equal to base example
            pd.testing.assert_frame_equal(
                p_df,
                this,
                check_names=False,  # Names are different
                rtol=1e-1,
                atol=1e-2,
            )

        else:
            with pytest.raises(
                ValueError,
                match="Invalid err_method=4.",
            ):
                syst.estimate_period(force=False, err_method=err)


class TestMassRadius:
    @pytest.mark.parametrize("model", ["ck17", "o20", "e23", "m24", "bad"])
    @pytest.mark.parametrize("force", [True, False])
    def test_estimate_mass(self, model: str, force: bool, k11):
        """Test StaticSystem class estimate_mass method."""
        syst = from_eu(name=simple_syst, verbose=False, exact_match=False)
        multivariate = (
            0.9999
            if model in ["o20", "m24"]
            else ((0.0001, 0.9999) if model == "ch17" else None)
        )  # Ensure that the test is always the same
        if not force:
            # Assert equal to base example
            mass_ss = syst.estimate_mass(
                model=model, force=force, multivariate=multivariate
            )
            assert mass_ss.equals(k11.data["mass"])

        elif model == "bad":
            with pytest.raises(ValueError):
                syst.estimate_mass(
                    model=model, force=force, multivariate=multivariate
                )
        else:
            if model == "o20":
                with pytest.warns(UserWarning):
                    mass_ss = syst.estimate_mass(
                        model=model, force=force, multivariate=multivariate
                    )
            else:
                mass_ss = syst.estimate_mass(
                    model=model, force=force, multivariate=multivariate
                )

            # Assert types
            assert isinstance(mass_ss, pd.Series)

            # Assert equal to base example
            pd.testing.assert_series_equal(
                mass_ss,
                k11.estimated_mass[model],
                check_names=False,  # Names are different
            )

    @pytest.mark.parametrize("model", ["ck17", "o20", "e23", "m24", "bad"])
    @pytest.mark.parametrize("force", [True, False])
    def test_estimate_radius(self, model: str, force: bool, k11):
        """Test StaticSystem class estimate_radius method."""
        syst = from_eu(name=simple_syst, verbose=False, exact_match=False)
        bivariate = 0.99 if model == "o20" else None
        if not force:
            # Assert equal to base example
            radii_ss = syst.estimate_radius(
                model=model, force=force, bivariate=bivariate
            )
            assert radii_ss.equals(k11.data["radius"])

        elif model == "bad":
            with pytest.raises(ValueError):
                syst.estimate_radius(
                    model=model, force=force, bivariate=bivariate
                )
        else:
            if model == "o20":
                with pytest.warns(UserWarning):
                    radii_ss = syst.estimate_radius(
                        model=model, force=force, bivariate=bivariate
                    )
            else:
                radii_ss = syst.estimate_radius(
                    model=model, force=force, bivariate=bivariate
                )

            # Assert types
            assert isinstance(radii_ss, pd.Series)

            # Assert equal to base example
            pd.testing.assert_series_equal(
                radii_ss,
                k11.estimated_radii[model],
                check_names=False,  # Names are different
                rtol=1e-3,
                atol=1e-4,
            )

    def test_estimate_radius_new(self, k11):
        syst = from_eu(name=simple_syst, verbose=False, exact_match=False)
        radii_ss = syst.estimate_radius(
            model="e23", force=True, new_system=True
        )

        assert isinstance(radii_ss, core.StaticSystem)


class TestPeriodRatios:
    def test_get_period_ratios(self, my_k47):
        syst = my_k47
        pr = syst.period_ratios_
        assert isinstance(pr, pd.DataFrame)
        assert pr.shape == (4, 4)
        assert pr.iloc[1, 1] == 1
        assert pr.iloc[1, 2] == 3.783778325322131

        pra = syst.period_ratio(
            "all", use_binary=False, fraction_kwargs={"max_error": 1e-1}
        )
        assert isinstance(pra, pd.DataFrame)
        assert pra.shape == (3, 3)
        assert pra.iloc[1, 1] == (1, 1)
        assert pra.iloc[1, 2] == (3, 2)

        pra = syst.period_ratio(
            "all", use_binary=True, fraction_kwargs={"max_error": 1e-1}
        )
        assert isinstance(pra, pd.DataFrame)
        assert pra.shape == (4, 4)
        assert pra.iloc[1, 1] == (1, 1)
        assert pra.iloc[2, 3] == (3, 2)

        assert syst.period_ratio([1, 2]) == 0.2642860955431012

        pre = syst.period_ratio("all", error=True)
        assert isinstance(pre, pd.DataFrame)
        assert pre.shape == (4, 4)
        assert isnan(pre.iloc[0, 0])
        assert pre.iloc[1, 1] == 0.0011424757138369715


class TestAddRemoveSwapReplace:
    def test_manipulate(self, my_k47):
        syst = my_k47
        mass = syst["mass"]
        assert isinstance(mass, pd.Series)
        assert isnan(mass.iloc[0])
        assert mass.iloc[1] == 0.05984

        assert "Kepler-47 (AB)c" in syst.planet_names_
        modif1 = syst.remove_planet(2)
        assert "Kepler-47 (AB)c" not in modif1.planet_names_

        modif2 = syst.remove_planet("Kepler-47 (AB)b")
        assert "Kepler-47 (AB)b" not in modif2.planet_names_

        # Get a planet from the original system
        planet2 = syst.planet(1)

        # Add the planet to the modified system
        with pytest.warns(UserWarning):
            modif1.add_planet(planet2)

        with pytest.warns(UserWarning):
            modif2.add_planet(planet2)

        modif3 = syst.remove_planet(1)
        modif3.add_planet(planet2)

        assert syst.suffixes_ == ["A", "B", "b", "d", "c"]
        modif4 = syst.swap_planets(1, 2)

        assert modif4.suffixes_ == ["A", "B", "b", "c", "d"]

        with pytest.warns(UserWarning):
            modif5 = syst.replace_planet(2, planet2)
        assert modif5.suffixes_ == ["A", "B", "b", "d", "d"]


class TestPlotTriplet:
    def test_plot_triplet(self, my_k47, triplet_plot_path, tmp_path):
        syst = my_k47

        matplotlib.use("svg")  # or 'svg', 'pdf', 'ps'
        plt.figure(dpi=120)
        bounds = [2.51, 4.5, 1.41, 2.1]  # [xmin, xmax, ymin, ymax]
        ax = syst.plot_triplet(
            error=True, capsize=10, label=True, draw_mmr=False
        )
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        rutils.mmr.plot_mmrs(
            bounds=bounds,
            ax=ax,
            label_mmrs=True,
            color="black",
        )
        ax.legend()
        ax.set_xlabel("Period Ratio ($P_2/P_1$)")
        ax.set_ylabel("Period Ratio ($P_3/P_2$)")

        image_file_path = tmp_path / "test_image.png"
        plt.savefig(image_file_path)

        compare_images(triplet_plot_path, image_file_path, tol=1e-6)

    def test_plot_triplet_wrong(self, my_k47, triplet_plot_path, tmp_path):
        syst = my_k47
        matplotlib.use("svg")  # or 'svg', 'pdf', 'ps'
        with pytest.raises(ValueError):
            syst.plot_triplet(
                which=123123,
                error=True,
                capsize=10,
                label=True,
                draw_mmr=False,
            )
        with pytest.raises(ValueError):
            syst.plot_triplet(
                error=True, capsize=10, label=["asd", 12], draw_mmr=False
            )

    def test_plot_triplet_specific(self, my_k47, triplet_plot_path, tmp_path):
        syst = my_k47
        matplotlib.use("svg")  # or 'svg', 'pdf', 'ps'

        with pytest.raises(ValueError):
            syst.plot_triplet(
                1, error=True, capsize=10, label=True, draw_mmr=False
            )
        with pytest.raises(ValueError):
            syst.plot_triplet(
                (0, 1, 2), error=True, capsize=10, label=True, draw_mmr=False
            )


class FakeSim:
    """Lightweight fake REBOUND Simulation."""

    def __init__(self):
        self.added = []
        self.removed = []
        self.particles = {}

    def add(self, **kwargs):
        self.added.append(kwargs)
        if "hash" in kwargs:
            self.particles[kwargs["hash"]] = core.rng

    def remove(self, hashh):
        self.removed.append(hashh)
        self.particles.pop(hashh, None)


class TestToRebound:
    rebound = pytest.importorskip("rebound")

    def test_to_rebound_single(self, simple_system, capsys):
        sys = simple_system
        sim = sys.to_rebound(sim=None, fillna=True, units=True, verbose=True)
        assert (
            isinstance(sim, core.Simulation)
            or isinstance(sim, FakeSim)
            or hasattr(sim, "add")
        )
        cap = capsys.readouterr()
        assert "Star" in cap.out or "planets" in cap.out

    def test_to_rebound_binary_non_circumbinary(self, simple_system):
        sys = simple_system
        fake_sim = FakeSim()
        result = sys.to_rebound(
            sim=fake_sim, fillna=True, units=False, verbose=False
        )
        assert isinstance(result, FakeSim)
        # Should have two stars and one planet
        hashes = [a["hash"] for a in fake_sim.added]
        assert "st1" in hashes and "pl1" in hashes and "pl2" in hashes

    def test_to_rebound_fillna_false(self, simple_system):
        sys = simple_system
        fake_sim = FakeSim()
        # Set planet attributes to trigger zero branches
        pl1 = sys.planet(1)
        pl1 = pl1.set_attr("mass", 0.1)
        pl1 = pl1.set_attr("radius", 0.4)
        pl1 = pl1.set_attr("a", 7)
        pl1 = pl1.set_attr("name", "nuevo")
        new = sys.remove_planet(1)
        new = new.add_planet(pl1)
        res = new.to_rebound(
            sim=fake_sim, fillna=False, units=True, verbose=False
        )
        assert isinstance(res, FakeSim)
        hashes = [a["hash"] for a in fake_sim.added]
        assert "nuevo" in hashes
