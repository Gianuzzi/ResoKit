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

import pandas as pd

import pytest

import resokit.utils as rutils
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
