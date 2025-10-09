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

import pytest

import resokit.units.units as units
import resokit.utils as utils


# ============================================================================
# TESTS
# ============================================================================


class TestUtils:

    # ======================================================
    # float_to_fraction
    # ======================================================

    def test_float_to_fraction_basic(self):
        num, den = utils.float_to_fraction(3.14159, max_iter=5, verbose=False)
        assert isinstance(num, int)
        assert isinstance(den, int)

        f = utils.float_to_fraction(
            1.3333, max_error=1e-5, as_fraction=True, verbose=False
        )
        assert isinstance(f, Fraction)

    def test_float_to_fraction_invalid_args(self):
        with pytest.raises(ValueError):
            utils.float_to_fraction(1.5, verbose=False)
        with pytest.raises(TypeError):
            utils.float_to_fraction("a", max_iter=5, verbose=False)
        with pytest.raises(TypeError):
            utils.float_to_fraction(1.5, max_iter="a", verbose=False)
        with pytest.raises(TypeError):
            utils.float_to_fraction(1.5, max_error="a", verbose=False)

    def test_float_to_fraction_stop_func(self):
        def stopf(n, d):
            return n > 2

        val = utils.float_to_fraction(1.414, stop_func=stopf, verbose=False)
        assert isinstance(val, tuple)

    def test_float_to_fraction_stop_func_invalid(self):
        with pytest.raises(TypeError):
            utils.float_to_fraction(
                1.414, stop_func="not_callable", max_iter=5, verbose=False
            )

        def badfunc(n, d):
            return "nope"

        with pytest.raises(TypeError):
            utils.float_to_fraction(
                1.414, stop_func=badfunc, max_iter=5, verbose=False
            )

    def test_float_to_fraction_min_max_iter(self, capfd):
        utils.float_to_fraction(1.414, max_iter=1, verbose=True)
        out = capfd.readouterr().out
        assert "Approximating float" in out

    # ======================================================
    # calc_period and calc_period_with_errors
    # ======================================================

    def test_calc_period(self, monkeypatch):
        # Patch MKS constants
        monkeypatch.setitem(units._mks, "G", 6.67e-11)
        monkeypatch.setitem(units._mks, "ms", 1.99e30)
        monkeypatch.setitem(units._mks, "mj", 1.9e27)
        monkeypatch.setitem(units._mks, "au", 1.496e11)
        monkeypatch.setitem(units._mks, "day", 86400)
        p = utils.calc_period(1, 1, 0.001)
        assert p > 0

    def test_calc_period_with_errors_methods(self):
        a, amin, amax = utils.utils.calc_period_with_errors(
            1, 0.01, 0.02, 1, 0.01, 0.02, 0.001, 0.001, 0.002, err_method=-1
        )
        assert amin == 0 and amax == 0

        utils.utils.calc_period_with_errors(
            1, 0.01, 0.02, 1, 0.01, 0.02, 0.001, 0.001, 0.002, err_method=1
        )
        utils.utils.calc_period_with_errors(
            1, 0.01, 0.02, 1, 0.01, 0.02, 0.001, 0.001, 0.002, err_method=2
        )
        utils.utils.calc_period_with_errors(
            1, 0.01, 0.02, 1, 0.01, 0.02, 0.001, 0.001, 0.002, err_method=3
        )

    def test_calc_period_with_errors_invalid(self):
        with pytest.raises(ValueError):
            utils.utils.calc_period_with_errors(
                1, 0, 0, 1, 0, 0, 1, 0, 0, err_method=99
            )

    # ======================================================
    # calc_a and calc_a_with_errors
    # ======================================================

    def test_calc_a(self, monkeypatch):
        monkeypatch.setitem(units._mks, "G", 6.67e-11)
        monkeypatch.setitem(units._mks, "ms", 1.99e30)
        monkeypatch.setitem(units._mks, "mj", 1.9e27)
        monkeypatch.setitem(units._mks, "au", 1.496e11)
        monkeypatch.setitem(units._mks, "day", 86400)
        a = utils.calc_a(365, 1, 0)
        assert a > 0

    def test_calc_a_with_errors_methods(self):
        args = (365, 1, 2, 1, 0.1, 0.1, 0.001, 0.0001, 0.0001)
        utils.utils.calc_a_with_errors(*args, err_method=-1)
        utils.utils.calc_a_with_errors(*args, err_method=1)
        utils.utils.calc_a_with_errors(*args, err_method=2)
        utils.utils.calc_a_with_errors(*args, err_method=3)
        with pytest.raises(ValueError):
            utils.utils.calc_a_with_errors(*args, err_method=9)

    # ======================================================
    # calc_hill_radius and calc_hill_radius_with_errors
    # ======================================================

    def test_calc_hill_radius(self, monkeypatch):
        monkeypatch.setitem(units._mks, "ms", 1.99e30)
        monkeypatch.setitem(units._mks, "mj", 1.9e27)
        r = utils.calc_hill_radius(1, 0.1, 1, 0.001)
        assert r > 0

    def test_calc_hill_radius_with_errors_methods(self):
        args = (
            1,
            0.01,
            0.02,
            0.1,
            0.01,
            0.02,
            1,
            0.01,
            0.02,
            0.001,
            0.001,
            0.002,
        )
        utils.utils.calc_hill_radius_with_errors(*args, err_method=-1)
        utils.utils.calc_hill_radius_with_errors(*args, err_method=1)
        utils.utils.calc_hill_radius_with_errors(*args, err_method=2)
        utils.utils.calc_hill_radius_with_errors(*args, err_method=3)
        with pytest.raises(ValueError):
            utils.utils.calc_hill_radius_with_errors(*args, err_method=5)

    # ======================================================
    # calc_sum_with_errors
    # ======================================================

    def test_calc_sum_with_errors_methods(self):
        vals = [(1, 0.1, 0.2), (2, 0.05, 0.05)]
        utils.utils.calc_sum_with_errors(*vals, err_method=-1)
        utils.utils.calc_sum_with_errors(*vals, err_method=1)
        utils.utils.calc_sum_with_errors(*vals, err_method=2)
        utils.utils.calc_sum_with_errors(*vals, err_method=3)
        with pytest.raises(ValueError):
            utils.utils.calc_sum_with_errors(*vals, err_method=42)
