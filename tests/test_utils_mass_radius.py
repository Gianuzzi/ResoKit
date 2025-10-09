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

import numpy as np

import pytest

import resokit.utils.mass_radius.models as models

# ============================================================================
# TESTS
# ============================================================================


class TestMassradius:

    # ============================================================================
    # Basic power-law utilities
    # ============================================================================

    def test_power_law_and_error_basic(self):
        y = models.power_law(2.0, 1.5, 0.5)
        assert np.isclose(y, 1.5 * np.sqrt(2))

        err = models.power_law_error(2.0, 0.1, 1.5, 0.1, 0.5, 0.05, y=y)
        assert err > 0

    def test_power_law_error_auto_y(self):
        err = models.power_law_error(1.0, 0.1, 2.0, 0.1, 1.0, 0.1)
        assert np.isfinite(err)

    def test_power_law_error_recompute_y(self, monkeypatch):
        monkeypatch.setattr(models, "power_law", lambda *a, **k: 42.0)
        err = models.power_law_error(1, 0.1, 1, 0.1, 1, 0.1, y=0)
        assert np.isfinite(err)

    # ============================================================================
    # Chen & Kipping 2017
    # ============================================================================

    def test_chen_kipp_radius(self, monkeypatch):
        # Mock convert to identity scaling
        monkeypatch.setattr(models, "convert", lambda *a, **k: 100.0)
        assert models.chen_kipp_2017_radius(0.5)[1][0] > 0
        assert models.chen_kipp_2017_radius(3.0)[1][0] > 0
        assert models.chen_kipp_2017_radius(200)[1][0] > 0
        assert models.chen_kipp_2017_radius(1e6)[1][0] > 0

    def test_chen_kipp_mass(self, monkeypatch):
        monkeypatch.setattr(np.random, "rand", lambda: 0.1)
        m1 = models.chen_kipp_2017_mass(1.0)
        assert m1[0] > 0

        # In trivariate region; trigger warning + third branch
        monkeypatch.setattr(np.random, "rand", lambda: 0.9)
        with warnings.catch_warnings(record=True) as w:
            m2 = models.chen_kipp_2017_mass(13)
        assert any("trivariate" in str(wi.message).lower() for wi in w)
        assert m2[0] > 0

        # Bad trivariate args
        with pytest.raises(ValueError):
            with pytest.warns(UserWarning):
                models.chen_kipp_2017_mass(13, trivariate=(1.0, 1.0))
        with pytest.raises(ValueError):
            with pytest.warns(UserWarning):
                models.chen_kipp_2017_mass(13, trivariate=(0.6, 0.6))
        with pytest.raises(ValueError):
            with pytest.warns(UserWarning):
                models.chen_kipp_2017_mass(13, trivariate="nope")

    def test_chen_kipp_2017_mass_nan_inputs(self):
        with pytest.warns(UserWarning):
            val = models.chen_kipp_2017_mass(np.nan)
        assert np.isnan(val[0])

    # ============================================================================
    # Otegi 2020
    # ============================================================================

    def test_otegi_radius_density_paths(self, monkeypatch):
        monkeypatch.setattr(models, "convert", lambda *a, **k: 100.0)
        assert models.otegi_2020_radius(1, density=4000)[0] > 0
        assert models.otegi_2020_radius(1, density=1000)[0] > 0

    def test_otegi_radius_mass_paths(self, monkeypatch):
        with pytest.raises(ValueError):
            with pytest.warns(UserWarning):
                models.otegi_2020_radius(10, bivariate=2.0)
        monkeypatch.setattr(models, "convert", lambda *a, **k: 100.0)
        assert models.otegi_2020_radius(1)[0] > 0
        assert models.otegi_2020_radius(50)[0] > 0
        # multivariate region, random control
        monkeypatch.setattr(np.random, "rand", lambda: 0.0)
        models.otegi_2020_radius(10)
        monkeypatch.setattr(np.random, "rand", lambda: 1.0)
        models.otegi_2020_radius(10, bivariate=0.5, silent=True)

    def test_otegi_mass_density_and_mass_paths(self, monkeypatch):
        with pytest.raises(ValueError):
            with pytest.warns(UserWarning):
                models.otegi_2020_mass(2.7, bivariate="as")
        monkeypatch.setattr(models, "convert", lambda *a, **k: 100.0)
        assert models.otegi_2020_mass(1, density=4000)[0] > 0
        assert models.otegi_2020_mass(1, density=1000)[0] > 0
        assert models.otegi_2020_mass(1)[0] > 0
        assert models.otegi_2020_mass(4)[0] > 0
        monkeypatch.setattr(np.random, "rand", lambda: 0.0)
        with warnings.catch_warnings(record=True):
            models.otegi_2020_mass(2.7)
        monkeypatch.setattr(np.random, "rand", lambda: 1.0)
        models.otegi_2020_mass(2.7, bivariate=0.5, silent=True)

    def test_otegi_2020_radius_zero(self, monkeypatch):
        monkeypatch.setattr(models, "convert", lambda *a, **k: 100.0)
        radius = models.otegi_2020_radius(0.0)
        assert radius[0] == 0.0

    def test_otegi_2020_mass_zero(self, monkeypatch):
        monkeypatch.setattr(models, "convert", lambda *a, **k: 100.0)
        mass = models.otegi_2020_mass(0.0)[0]
        assert mass == 0.0

    def test_otegi_2020_mass_nan(self, monkeypatch):
        monkeypatch.setattr(models, "convert", lambda *a, **k: 100.0)
        val = models.otegi_2020_mass(np.nan)
        assert np.isnan(val[0])

    # ============================================================================
    # Edmondson 2023
    # ============================================================================

    def test_edmonson_radius_and_mass(self, monkeypatch):
        monkeypatch.setattr(models, "convert", lambda *a, **k: 100.0)
        assert models.edmonson_2023_radius(1)[0] > 0
        assert models.edmonson_2023_radius(10)[0] > 0
        assert models.edmonson_2023_radius(1000)[0] > 0
        m = models.edmonson_2023_mass(1)
        assert m[0] > 0
        m2 = models.edmonson_2023_mass(2)
        assert m2[0] > 0
        # Above max radius triggers ValueError
        with pytest.raises(ValueError):
            models.edmonson_2023_mass(1e6)

    def test_edmonson_2023_mass_invalid_inputs(self):
        mass = models.edmonson_2023_mass(0)
        assert mass[0] == 0.0
        val = models.edmonson_2023_mass(np.nan)
        assert np.isnan(val[0])

    # ============================================================================
    # Müller 2024
    # ============================================================================

    def test_muller_radius_and_mass(self, monkeypatch):
        monkeypatch.setattr(models, "convert", lambda *a, **k: 100.0)
        assert models.muller_2024_radius(1)[0] > 0
        assert models.muller_2024_radius(10)[0] > 0
        assert models.muller_2024_radius(1000)[0] > 0

        with warnings.catch_warnings(record=True):
            models.muller_2024_mass(1.0)

        with pytest.raises(ValueError):
            models.muller_2024_mass(9999)  # above max
        monkeypatch.setattr(np.random, "rand", lambda: 0.0)
        models.muller_2024_mass(13, bivariate=0.5, silent=True)
        with pytest.raises(ValueError):
            with pytest.warns(UserWarning):
                models.muller_2024_mass(13, bivariate=-1)
        with pytest.raises(ValueError):
            with pytest.warns(UserWarning):
                models.muller_2024_mass(13, bivariate="bad")

    # ============================================================================
    # Error estimator
    # ============================================================================

    def test_aux_error_estimator_methods(self):
        c = (1.0, 0.1)
        s = (1.0, 0.1)
        x0 = (1.0, 0.0)
        out = models._aux_error_estimator(
            1, 0.1, 0.1, 1.0, c, s, x0, 1, True, 0
        )
        assert isinstance(out, tuple)
        out = models._aux_error_estimator(
            1, 0.1, 0.1, 1.0, c, s, x0, 2, True, 0
        )
        out = models._aux_error_estimator(
            1, 0.1, 0.1, 1.0, c, s, x0, 3, True, 0
        )
        with pytest.raises(ValueError):
            models._aux_error_estimator(1, 0.1, 0.1, 1.0, c, s, x0, 9, True, 0)

    def test_aux_error_estimator_invalid_flag(self):
        c = (1.0, 0.1)
        s = (1.0, 0.1)
        x0 = (1.0, 0.0)
        with pytest.raises(ValueError):
            models._aux_error_estimator(
                1, 0.1, 0.1, 1.0, c, s, x0, 99, False, 0
            )

    # ============================================================================
    # estimate_mass_single / estimate_radius_single
    # ============================================================================

    def test_estimate_mass_single_and_radius_single(self, monkeypatch):
        monkeypatch.setattr(np.random, "rand", lambda: 0.5)
        m = models.estimate_mass_single(1.0, model="ck17")
        assert m[0] > 0
        m = models.estimate_mass_single(1.0, model="o20")
        models.estimate_radius_single(1.0, model="ck17")
        models.estimate_radius_single(1.0, model="o20")
        with pytest.raises(ValueError):
            models.estimate_mass_single(1.0, model="xx")
        with pytest.raises(ValueError):
            models.estimate_radius_single(1.0, model="xx")
        # NaN inputs
        assert all(np.isnan(models.estimate_mass_single(np.nan)))
        assert all(np.isnan(models.estimate_radius_single(np.nan)))

    def test_estimate_mass_single_invalid_model(self):
        with pytest.raises(ValueError):
            models.estimate_mass_single(1.0, model="unknown", err_method=1)

    def test_estimate_mass_single_nan(self):
        out = models.estimate_mass_single(np.nan)
        assert np.all(np.isnan(out))

    def test_estimate_radius_single_array(self):
        arr = np.array([1.0, 2])
        with pytest.raises(ValueError):
            models.estimate_radius_single(arr)

    # ============================================================================
    # Vectorized and wrapper functions
    # ============================================================================

    def test_estimate_radius_and_mass_wrappers(self, monkeypatch):
        monkeypatch.setattr(np.random, "rand", lambda: 0.5)
        # Scalar inputs
        res = models.estimate_radius(1.0)
        assert np.isfinite(res)
        arr = models.estimate_radius(1.0, err_method=0)
        assert arr.shape == (3,)
        # Array inputs
        vals = np.array([1.0, 2.0])
        res2 = models.estimate_radius(vals)
        assert isinstance(res2, np.ndarray)
        res3 = models.estimate_mass(vals)
        assert isinstance(res3, np.ndarray)
        # Scalar again, mass
        m = models.estimate_mass(1.0)
        assert np.isfinite(m)
        with pytest.raises(ValueError):
            models.estimate_mass(1.0, model="bad")

    def test_estimate_radius_invalid_model(self):
        with pytest.raises(ValueError):
            models.estimate_radius(1.0, model="bad")

    def test_estimate_radius_invalid_array_dtype(self):
        arr = np.array(["bad"])
        with pytest.raises(TypeError):
            models.estimate_radius(arr)

    def test_estimate_mass_invalid_array_dtype(self):
        arr = np.array(["bad"])
        with pytest.raises(TypeError):
            models.estimate_mass(arr)

    def test_estimate_radius_invalid_err_method(self):
        with pytest.raises(ValueError):
            models.estimate_radius(1.0, err_method=99)
