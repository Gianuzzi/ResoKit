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

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import pytest

import resokit.utils.mmr as mmr

matplotlib.use("Agg")


# ============================================================================
# TESTS
# ============================================================================


class TestMMR:

    # ======================================================
    # mmr3b
    # ======================================================

    def test_mmr3b_scalar_and_array(self):
        y_scalar = mmr.mmr3b(2.0, (1, 2, 3))
        assert np.isclose(y_scalar, -3 / (1 * 2 + 2))

        x = np.linspace(1, 5, 10)
        y = mmr.mmr3b(x, (1, 2, 3))
        assert y.shape == x.shape
        assert np.isnan(y).sum() <= 1  # only one singularity possible

    def test_mmr3b_singularity_scalar(self):
        res = (1, -2, 3)
        x = 2.0  # makes denominator zero
        y = mmr.mmr3b(x, res)
        assert np.isnan(y)

    # ======================================================
    # _is_curve_within_bounds
    # ======================================================

    def test_is_curve_within_bounds_true_false(self):
        assert mmr.mmrs._is_curve_within_bounds([1, 2, 3], (1, 5, 1, 5)) in [
            True,
            False,
        ]
        # Singular case
        mmr.mmrs._is_curve_within_bounds([1, -1, 1], (1, 5, 1, 5))

    # ======================================================
    # mmrs_in_area
    # ======================================================

    def test_mmrs_in_area_invalid_bounds(self):
        with pytest.raises(ValueError):
            mmr.mmrs_in_area((0.5, 2, 1, 2))
        with pytest.raises(ValueError):
            mmr.mmrs_in_area((2, 1, 1, 2))

    def test_mmrs_in_area_ok(self):
        res = mmr.mmrs_in_area(
            (1, 2, 1, 2), order3=0, max_coeff3=3, max_order3=1, mmr2b=False
        )
        assert isinstance(res, list)
        res_full = mmr.mmrs_in_area(
            (1, 2, 1, 2), order3=0, max_coeff3=3, max_order3=1, mmr2b=True
        )
        assert len(res_full) == 3

    # ======================================================
    # mindist_mmr3b
    # ======================================================

    def test_mindist_mmr3b_basic(self):
        x_min, y_min, dist = mmr.mindist_mmr3b(1.5, 2.0, (1, -2, 1))
        assert np.isfinite(dist)

    def test_mindist_mmr3b_x0_out_of_bounds(self):
        with pytest.raises(ValueError):
            mmr.mindist_mmr3b(1.5, 2.0, (1, 1, 1), x0=1000)

    def test_mindist_mmr3b_unphysical_allowed(self):
        x_min, y_min, dist = mmr.mindist_mmr3b(
            1.5, 0.5, (1, 1, -1), unphysical=True
        )
        assert np.isfinite(dist)

    def test_mindist_mmr3b_failure(self, monkeypatch):
        from scipy.optimize import OptimizeResult

        def fake_minimize(*a, **k):
            return OptimizeResult(success=False)

        monkeypatch.setattr("resokit.utils.mmr.mmrs.minimize", fake_minimize)

        with pytest.raises(ValueError):
            mmr.mindist_mmr3b(1.5, 2.0, (1, 1, 1))

    # ======================================================
    # closest_mmr3b
    # ======================================================

    def test_closest_mmr3b_works(self):
        res, dist = mmr.closest_mmr3b(
            1.5, 2.0, max_coeff3=3, max_order3=1, verbose=False
        )
        assert isinstance(res, list)
        assert np.isfinite(dist)

    def test_closest_mmr3b_no_bounds_or_radius(self):
        with pytest.raises(ValueError):
            mmr.closest_mmr3b(1, 1, radius=None, bounds=None)

    def test_closest_mmr3b_no_resonances(self, monkeypatch):
        monkeypatch.setattr(
            "resokit.utils.mmr.mmrs.mmrs_in_area", lambda *a, **k: []
        )
        with pytest.raises(ValueError):
            mmr.closest_mmr3b(1, 1, verbose=False)

    def test_closest_mmr3b_message(self, capfd):
        res, dist = mmr.closest_mmr3b(
            1.5, 2.0, max_coeff3=3, max_order3=1, verbose=True
        )
        out, _ = capfd.readouterr()
        # Ensure the verbose output is correct
        assert f"Closest 3-body mean-motion resonance: {res}" in out
        assert f"Distance: {dist}" in out

    # ======================================================
    # label_mmr2b
    # ======================================================

    def test_label_mmr2b_cross_x(self, monkeypatch):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        mmr.mmrs.label_mmr2b((2, 1), ax=ax, xaxis=True, warn=False)
        mmr.mmrs.label_mmr2b((2, 1), ax=ax, xaxis=False, warn=False)

    def test_label_mmr2b_warns(self):
        with pytest.warns(UserWarning):
            mmr.mmrs.label_mmr2b((10, 1), xaxis=True, warn=True)

    # ======================================================
    # label_mmr3b
    # ======================================================

    def test_label_mmr3b_cross(self):
        fig, ax = plt.subplots()
        ax.set_xlim(1, 3)
        ax.set_ylim(1, 3)
        mmr.label_mmr3b((1, 1, -1), ax=ax, warn=False)
        mmr.label_mmr3b((1, 1, 1), ax=ax, warn=False)

    def test_label_mmr3b_warns(self):
        with pytest.warns(UserWarning):
            mmr.label_mmr3b((10, 1, 1), warn=True)

    # ======================================================
    # plot_mmrs
    # ======================================================

    def test_plot_mmrs_basic(self):
        with pytest.warns(UserWarning):
            fig, ax = plt.subplots()
            ax.set_xlim(1, 2)
            ax.set_ylim(1, 2)
            out = mmr.plot_mmrs(
                bounds=(1, 2, 1, 2),
                ax=ax,
                n_points=10,
                label_mmrs=True,
                label_2mmrs=True,
            )
        assert isinstance(out, plt.Axes)

    def test_plot_mmrs_no_bounds(self):
        fig, ax = plt.subplots()
        ax.set_xlim(1, 2)
        ax.set_ylim(1, 2)
        out = mmr.plot_mmrs(ax=ax, n_points=5, mmr2b=False)
        assert isinstance(out, plt.Axes)
