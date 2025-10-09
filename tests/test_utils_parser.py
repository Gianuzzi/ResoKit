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

import types

import pandas as pd

import pytest

import resokit.utils.parser as parser

# ============================================================================
# TESTS
# ============================================================================


class TestParser:

    # ======================================================
    # assert_module_imported
    # ======================================================

    def test_assert_module_imported_success(self):
        # Case: already imported
        assert parser.assert_module_imported(True, "math")

    def test_assert_module_imported_retry_success(self, monkeypatch):
        called = {}

        def fake_import(name, package=None):
            called["ok"] = True
            return types.SimpleNamespace()

        monkeypatch.setattr(parser.importlib, "import_module", fake_import)
        assert parser.assert_module_imported(False, "math")

    def test_assert_module_imported_retry_fail(self, monkeypatch):
        def fake_import(*a, **k):
            raise ImportError

        monkeypatch.setattr(parser.importlib, "import_module", fake_import)
        with pytest.raises(ImportError):
            parser.assert_module_imported(False, "nope")

    def test_assert_module_imported_no_retry(self):
        with pytest.raises(ImportError):
            parser.assert_module_imported(
                False, "nope", retry=False, alias="alias"
            )

    # ======================================================
    # parse_to_iter
    # ======================================================

    def test_parse_to_iter_string_and_iterable(self):
        assert parser.parse_to_iter("abc") == ["abc"]
        assert parser.parse_to_iter(5) == [5]
        assert tuple(parser.parse_to_iter([1, 2, 3], to=tuple)) == (1, 2, 3)
        assert parser.parse_to_iter([1, 2], to=None) == [1, 2]

    # ======================================================
    # parse_name
    # ======================================================

    @pytest.mark.parametrize(
        "name,force,expected",
        [
            ("Alpha A", False, "alpha"),
            ("Beta B", False, "beta"),
            ("Gamma AB", False, "gamma"),
            ("Delta (AB)", False, "delta"),
            ("Epsilon(AB)", True, "epsilon"),
        ],
    )
    def test_parse_name_variants(self, name, force, expected):
        out = parser.parse_name(name, force)
        assert out == expected

    # ======================================================
    # _similar and _n_close
    # ======================================================

    def test_similar_and_n_close(self):
        assert 0 <= parser._similar("abc", "abd") <= 1
        assert parser._n_close("abc", "ab", 2, 1)
        assert not parser._n_close("abc", "xx", 2, 1)

    # ======================================================
    # find_best_match
    # ======================================================

    def make_series(self):
        return pd.Series(
            ["Alpha", "Alpha A", "Beta", "Gamma(AB)", "Delta  B", "Zeta"]
        )

    def test_find_best_match_exact_1_0(self):
        s = pd.Series(["Alpha"])
        idx, vals, ratio = parser.find_best_match(s, "Alpha")
        assert ratio == 1.0

    def test_find_best_match_exact_almost(self):
        s = pd.Series(["ALPHA"])
        idx, vals, ratio = parser.find_best_match(s, "Alpha")
        assert ratio == pytest.approx(0.99999)

    def test_find_best_match_space_close_1(self):
        s = pd.Series(["Alpha A", "Beta"])
        idx, vals, ratio = parser.find_best_match(s, "Alpha")
        assert ratio > 0.8
        assert idx == 0

    def test_find_best_match_space_close_2(self):
        s = pd.Series(["Beta", "Alpha  A"])
        idx, vals, ratio = parser.find_best_match(s, "Alpha")
        assert ratio > 0.8
        assert idx == 1

    def test_find_best_match_similarity(self, monkeypatch):
        s = pd.Series(["ZZZ", "YYY", "XXX"])
        # Force threshold high so fallback triggers
        monkeypatch.setattr(parser, "RATIOS_THRESHOLD", 1.5)
        idx, vals, ratio = parser.find_best_match(s, "ZZY")
        assert len(idx) <= 3
        assert isinstance(ratio, float)

    def test_find_best_match_parse_none(self):
        s = pd.Series(["Alpha", "Beta"])
        idx, vals, ratio = parser.find_best_match(s, "Alpha", parse=None)
        assert isinstance(idx, pd.Index)

    def test_find_best_match_force_true(self):
        s = pd.Series(["Gamma(AB)"])
        idx, vals, ratio = parser.find_best_match(
            s, "Gamma(AB)", parse=True, force=True
        )
        assert isinstance(idx, pd.Index)

    # ======================================================
    # MAPPINGS and constants sanity
    # ======================================================

    def test_mapping_integrity(self):
        assert "eu" in parser.MAPPINGS
        assert "nasa" in parser.MAPPINGS
        assert isinstance(parser.DEFAULT_METADATA, types.MappingProxyType)
        assert isinstance(parser.QUERY_MAPPINGS, types.MappingProxyType)
        assert isinstance(parser.RESO_DTYPES, types.MappingProxyType)
        assert isinstance(parser.QUERY_MISSING, types.MappingProxyType)
        assert parser.RATIOS_THRESHOLD > 0
