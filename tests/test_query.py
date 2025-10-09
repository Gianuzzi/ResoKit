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

import resokit.query.query as qry

# ============================================================================
# TESTS
# ============================================================================


class TestQuery:
    # ---------------------------------------------------------------------
    # build_query
    # ---------------------------------------------------------------------

    def test_build_query_basic_and_alias(self):
        q = qry.build_query(
            "eu", select="name", alias="t", conditions="a=1", order_by="x"
        )
        assert "ORDER BY x" in q

    def test_build_query_invalid_select_alias_order(self):
        with pytest.raises(ValueError):
            qry.build_query("nasa", select=123)
        with pytest.raises(ValueError):
            qry.build_query("nasa", alias=123)
        with pytest.raises(ValueError):
            qry.build_query("nasa", order_by=123)

    def test_build_query_multiple_conditions(self):
        q = qry.build_query("eu", conditions=["a>1", "b<2"])
        assert "AND" in q

    def test_execute_query_invalid_source(self):
        with pytest.raises(ValueError):
            qry.execute_query("bad", "SELECT * FROM x WHERE y=1")

    def test_execute_query_missing_where_and_soft_false(self):
        with pytest.raises(ValueError):
            qry.execute_query("nasa", "SELECT * FROM ps", soft=False)

    def test_execute_query_soft_true(self, mock_requests):
        # mock_requests provides a fake CSV response
        out = qry.execute_query(
            "nasa", "SELECT * FROM ps", soft=True, verbose=True
        )
        assert isinstance(out, pd.DataFrame)
        assert not out.empty

    def test_execute_query_cache(self, monkeypatch):
        df = pd.DataFrame({"x": [1, 2]})
        monkeypatch.setattr(
            qry,
            "_session_queries",
            {
                "https://exoplanetarchive.ipac.caltech.edu/"
                + "TAP/sync?"
                + "query=SELECT+%2A+FROM+ps+WHERE+1%3D1&format=csv": df
            },
        )

        out = qry.execute_query("nasa", "SELECT * FROM ps WHERE 1=1")
        assert isinstance(out, pd.DataFrame)

        outb = qry.execute_query(
            "nasa", "SELECT * FROM ps WHERE 1=1", to_bytes=True
        )
        assert isinstance(outb, (bytes, bytearray))

    def test_execute_query_cache_corrupted(self, monkeypatch):
        monkeypatch.setattr(
            qry,
            "_session_queries",
            {
                "https://exoplanetarchive.ipac.caltech.edu/"
                + "TAP/sync?"
                + "query=SELECT+%2A+FROM+ps+WHERE+1%3D1&format=csv": 32
            },
        )

        with pytest.raises(ValueError):
            qry.execute_query("nasa", "SELECT * FROM ps WHERE 1=1")

    def test_execute_query_eu_branch(self, mock_requests, monkeypatch):
        # astropy.Table.read mocked to a simple dataframe
        class DummyTable:
            @staticmethod
            def read(data):
                return type(
                    "Obj",
                    (),
                    {"to_pandas": lambda self=None: pd.DataFrame({"y": [3]})},
                )()

        monkeypatch.setattr(qry, "Table", DummyTable, raising=False)
        monkeypatch.setattr(qry, "astropy_imported", True)
        out = qry.execute_query(
            "eu", "SELECT * FROM exoplanet.epn_core WHERE a=1"
        )
        monkeypatch.undo()
        assert isinstance(out, pd.DataFrame)

    def test_execute_query_request_exception(self, fake_requests_failure):
        # fake_requests_failure simulates RequestException
        qry.execute_query("nasa", "SELECT * FROM ps WHERE x=1", soft=True)

    # ---------------------------------------------------------------------
    # query_system coverage
    # ---------------------------------------------------------------------

    def make_df(self):
        return pd.DataFrame({"col": [1]})

    def test_query_system_invalid_name_args(self):
        with pytest.raises(ValueError):
            qry.query_system("eu")  # neither star nor planet
        with pytest.raises(ValueError):
            qry.query_system("eu", star_name="s", planet_name="p")  # both

    def test_query_system_invalid_source(self):
        with pytest.raises(ValueError):
            qry.query_system("bad", star_name="a")

    def test_query_system_eu_and_nasa(
        self, mock_requests, mock_csv_data, monkeypatch
    ):
        monkeypatch.setattr(qry, "execute_query", lambda **k: mock_csv_data)
        monkeypatch.setattr(qry, "df_to_resokit", lambda **k: {"dummy": True})
        monkeypatch.setattr(
            qry, "resokit_to_system", lambda x: {"system": True}
        )

        monkeypatch.setattr(qry, "astropy_imported", True)
        r1 = qry.query_system("eu", star_name="Alpha")
        assert isinstance(r1, dict)

        r2 = qry.query_system("nasa", planet_name="Beta")
        assert isinstance(r2, dict)

        r3 = qry.query_system(
            "nasa", star_name="Ceti", default_flag=1, controversial_flag=1
        )
        assert isinstance(r3, dict)

    def test_query_system_raw_and_as_frame(self, mock_csv_data, monkeypatch):
        monkeypatch.setattr(qry, "execute_query", lambda **k: mock_csv_data)
        monkeypatch.setattr(
            qry, "df_to_resokit", lambda **k: pd.DataFrame({"a": [1]})
        )

        out = qry.query_system("nasa", planet_name="Beta", raw=True)
        assert not isinstance(out, pd.DataFrame)

        out2 = qry.query_system("nasa", planet_name="Gamma", as_frame=True)
        assert isinstance(out2, pd.DataFrame)
