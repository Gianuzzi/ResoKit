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


# import pathlib

# import numpy as np

from pathlib import Path

from pandas import DataFrame

import pytest

import requests

import resokit as rk


@pytest.fixture(scope="session")
def fake_df():
    return DataFrame(
        {
            "name": ["109 Psc b", "112 Psc b"],
            "star_name": ["109 Psc", "112 Psc"],
        }
    )


@pytest.fixture(scope="session")
def fake_header():
    return "This is a fake header."


@pytest.fixture(scope="session")
def fake_csv_bytes():
    return b"pl_name,hostname\nPlanetX,StarX\nPlanetY,StarY\n"


@pytest.fixture(scope="session")
def fake_bina_bytes():
    return (
        b"===\nByte-b\n-------\nBytes Form\n-------------\n1- 15 A15  "
        + b"\n-----------------------------\n"
        + b"DPLeo         X 0.690 0.090  305.00 3 0.0027  0.001 1"
        + b"    8.1900 0.390  6.05000 0.0054 999.000\n"
        + b"NNSer         X 0.535 0.111  521.00 3 0.0039  0.001 3"
        + b"    3.3900 0.200  2.28000 0.0084 999.000"
    )


@pytest.fixture
def fake_requests_success(monkeypatch):
    """Mocks requests.get for a successful binary line count."""

    class MockResponse:

        def iter_lines(self):
            return [b"line1", b"line2", b"line3"]

        def raise_for_status(self):
            pass

    def mock_get(url, stream=True):
        return MockResponse()

    monkeypatch.setattr("resokit.datasets.utils.requests.get", mock_get)


@pytest.fixture
def fake_requests_failure(monkeypatch):
    """Mocks requests.get to raise a simulated network error."""

    def mock_get(url, stream=True):
        raise requests.RequestException("Simulated network failure")

    monkeypatch.setattr("resokit.datasets.utils.requests.get", mock_get)


@pytest.fixture
def mock_requests(monkeypatch, fake_csv_bytes):
    class FakeResponse:
        def __init__(self, content):
            self.content = content
            self.status_code = 200
            self.ok = True
            self.text = content.decode("utf-8")

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=1024):
            for i in range(0, len(self.content), chunk_size):
                yield self.content[i : i + chunk_size]  # noqa

    def fake_get(url, stream=True, timeout=30):
        return FakeResponse(fake_csv_bytes)

    monkeypatch.setattr("requests.get", fake_get)


@pytest.fixture
def mock_requests_bina(monkeypatch, fake_bina_bytes):
    class FakeResponse:
        def __init__(self, content):
            self.content = content
            self.status_code = 200
            self.ok = True
            self.text = content.decode("utf-8")

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=1024):
            for i in range(0, len(self.content), chunk_size):
                yield self.content[i : i + chunk_size]  # noqa

    def fake_get(url, stream=True, timeout=30):
        return FakeResponse(fake_bina_bytes)

    monkeypatch.setattr("requests.get", fake_get)


@pytest.fixture
def mock_requests_eu_html_success(monkeypatch):
    """Mock requests.get with a valid <div class='stat'> number."""
    html = """
        <html><body>
        <p>Last update: June 1, 2020 currently 5,435 planets</p>
        </body></html>
        """

    class MockResponse:
        text = html
        status_code = 200

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        "resokit.datasets.utils.requests.get", lambda *a, **k: MockResponse()
    )


@pytest.fixture
def mock_requests_eu_html_wrong(monkeypatch):
    """Mock requests.get with a valid <div class='stat'> number."""
    html = """
        <html><body>
        <p>Last update: June 1, 2020 currently bad planets</p>
        </body></html>
        """

    class MockResponse:
        text = html
        status_code = 200

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        "resokit.datasets.utils.requests.get", lambda *a, **k: MockResponse()
    )


@pytest.fixture
def mock_requests_nasa_html_success(monkeypatch):
    """Mock requests.get with a valid <div class='stat'> number."""
    html = """
        <html><body>
        <div class='stat'>5,432</div>
        <div class='date'>01/01/1990</div>
        </body></html>
        """

    class MockResponse:
        text = html

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        "resokit.datasets.utils.requests.get", lambda *a, **k: MockResponse()
    )


@pytest.fixture
def mock_requests_nasa_html_wrong(monkeypatch):
    """Mock requests.get with a valid <div class='stat'> number."""
    html = """
        <html><body>
        <div class='stat'>bad</div>
        <div class='date'>01/01/1990</div>
        </body></html>
        """

    class MockResponse:
        text = html

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        "resokit.datasets.utils.requests.get", lambda *a, **k: MockResponse()
    )


@pytest.fixture
def mock_requests_nasa_html_no_match(monkeypatch):
    """Mock requests.get with HTML that lacks the required div."""
    html = "<html><body><p>No stat here</p></body></html>"

    class MockResponse:
        text = html

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        "resokit.datasets.utils.requests.get", lambda *a, **k: MockResponse()
    )


@pytest.fixture
def mock_requests_html_failure(monkeypatch):
    """Mock requests.get to simulate a network error."""

    def mock_get(*args, **kwargs):
        raise requests.RequestException("Simulated network error")

    monkeypatch.setattr("resokit.datasets.utils.requests.get", mock_get)


@pytest.fixture
def test_dir(tmp_path):
    """Temporary directory with subdirs for datasets."""
    d = tmp_path / "datasets"
    d.mkdir()
    return d


@pytest.fixture
def mock_data_manager():
    return rk.datasets.databases.DatasetManager()


@pytest.fixture
def mock_bina_manager():
    return rk.datasets.databases.BinaryDatasetManager()


@pytest.fixture(scope="session")
def sample_eu_csv_path():
    return Path(__file__).parent / "data" / "sample_eu.csv"


@pytest.fixture(scope="session")
def sample_bin_p_txt_path():
    return Path(__file__).parent / "data" / "sample_bin_p.txt"


@pytest.fixture(scope="session")
def mock_csv_data():
    return """name,star_name,orbital_period,eccentricity,mass
109 Psc b,109 Psc,1075.4,0.104,5.743
112 Psc b,112 Psc,4.4,0.376,
112 Psc c,112 Psc,36336.7,0.174,9.866
"""


@pytest.fixture
def patch_download(monkeypatch):
    mock_csv = """name,star_name,orbital_period,eccentricity,mass
109 Psc b,109 Psc,1075.4,0.104,5.743
"""
    monkeypatch.setattr(
        "resokit.databases.request_dataset",
        lambda *args, **kwargs: mock_csv.encode("utf-8"),
    )


@pytest.fixture(scope="class")
def load_eu_data(sample_eu_csv_path):
    # load small test data
    rk.datasets.load_dataset(
        source="eu",
        from_file=sample_eu_csv_path,
        from_zip=False,
        store=True,
        verbose=False,
    )
    yield


@pytest.fixture(scope="class")
def load_binary_data(sample_bin_p_txt_path):
    # load small test data
    rk.datasets.load_binary_dataset(
        which="p", from_file=sample_bin_p_txt_path, verbose=False
    )
    yield


# # ============================================================================
# # CONSTANTS
# # ============================================================================

# default_ss = Series(data=[1, 2, 3])
# default_df = DataFrame(data=[[1, 2, 3], [4, 5, 6]])

# # ============================================================================
# # CONFIG
# # ============================================================================


# def pytest_configure(config):
#     pytest.default_ss = default_ss
#     pytest.default_df = default_df


# # ============================================================================
# # FIXTURES
# # ============================================================================


# @pytest.fixture(scope="session")
# def zip_path():
#     return (
#         pathlib.Path(__file__).parents[1]
#         / "resokit"
#         / "datasets"
#         / "datasets.zip"
#     )


# @pytest.fixture(scope="session")
# def random_int_gen():
#     def _random_int(size=1):
#         rng = np.random.default_rng(seed=42)
#         return rng.integers(1, 999999999, size=size)

#     return _random_int


# @pytest.fixture(scope="session")
# def db_temp_path(tmp_path_factory):
#     def make_temp_path(source="data", ext="csv"):
#         fn = tmp_path_factory.mktemp("tmp") / f"{source}.{ext}"
#         return fn

#     return make_temp_path


# @pytest.fixture(scope="session")
# def mock_in_mem(session_mocker):
#     # Define def ss and df
#     def _mock_in_mem(which="all"):
#         if which == "index":
#             # Mock the variables
#             session_mocker.patch(
#                 "resokit.datasets.databases._IN_MEMORY_INDEXES",
#                 {"eu": default_ss, "nasa": default_ss},
#             )
#         elif which == "parsed":
#             # Mock the variables
#             session_mocker.patch(
#                 "resokit.datasets.databases._IN_MEMORY_PARSED_INDEXES",
#                 {"eu": default_ss, "nasa": default_ss},
#             )
#         elif which == "data":
#             session_mocker.patch(
#                 "resokit.datasets.databases._IN_MEMORY_DATASETS",
#                 {"eu": default_df, "nasa": default_df},
#             )
#         elif which == "binary":
#             session_mocker.patch(
#                 "resokit.datasets.databases._IN_MEMORY_BINARIES_HEADERS",
#                 {"p": "Header_p", "s": "Header_s"},
#             )
#             session_mocker.patch(
#                 "resokit.datasets.databases._IN_MEMORY_BINARIES",
#                 {"p": default_df, "s": default_df},
#             )
#         elif which == "fully":
#             session_mocker.patch(
#                 "resokit.datasets.databases._IS_FULLY_STORED",
#                 {"eu": True, "nasa": True},
#             )
#         elif which == "all":
#             _mock_in_mem(which="index")
#             _mock_in_mem(which="data")
#             _mock_in_mem(which="fully")
#             _mock_in_mem(which="binary")
#             _mock_in_mem(which="parsed")
#         else:
#             raise ValueError(f"which={which} not recognized.")

#     return _mock_in_mem
