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

from pathlib import Path

import numpy as np

from pandas import DataFrame, Series

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

    def mock_get(url, params=None, stream=True):
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

    def fake_get(url, params=None, stream=True, timeout=30):
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
    rk.datasets.load(
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
    rk.datasets.load_binary(
        which="p", from_file=sample_bin_p_txt_path, verbose=False
    )
    yield


class MySharedK11:
    npl = 6
    pl_names = [
        "Kepler-11 b",
        "Kepler-11 c",
        "Kepler-11 d",
        "Kepler-11 e",
        "Kepler-11 f",
        "Kepler-11 g",
    ]
    data = DataFrame(
        data=[
            [  # Kepler-11 b
                6.000000e-03,
                1.610000e-01,
                1.030390e01,
                1.000000e-03,
                6.000000e-04,
            ],
            [  # Kepler-11 c
                9.000000e-03,
                2.560000e-01,
                1.302410e01,
                8.000000e-04,
                1.300000e-03,
            ],
            [  # Kepler-11 d
                2.300000e-02,
                2.780000e-01,
                2.268450e01,
                9.000000e-04,
                9.000000e-04,
            ],
            [  # Kepler-11 e
                2.500000e-02,
                3.740000e-01,
                3.199960e01,
                1.200000e-03,
                8.000000e-04,
            ],
            [  # Kepler-11 f
                6.000000e-03,
                2.220000e-01,
                4.668880e01,
                3.200000e-03,
                2.700000e-03,
            ],
            [  # Kepler-11 g
                7.900000e-02,
                2.970000e-01,
                1.183807e02,
                6.000000e-04,
                1.000000e-03,
            ],
        ],
        columns=["mass", "radius", "P", "P_err_min", "P_err_max"],
        index=pl_names,
    )
    nasa_p_err_min = [0.0006, 0.0013, 0.0009, 0.0008, 0.0027, 0.001]
    nasa_p_err_max = [0.001, 0.0008, 0.0009, 0.0012, 0.0032, 0.0006]

    perat = DataFrame(
        data=[
            [1.0, 1.26399713, 2.20154505, 3.10558138, 4.53117752, 11.48892167],
            [0.79114104, 1.0, 1.74173263, 2.45695288, 3.58480049, 9.08935742],
            [0.45422645, 0.57414093, 1.0, 1.41063722, 2.0581807, 5.21857215],
            [0.3220009, 0.40700821, 0.70889949, 1.0, 1.45904324, 3.69944312],
            [0.22069319, 0.27895555, 0.48586599, 0.68538065, 1.0, 2.53552672],
            [0.08704037, 0.11001878, 0.1916233, 0.27031095, 0.39439537, 1.0],
        ],
        columns=pl_names,
        index=pl_names,
    )

    estimated_mass = DataFrame(
        data=[
            [  # ck17
                0.011872141051038074,
                0.02609086347629103,
                0.030010804546323423,
                0.049659103103399396,
                0.020484112885561786,
                0.03357562292185889,
            ],
            [  # o20
                0.020172789575398694,
                0.09991819186770531,
                0.13279179209679726,
                0.05095991162664184,
                0.061113118825503895,
                0.0354035428798858,
            ],
            [  # e23
                0.018482520943664683,
                0.03655597714344237,
                0.0412679180806311,
                0.06383571769905695,
                0.029644772131728718,
                0.045481583779843,
            ],
            [  # m24
                0.01748027769110997,
                0.034927409114917526,
                0.03950084780072523,
                0.061501432603222425,
                0.028235645987476073,
                0.04359730370417228,
            ],
        ],
        columns=pl_names,
        index=["ck17", "o20", "e23", "m24"],
    ).T

    estimated_radii = DataFrame(
        data=[
            [  # ck17
                0.10998515,
                0.13676285,
                0.23767145,
                0.24963529,
                0.10998515,
                0.49161109,
            ],
            [  # o20
                0.11318642,
                0.1273095,
                0.1671212,
                0.17121156,
                0.11318642,
                0.23902445,
            ],
            [  # e23
                0.11027453,
                0.12353237,
                0.18680687,
                0.19770476,
                0.11027453,
                0.43231902,
            ],
            [  # m24
                0.11064982,
                0.12345122,
                0.19349331,
                0.20461059,
                0.11064982,
                0.44230155,
            ],
        ],
        columns=pl_names,
        index=["ck17", "o20", "e23", "m24"],
    ).T

    _estimated_a = [0.091, 0.107, 0.155, 0.195, 0.25, 0.466]
    _estimated_a_forced = [
        0.09142604,
        0.10688114,
        0.1547239,
        0.19461124,
        0.2503479,
        0.46551455,
    ]
    _estimated_a_err0_f = [
        [0.003, 0.003],
        [0.001, 0.001],
        [0.001, 0.001],
        [0.002, 0.002],
        [0.009, 0.009],
        [0.004, 0.004],
    ]
    _estimated_a_err1_f = [
        [0.00096839, 0.0009463],
        [0.00112965, 0.00110943],
        [0.00163307, 0.00159945],
        [0.00205392, 0.00200999],
        [0.00264699, 0.00259107],
        [np.nan, np.nan],
    ]
    _estimated_a_err2_f = [
        [0.00095237, 0.00095237],
        [0.00111336, 0.00111336],
        [0.00161168, 0.00161168],
        [0.00202716, 0.00202716],
        [0.0026078, 0.0026078],
        [np.nan, np.nan],
    ]
    _estimated_a_err3_f = [
        [0.00095236, 0.00095236],
        [0.00111335, 0.00111335],
        [0.00161168, 0.00161168],
        [0.00202715, 0.00202715],
        [0.0026078, 0.0026078],
        [np.nan, np.nan],
    ]
    estimated_a = Series(
        data=_estimated_a,
        index=pl_names,
        name="a",
    )
    estimated_a_forced = Series(
        data=_estimated_a_forced,
        index=pl_names,
        name="a",
    )

    a_with_err = DataFrame(
        data={
            "a": _estimated_a,
            "a_err_min": [e[0] for e in _estimated_a_err0_f],
            "a_err_max": [e[1] for e in _estimated_a_err0_f],
        },
        index=pl_names,
    )

    _estimated_a_erri_f_cache = None  # internal cache

    @classmethod
    def get_estimated_a_erri_f(cls):
        if cls._estimated_a_erri_f_cache is None:
            cls._estimated_a_erri_f_cache = {
                i: DataFrame(
                    data={
                        "a": cls._estimated_a_forced,
                        "a_err_min": [e[0] for e in ierr_],
                        "a_err_max": [e[1] for e in ierr_],
                    },
                    index=cls.pl_names,
                )
                for i, ierr_ in enumerate(
                    [
                        cls._estimated_a_err0_f,
                        cls._estimated_a_err1_f,
                        cls._estimated_a_err2_f,
                        cls._estimated_a_err3_f,
                    ]
                )
            }
        return cls._estimated_a_erri_f_cache

    _estimated_p = [10.3039, 13.0241, 22.6845, 31.9996, 46.6888, 118.3807]
    _estimated_p_forced = [
        10.23196061,
        13.04583225,
        22.74524654,
        32.09553259,
        46.5915098,
        118.56592364,
    ]
    _estimated_p_err0_f = [
        [0.001, 0.0006],
        [0.0008, 0.0013],
        [0.0009, 0.0009],
        [0.0012, 0.0008],
        [0.0032, 0.0027],
        [0.0006, 0.001],
    ]
    _estimated_p_err1_f = [
        [0.6503614, 0.68202358],
        [0.37890969, 0.39502443],
        [0.56370503, 0.58799598],
        [0.97508968, 1.01662931],
        [3.16651597, 3.32465226],
        [np.nan, np.nan],
    ]
    _estimated_p_err2_f = [
        [0.15987343, 0.15987343],
        [0.20383931, 0.20383931],
        [0.35538636, 0.35538636],
        [0.50148025, 0.50148025],
        [0.727988, 0.727988],
        [np.nan, np.nan],
    ]
    _estimated_p_err3_f = [
        [0.15987343, 0.15987343],
        [0.20383931, 0.20383931],
        [0.35538635, 0.35538635],
        [0.50148024, 0.50148024],
        [0.727988, 0.727988],
        [np.nan, np.nan],
    ]
    estimated_p = Series(
        data=_estimated_p,
        index=pl_names,
        name="P",
    )
    estimated_p_forced = Series(
        data=_estimated_p_forced,
        index=pl_names,
        name="P",
    )

    p_with_err = DataFrame(
        data={
            "P": _estimated_p,
            "P_err_min": [e[0] for e in _estimated_p_err0_f],
            "P_err_max": [e[1] for e in _estimated_p_err0_f],
        },
        index=pl_names,
    )

    _estimated_p_erri_f_cache = None  # internal cache

    @classmethod
    def get_estimated_p_erri_f(cls):
        if cls._estimated_p_erri_f_cache is None:
            cls._estimated_p_erri_f_cache = {
                i: DataFrame(
                    data={
                        "P": cls._estimated_p_forced,
                        "P_err_min": [e[0] for e in ierr_],
                        "P_err_max": [e[1] for e in ierr_],
                    },
                    index=cls.pl_names,
                )
                for i, ierr_ in enumerate(
                    [
                        cls._estimated_p_err0_f,
                        cls._estimated_p_err1_f,
                        cls._estimated_p_err2_f,
                        cls._estimated_p_err3_f,
                    ]
                )
            }
        return cls._estimated_p_erri_f_cache


@pytest.fixture(scope="session")
def k11():
    cls = MySharedK11()
    cls.get_estimated_a_erri_f()
    return cls


@pytest.fixture
def planet_series():
    return Series(
        {
            "name": "pl1",
            "radius": 1.0,
            "mass": 1.0,
            "star_name": "st1",
            "P": 10.0,
            "radius_err_min": -0.12,
            "radius_err_max": 0.15,
            "mass_err_min": 0.13,
            "mass_err_max": 0.14,
        }
    )


@pytest.fixture
def star_series():
    return Series({"name": "st1", "mass": 1.5})


@pytest.fixture
def simple_binary(star_series):
    st0 = rk.core.StaticStar(data=star_series, source="user", metadata={})
    st1_series = star_series.copy()
    st1_series["name"] = "st2"
    st1_series["mass"] = 0.5
    st1 = rk.core.StaticStar(data=st1_series, source="user", metadata={})
    data = Series({"a": 1.0, "e": 0.1, "P": 20.0})
    bin_sys = rk.core.StaticBinaryStar(
        star0=st0, star1=st1, data=data, name="bin1", metadata={}
    )
    return bin_sys


@pytest.fixture
def simple_system(star_series, planet_series):
    st = rk.core.StaticStar(data=star_series, source="user", metadata={})
    p1_series = planet_series.copy()
    p1_series["name"] = "pl1"
    p1_series["P"] = 10.0
    p1_series["P_err_min"] = 1.0
    p1_series["P_err_max"] = 2.0
    p1 = rk.core.StaticPlanet(data=p1_series, source="user", metadata={})
    p2_series = planet_series.copy()
    p2_series["name"] = "pl2"
    p2_series["P"] = 20.0
    p2_series["P_err_min"] = 2.0
    p2_series["P_err_max"] = 1.0
    p2 = rk.core.StaticPlanet(data=p2_series, source="user", metadata={})
    sys = rk.core.StaticSystem(
        star=st, planets=[p1, p2], name="sys1", metadata={}
    )
    return sys


@pytest.fixture
def simple_planet(planet_series):
    return rk.core.StaticPlanet(data=planet_series, source="user", metadata={})


@pytest.fixture
def simple_star(star_series):
    return rk.core.StaticStar(data=star_series, source="user", metadata={})


@pytest.fixture()
def patch_units_and_estimators(monkeypatch):
    """Default lightweight patches for convert, estimate_mass, estimate_radius.

    By default convert returns its input(s) unchanged (identity) so tests don't
    depend on real unit conversions. Specific tests can override these.
    """

    def convert_dummy(*args, **kwargs):
        # If single value -> return single value
        if len(args) == 1:
            return args[0]
        # If triple values -> return tuple of them
        if len(args) >= 3:
            return (args[0], args[1], args[2])
        # fallback
        return args if args else None

    monkeypatch.setattr(rk.core, "convert", convert_dummy)
    monkeypatch.setattr(
        rk.core, "estimate_mass", lambda **kwargs: (100.0, 90.0, 110.0)
    )
    monkeypatch.setattr(
        rk.core, "estimate_radius", lambda **kwargs: (1.0, 0.9, 1.1)
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
