#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2025, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

import zipfile
from pathlib import Path

import pandas as pd

import pytest

import resokit.datasets.databases as rdb
from resokit.datasets.databases import (
    BINARIES_FILENAMES,
    DATASET_FILENAMES,
    DATASET_ZIPNAMES,
)


class TestClearMemory:
    @pytest.mark.parametrize("source", ["eu", "nasa", "datasets", "all"])
    def test_clear_memory(self, source, fake_df):
        for src in ["eu", "nasa"]:
            # Create ResokiDataset
            rkset = rdb._df_to_dataset(
                fake_df, src, 3, origin="mixed", is_full=True
            )

            rdb._full_manager._datasets[src] = rkset.copy()
            rdb._full_manager._indexes[src] = rkset[
                ["name", "star_name"]
            ].copy()
            rdb._full_manager._is_fully_stored[src] = True
            rdb._full_manager._parsed_indexes[src] = rkset[
                ["name", "star_name"]
            ].copy()

            # Sanity check that memory is set
            assert not rdb._full_manager._datasets[src].dataset.empty

        # Clear memory using the external function
        rdb.clear_memory(source, verbose=False)

        if source in ["eu", "nasa"]:
            # Now check memory is cleared
            assert rdb._full_manager._datasets[source].dataset.empty
            assert rdb._full_manager._indexes[source].dataset.empty
            assert not rdb._full_manager._is_fully_stored[source]
            assert rdb._full_manager._parsed_indexes[source] is None

            other = "nasa" if source == "eu" else "eu"
            assert len(rdb._full_manager._datasets[other].to_dataframe()) == 2
            assert rdb._full_manager._is_fully_stored[other]

        else:
            for source in ["nasa", "eu"]:
                # Now check memory is cleared
                assert rdb._full_manager._datasets[source].dataset.empty
                assert rdb._full_manager._indexes[source].dataset.empty
                assert not rdb._full_manager._is_fully_stored[source]
                assert rdb._full_manager._parsed_indexes[source] is None

    @pytest.mark.parametrize("source", ["p", "s", "binary", "all"])
    def test_clear_memory_bina(self, source, fake_df, fake_header):
        for src in ["p", "q"]:
            # Create ResokiDataset

            rdb._binary_manager._datasets[src] = fake_df.copy()
            rdb._binary_manager._headers[src] = fake_header

            # Sanity check that memory is set
            assert not rdb._binary_manager._datasets[src].empty

        # Clear memory using the external function
        rdb.clear_memory(source, verbose=False)

        if source in ["p", "s"]:
            # Now check memory is cleared
            assert rdb._binary_manager._datasets[source].empty
            assert rdb._binary_manager._headers[source] == ""

            other = "p" if source == "s" else "q"
            assert len(rdb._binary_manager._datasets[other]) == 2
            assert rdb._binary_manager._headers[other] == fake_header

        else:
            for source in ["p", "s"]:
                # Now check memory is cleared
                assert rdb._binary_manager._datasets[source].empty
                assert rdb._binary_manager._headers[source] == ""

    @pytest.mark.parametrize("source", ["nasa", "eu", "datasets", "all"])
    def test_clear_memory_removes_zip(self, tmp_path, source, monkeypatch):
        path_list = {}
        for ss in ["eu", "nasa"]:
            zip_path = tmp_path / DATASET_ZIPNAMES[ss]
            path_list[ss] = zip_path
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.writestr(ss, "name,star_name\nFake,FakeStar")

            # Verify file exists inside ZIP
            with zipfile.ZipFile(zip_path, "r") as zipf:
                assert ss in zipf.namelist()

        # Monkeypatch the datasets directory to our temp path
        monkeypatch.setattr(
            "resokit.datasets.databases.DATASETS_DIR", tmp_path
        )

        # Call clear_memory with files=True
        rdb.clear_memory(source, files=True, verbose=False)

        # Verify file is gone from ZIP
        if source == "eu":
            assert not path_list["eu"].exists()
            assert path_list["nasa"].exists()
        elif source == "nasa":
            assert not path_list["nasa"].exists()
            assert path_list["eu"].exists()
        else:
            assert not path_list["nasa"].exists()
            assert not path_list["eu"].exists()

    @pytest.mark.parametrize("source", ["p", "s", "binary", "all"])
    def test_clear_memory_removes_bina_file(
        self, tmp_path, source, monkeypatch
    ):
        path_list = {}
        for ss in ["p", "s"]:
            file_path = tmp_path / BINARIES_FILENAMES[ss]
            path_list[ss] = file_path
            with open(file_path, "w") as f:
                f.write("name,star_name\nFake,FakeStar")

            # Verify file exists inside ZIP
            assert file_path.exists()

        # Monkeypatch the datasets directory to our temp path
        monkeypatch.setattr(
            "resokit.datasets.databases.DATASETS_DIR", tmp_path
        )

        # Call clear_memory with files=True
        rdb.clear_memory(source, files=True, verbose=False)

        # Verify file is gone from ZIP
        if source == "p":
            assert not path_list["p"].exists()
            assert path_list["s"].exists()
        elif source == "s":
            assert not path_list["s"].exists()
            assert path_list["p"].exists()
        else:
            assert not path_list["s"].exists()
            assert not path_list["p"].exists()


class TestDownloadAndLoadDataset:
    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_download_to_file(self, source, mock_requests, test_dir):
        file_path = test_dir / DATASET_FILENAMES[source]
        result = rdb.download(
            source,
            to_file=True,
            to_zip=False,
            dir_path=test_dir,
            to_memory=False,
            overwrite=True,
        )
        assert file_path.exists()
        assert isinstance(result, Path)
        assert file_path.read_text().startswith("pl_name")

    def test_download_and_load_from_zip(self, mock_requests, test_dir):
        source = "nasa"
        zip_path = test_dir / DATASET_ZIPNAMES[source]
        rdb.download(
            source,
            to_zip=True,
            dir_path=test_dir,
            to_memory=False,
            overwrite=False,
        )
        # assert zip_path.exists()
        with zipfile.ZipFile(zip_path) as z:
            assert DATASET_FILENAMES[source] in z.namelist()

        df = rdb.load(
            source=source,
            from_memory=False,
            from_file=DATASET_FILENAMES[source],
            from_zip=zip_path,
            dir_path=test_dir,
            to_resokit=False,
            to_df=True,
        )

        assert isinstance(df, pd.DataFrame)
        assert "pl_name" in df.columns

    def test_download_and_overwrite_from_zip(self, mock_requests, test_dir):
        source = "nasa"
        zip_path = test_dir / DATASET_ZIPNAMES[source]
        rdb.download(
            source,
            to_zip=True,
            dir_path=test_dir,
            to_memory=False,
            overwrite=False,
        )
        # edit the whole zifile
        with zipfile.ZipFile(
            zip_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as z:
            z.writestr(DATASET_FILENAMES[source], "TEST DATA")

        # assert zip_path.exists()
        with zipfile.ZipFile(zip_path) as zipread:
            inside = False
            for item in zipread.infolist():
                print(item.filename)
                if item.filename == DATASET_FILENAMES[source]:
                    inside = True
            assert inside

        # overwrite
        rdb.download(
            source,
            to_zip=True,
            dir_path=test_dir,
            to_memory=False,
            overwrite=True,
        )

        # load from path
        df = rdb.load(
            source=source,
            from_memory=False,
            from_file=True,
            from_zip=zip_path,
            dir_path=False,
            to_resokit=False,
            to_df=True,
        )

        assert isinstance(df, pd.DataFrame)
        assert "pl_name" in df.columns

    def test_download_and_load_as_resokit(self, mock_requests):
        result = rdb.download(
            "nasa", to_memory=True, to_resokit=True, to_file=False
        )
        assert hasattr(result, "to_dataframe")
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_load_only_index(self, sample_eu_csv_path):
        """Test the load function with the only_index parameter.

        - Test if only the indexes are loaded. (only_index=True)
        - Test if the indexes are loaded from the cache if saved.
        """
        source = "eu"
        # Clear the memory
        rdb.clear_memory(which=source, verbose=False)

        # -------------------------------------------------------------------
        # Only the indexes are loaded. (only_index=True)
        # -------------------------------------------------------------------

        # Load the dataset
        index1 = rdb.load(
            source=source,
            only_index=True,
            from_file=sample_eu_csv_path,
            from_zip=False,
        )

        # Check if the data is a ResokitDataset
        assert isinstance(index1, rdb.ResoKitDataset)

        # Check the index columns
        assert index1.columns.tolist() == ["name", "star_name"]

        # Check the shape
        assert index1.shape == (18, 2)

    def test_load_only_rows(self, sample_eu_csv_path):
        """Test the load function with the only_index parameter.

        - Test if only the indexes are loaded. (only_index=True)
        - Test if the indexes are loaded from the cache if saved.
        """
        source = "eu"

        # Clear the memory
        rdb.clear_memory(which=source, verbose=False)

        # Get original data
        data = pd.read_csv(sample_eu_csv_path, header=0)

        good_rows = [2, 4]

        # -------------------------------------------------------------------
        # ValueError if True. (only_rows=True)
        # -------------------------------------------------------------------

        # Load the dataset
        with pytest.raises(ValueError):
            rdb.load(source=source, only_rows=True, verbose=False)

        # -------------------------------------------------------------------
        # ValueError if used with only_index=True.
        # -------------------------------------------------------------------

        # Load the dataset
        with pytest.raises(ValueError):
            rdb.load(
                source=source,
                from_zip=False,
                from_file=sample_eu_csv_path,
                only_index=True,
                only_rows=good_rows,
                verbose=False,
            )

        # -------------------------------------------------------------------
        # Only the rows are loaded. (only_rows=<int>)
        # -------------------------------------------------------------------

        # Load the dataset
        rows1 = rdb.load(
            source=source,
            only_rows=good_rows,
            from_zip=False,
            from_file=sample_eu_csv_path,
            verbose=False,
            to_resokit=False,
            to_df=True,
        )

        # Check if the data is a ResokitDataset
        assert isinstance(rows1, pd.DataFrame)

        # Check the index columns
        for col in ["name", "mass_error_max"]:
            assert col in rows1.columns.tolist()

        # Check the shape
        assert rows1.shape == (2, 98)

        # Check the values
        pd.testing.assert_frame_equal(
            data.loc[good_rows], rows1, check_dtype=False
        )

        # -------------------------------------------------------------------
        # The rows are loaded from the cache if saved.
        # -------------------------------------------------------------------

        rdb.clear_memory(source)

        # Check the dictionary of the indexes are empty
        assert not rdb._full_manager._is_fully_stored[source]
        assert rdb._full_manager._datasets[source].empty
        assert rdb._full_manager._indexes[source].empty

        # Load the dataset and store a row
        rdb.load(
            source=source,
            only_rows=good_rows,
            store=True,
            verbose=False,
            to_resokit=False,
            from_file=sample_eu_csv_path,
            from_zip=False,
        )

        # Check the stored dataset is not empty
        assert not rdb._full_manager._datasets[source].empty
        assert not rdb._full_manager._is_fully_stored[source]

        len_old = rdb._full_manager._datasets[source].shape[0]

        # Load with only_rows
        rdb.load(
            source=source,
            only_rows=3,
            verbose=False,
            to_resokit=False,
            to_df=True,
            from_file=sample_eu_csv_path,
            from_zip=False,
            store=True,
        )

        # Check the length
        assert rdb._full_manager._datasets[source].shape[0] == len_old + 1

        # -------------------------------------------------------------------
        # The rows can be overwritten
        # -------------------------------------------------------------------

        # Load with only_rows
        rdb.load(
            source=source,
            only_rows=3,
            verbose=False,
            to_resokit=False,
            to_df=True,
            from_file=sample_eu_csv_path,
            from_zip=False,
            store=True,
            store_index=True,
        )

        # Check the length
        assert rdb._full_manager._datasets[source].shape[0] == len_old + 1

        # -------------------------------------------------------------------
        # If assumed full, load the whole dataset
        # -------------------------------------------------------------------

        rdb._full_manager._is_fully_stored[source] = True

        df = rdb.load(source)

        assert isinstance(df, rdb.ResoKitDataset)
        assert len(df) == len_old + 1

        # Final memory clear
        rdb.clear_memory(source)

    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_error(self, source, test_dir):
        wrong_filepath = test_dir / "wrong.csv"
        with pytest.raises(FileNotFoundError):
            rdb.load(
                source,
                from_memory=False,
                from_zip=False,
                from_file=wrong_filepath,
            )


class TestDownloadAndLoadBinary:
    @pytest.mark.parametrize("source", ["p", "s"])
    def test_download(self, source, mock_requests_bina, test_dir):
        file_path = test_dir / BINARIES_FILENAMES[source]
        result = rdb.download_binary(
            which=source,
            to_file=file_path,
            dir_path=test_dir,
            to_memory=False,
            overwrite=True,
            return_data=False,
        )
        assert file_path.exists()
        assert isinstance(result, Path)
        assert file_path.read_text().startswith("===")

    def test_load_p(self, sample_bin_p_txt_path):
        df = rdb.load_binary(
            which="p",
            from_memory=False,
            from_file=sample_bin_p_txt_path,
            dir_path=False,
        )

        assert isinstance(df, pd.DataFrame)

    @pytest.mark.parametrize("source", ["p", "s"])
    def test_load_error(self, source, test_dir):
        wrong_filepath = test_dir / "wrong.csv"
        with pytest.raises(FileNotFoundError):
            rdb.load_binary(
                source,
                from_memory=False,
                from_file=wrong_filepath,
            )


class TestCheckOnline:
    def test_check_outdated_eu_success(self, mock_requests_eu_html_success):
        result = rdb.check_outdated_dataset("eu", verbose=True)
        assert result[0] == 5435
        assert result[1] > 300

    def test_check_outdated_nasa_success(
        self, mock_requests_nasa_html_success
    ):
        result = rdb.check_outdated_dataset("nasa", verbose=True)
        assert result[0] == 5432
        assert result[1] > 12000

    def test_check_outdated_eu_wrong(self, mock_requests_eu_html_wrong):
        result = rdb.check_outdated_dataset("eu", verbose=True)
        assert result[0] == -1
        assert result[1] is None

    def test_check_outdated_nasa_wrong(self, mock_requests_nasa_html_wrong):
        result = rdb.check_outdated_dataset("nasa", verbose=True)
        assert result[0] == -1
        assert result[1] > 12000

    def test_check_outdated_nasa_no_match(
        self, mock_requests_nasa_html_no_match
    ):
        result = rdb.check_outdated_dataset("nasa", verbose=True)
        assert result[0] == -1

    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_check_outdated_nasa_failure(
        self, source, mock_requests_html_failure
    ):
        result = rdb.check_outdated_dataset(source, verbose=True)
        assert result[0] == -1
        assert result[1] is None

    def test_check_outdated_invalid_source(self):
        with pytest.raises(ValueError):
            rdb.check_outdated_dataset("Z", verbose=True)

    @pytest.mark.parametrize("source", ["p", "s"])
    def test_check_outdated_binary_valid_sources(
        self, source, fake_requests_success
    ):
        assert rdb.check_outdated_binary(source, verbose=True) == 3

    @pytest.mark.parametrize("source", ["p", "s"])
    def test_check_outdated_binary_failure(
        self, source, fake_requests_failure
    ):
        count = rdb.check_outdated_binary(source, verbose=True)
        assert count == -1

    def test_check_outdated_binary_invalid_source(self):
        with pytest.raises(ValueError):
            rdb.check_outdated_binary("Z", verbose=True)


class TestQuery:
    def test_query_new_all_branches(self, monkeypatch, load_eu_data):
        mgr = rdb.DatasetManager()

        def fake_load(source, **kwargs):
            if source == "empty":
                # Trigger IndexError branch
                return pd.DataFrame(columns=["rowupdate"])
            if source == "eu":
                return pd.DataFrame(
                    {
                        "updated": ["2023-01-01"],
                        "modification_date": ["2023-01-01"],
                    }
                )
            # Default: nasa
            return pd.DataFrame({"rowupdate": ["2023-01-01"]})

        monkeypatch.setattr(mgr, "load", fake_load)

        # --- Mock build_query and execute_query
        monkeypatch.setattr(
            rdb, "build_query", lambda **kw: f"QUERY for {kw['source']}"
        )
        monkeypatch.setattr(
            rdb,
            "execute_query",
            lambda **kw: pd.DataFrame(
                {"rowupdate": ["2024-01-01"], "extra": [1]}
            ),
        )

        out = mgr.query_new("nasa", verbose=False)
        assert isinstance(out, pd.DataFrame)
        assert "rowupdate" in out.columns

        with pytest.raises(ValueError):
            mgr.query_new("invalid")

        assert rdb.query_new_rows("eu") is not None


class TestBinaryOutdated:
    @pytest.mark.parametrize("source", ["p", "s", True, False, "both"])
    def test_check_binary_outdated_all_paths(
        self, monkeypatch, source, capsys
    ):
        """Cover every logical path in check_binary_outdated."""
        # --- Mock global constants
        monkeypatch.setattr(
            "resokit.datasets.databases.BINARIES_FILENAMES",
            {"p": "filep", "s": "files"},
        )
        monkeypatch.setattr(
            "resokit.datasets.databases.DATASET_FILENAMES",
            {"planets": "planets.csv"},
        )

        # --- Mock _binary_manager.load behavior
        class DummyManager:
            def __init__(self):
                self.calls = 0

            def load(self, source, ret_header):
                self.calls += 1
                if ret_header:
                    return "header\nrow1\nrow2"
                else:
                    return pd.DataFrame({"a": [1, 2]})

        dummy = DummyManager()
        monkeypatch.setattr(rdb, "_binary_manager", dummy)

        # --- Mock check_outdated_binary
        monkeypatch.setattr(rdb, "check_outdated_binary", lambda **k: 10)

        result = rdb.check_binary_outdated(which=source, verbose=False)
        # both → tuple, others → bool
        if source in ("both",):
            assert isinstance(result, tuple)
        else:
            assert result is True

        monkeypatch.setattr(rdb, "check_outdated_binary", lambda **k: 5)
        assert rdb.check_binary_outdated(which="p", verbose=False) is False

        monkeypatch.setattr(rdb, "check_outdated_binary", lambda **k: 0)
        assert rdb.check_binary_outdated(which="p", verbose=False) is True

        monkeypatch.setattr(rdb, "check_outdated_binary", lambda **k: 1)
        assert rdb.check_binary_outdated(which="p", verbose=False) is False

        def bad_load(*a, **k):
            raise FileNotFoundError("missing")

        monkeypatch.setattr(dummy, "load", bad_load)
        assert (
            rdb.check_binary_outdated(which="p", verbose=False, soft=True)
            is True
        )
        with pytest.raises(FileNotFoundError):
            rdb.check_binary_outdated(which="p", verbose=False, soft=False)

        with pytest.raises(ValueError):
            rdb.check_binary_outdated(which="invalid", verbose=False)

        rdb.BINARIES_FILENAMES = {"p": "filep", "s": "files"}
        rdb.DATASET_FILENAMES = {"invalid": "something"}
        with pytest.raises(ValueError):
            rdb.check_binary_outdated(which="invalid", verbose=True)

        cap = capsys.readouterr()
        assert "dataset" in cap.out or cap.out == ""
