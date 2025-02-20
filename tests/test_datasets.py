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

import zipfile
from pathlib import Path

import numpy as np

import pandas as pd

import pytest
from pytest import default_df, default_ss

from resokit.datasets import databases, utils
from resokit.utils.parser import assert_module_imported

# ============================================================================
# TESTS
# ============================================================================


class TestPath:
    def test_zip_path(self):
        """Test the zip_path variable."""
        zip_path = databases.BASE_PATH / utils.ZIP_FILENAME
        # Check if the zip file exists
        assert zipfile.is_zipfile(zip_path)
        # Check if the zip file is not empty
        assert zip_path.stat().st_size > 0
        # Check the zipfile has all files:
        #  "eu.csv", "nasa.csv", "plan_bin500au.txt", "plan_circ.txt"
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            assert "exoplanet_eu.csv" in zip_ref.namelist()
            assert "nasa.csv" in zip_ref.namelist()
            assert "plan_bin500au.txt" in zip_ref.namelist()
            assert "plan_circ.txt" in zip_ref.namelist()


class TestClearMemory:
    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_clear_memory(self, source: str, mock_in_mem):
        """Test the clear_memory function."""
        # Mock the variables
        mock_in_mem(which="all")

        # Clear the memory
        databases.clear_memory(source=source, verbose=False)

        # Assert the type of the new objects
        assert isinstance(
            databases._IN_MEMORY_INDEXES[source], databases.ResoKitDataset
        )
        assert isinstance(
            databases._IN_MEMORY_DATASETS[source], databases.ResoKitDataset
        )

        # Check if the dictionary of the indexes is empty
        assert databases._IN_MEMORY_INDEXES[source].empty

        # Check if the dictionary of the datasets is empty
        assert databases._IN_MEMORY_DATASETS[source].empty

        # Check if the variable _IS_FULLY_STORED is False
        assert databases._IS_FULLY_STORED[source] is False

        # Check the other source is not touched
        other = "eu" if source == "nasa" else "nasa"
        pd.testing.assert_series_equal(
            databases._IN_MEMORY_INDEXES[other], default_ss
        )
        pd.testing.assert_frame_equal(
            databases._IN_MEMORY_DATASETS[other], default_df
        )
        assert databases._IS_FULLY_STORED[other]

    def test_clear_memory_both(self, mock_in_mem):
        """Test the clear_memory function with both sources."""
        # Mock the variables
        mock_in_mem(which="all")

        # Clear the memory
        databases.clear_memory(source="both", verbose=False)

        # Assert the type of the new objects
        assert isinstance(
            databases._IN_MEMORY_INDEXES["eu"], databases.ResoKitDataset
        )
        assert isinstance(
            databases._IN_MEMORY_DATASETS["eu"], databases.ResoKitDataset
        )
        assert isinstance(
            databases._IN_MEMORY_INDEXES["nasa"], databases.ResoKitDataset
        )
        assert isinstance(
            databases._IN_MEMORY_DATASETS["nasa"], databases.ResoKitDataset
        )

        # Check if the dictionary of the indexes is empty
        assert databases._IN_MEMORY_INDEXES["eu"].empty
        assert databases._IN_MEMORY_INDEXES["nasa"].empty

        # Check if the dictionary of the datasets is empty
        assert databases._IN_MEMORY_DATASETS["eu"].empty
        assert databases._IN_MEMORY_DATASETS["nasa"].empty

        # Check if the variable _IS_FULLY_STORED is False
        assert databases._IS_FULLY_STORED["eu"] is False
        assert databases._IS_FULLY_STORED["nasa"] is False


class TestLoadDataset:
    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_naive(self, zip_path: str, source: str):
        """Test the load function with the naive approach."""
        # Clear the memory
        databases.clear_memory(source=source, verbose=False)

        # Load the dataset
        data = databases.load_full(
            source=source,
            from_memory=False,
            from_zip=True,
            from_file=False,
            to_resokit=False,
            to_df=True,
            verbose=False,
        )

        # Check if the data is a DataFrame
        assert isinstance(data, pd.DataFrame)

        # Check is not empty
        assert not data.empty

        # Load the dataset from the zip file with pandas
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            this_source = "exoplanet_" + source if source == "eu" else source
            with zip_ref.open(this_source + ".csv") as filestream:
                if source == "nasa":
                    with pytest.warns(pd.errors.DtypeWarning):
                        df = pd.read_csv(filestream)
                else:
                    df = pd.read_csv(filestream)

        # Check if the data is equal to the DataFrame loaded from the zip file
        pd.testing.assert_frame_equal(
            data.iloc[:, :49], df.iloc[:, :49], check_dtype=False
        )
        # [:,:49] to avoid the obj columns that are same, but different dtype.

        # Check some columns
        if source == "nasa":
            assert data.columns[0] == "pl_name"
            assert data.columns[1] == "pl_letter"
            assert data.columns[2] == "hostname"
        else:
            assert data.columns[0] == "name"
            assert data.columns[1] == "planet_status"
            assert data.columns[2] == "mass"

    @pytest.mark.parametrize("source", ["eu", "nasa"])
    @pytest.mark.parametrize("to_resokit", [True, False])
    def test_load_full(self, source: str, to_resokit: bool):
        """Test the load_full function, with same result as individual."""
        # Clear the memory
        databases.clear_memory(source=source, verbose=False)

        # Load the dataset
        if source == "nasa":
            data = databases.load_nasa(to_resokit=to_resokit, verbose=False)
        else:
            data = databases.load_eu(to_resokit=to_resokit, verbose=False)

        # Load the dataset
        data2 = databases.load_full(
            source=source,
            from_memory=False,
            from_zip=True,
            from_file=False,
            to_resokit=to_resokit,
            verbose=False,
        )

        # Check if a ResokitDataset
        assert isinstance(data, databases.ResoKitDataset)
        assert isinstance(data2, databases.ResoKitDataset)

        # Check if the datasets of data is equal
        pd.testing.assert_frame_equal(data.dataset, data2.dataset)

    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_store(self, source: str, mocker):
        """Test the load function with the in-memory cache features.

        - Test if nothing is saved. (store_index=False)
        - Test if the index is saved and the dataset is not. (Default)
        - Test if the index and dataset is saved. (store=True)
        - Test if the datasets is loaded from the cache if saved.
        """
        # Clear the memory
        databases.clear_memory(source=source, verbose=False)

        # Index columns saved
        if source == "nasa":
            saved_index_cols = ["pl_name", "hostname"]
        else:
            saved_index_cols = ["name", "star_name"]

        # -------------------------------------------------------------------
        # Nothing is saved. (store_index=False)
        # -------------------------------------------------------------------

        # Load the dataset without saving
        data = databases.load_full(
            source=source, store_index=False, verbose=False, to_resokit=False
        )

        # Check the dictionary of the indexes if empty
        assert databases._IN_MEMORY_INDEXES[source].empty
        assert databases._IN_MEMORY_DATASETS[source].empty
        assert not databases._IS_FULLY_STORED[source]

        # -------------------------------------------------------------------
        # The index is saved and the dataset is not. (Default)
        # -------------------------------------------------------------------

        # Load the dataset again
        data = databases.load_full(
            source=source, verbose=True, to_resokit=False
        )

        # Check the dictionary of the indexes if filled
        assert not databases._IN_MEMORY_INDEXES[source].empty
        pd.testing.assert_frame_equal(
            data.dataset[saved_index_cols],
            databases._IN_MEMORY_INDEXES[source].dataset,
        )

        # Check the dictionary of the datasets is empty
        assert databases._IN_MEMORY_DATASETS[source].empty
        assert not databases._IS_FULLY_STORED[source]

        # -------------------------------------------------------------------
        # The index and dataset is saved. (store=True)
        # -------------------------------------------------------------------

        # Load the dataset again
        data = databases.load_full(
            source=source, store=True, verbose=False, to_resokit=False
        )

        # Check the dictionary of the indexes is not empty
        assert not databases._IN_MEMORY_INDEXES[source].empty

        # Check the dictionary of the datasets if filled
        pd.testing.assert_frame_equal(
            data.dataset, databases._IN_MEMORY_DATASETS[source].dataset
        )

        # Check the variable _IS_FULLY_STORED
        assert databases._IS_FULLY_STORED[source]

        # -------------------------------------------------------------------
        # The datasets is loaded from the cache if saved
        # -------------------------------------------------------------------

        # Mock the zip file
        mocker.patch("resokit.datasets.utils.ZIP_FILENAME", "wrong.zip")

        # Load the dataset again
        data2 = databases.load_full(
            source=source, verbose=False, to_resokit=False
        )

        # Check the dictionary of the indexes if filled
        assert not databases._IN_MEMORY_INDEXES[source].empty

        # Check if the data is equal to loaded from the original zip file
        pd.testing.assert_frame_equal(data.dataset, data2.dataset)

    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_only_index(self, source: str, mocker):
        """Test the load function with the only_index parameter.

        - Test if only the indexes are loaded. (only_index=True)
        - Test if the indexes are loaded from the cache if saved.
        """
        # Clear the memory
        databases.clear_memory(source=source, verbose=False)

        # -------------------------------------------------------------------
        # Only the indexes are loaded. (only_index=True)
        # -------------------------------------------------------------------

        # Load the dataset
        index1 = databases.load_full(source=source, only_index=True)

        # Check if the data is a ResokitDataset
        assert isinstance(index1, databases.ResoKitDataset)

        # Check the index columns
        assert index1.columns.tolist() == ["name", "star_name"]

        # Check the shape
        assert index1.shape[1] == 2

        # -------------------------------------------------------------------
        # The indexes are loaded from the cache if saved
        # -------------------------------------------------------------------

        # Mock the zip file
        mocker.patch("resokit.datasets.utils.ZIP_FILENAME", "wrong.zip")

        # Load the dataset again
        index2 = databases.load_full(
            source=source, only_index=True, verbose=False
        )

        # Check if the index dictionary is not empty
        assert not databases._IN_MEMORY_INDEXES[source].empty

        # Check if the data is equal to loaded from the original zip file
        pd.testing.assert_frame_equal(index1.dataset, index2.dataset)

    # @pytest.mark.parametrize("random_seed", random_int(2))
    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_only_rows(self, random_int_gen: int, source: str, mocker):
        """Test the load function with the only_rows parameter.

        - Test ValueError if True. (only_rows=True)
        - Test ValueError if used with only_index=True.
        - Test if only the rows are loaded. (only_rows=<int>)
        - Test if the rows are loaded from the cache if saved.
        - Test empty df if the number of rows is greater than the dataset.
        """
        # Clear the memory
        databases.clear_memory(source=source, verbose=False)

        # Get the dataset
        data = databases.load_full(
            source=source, store_index=False, store=False, verbose=False
        )

        # Define the row numbers
        # Use rng
        random_seed = random_int_gen()[0]
        rng = np.random.default_rng(seed=random_seed)
        good_row = rng.integers(low=0, high=data.shape[0]).tolist()
        bad_row = rng.integers(low=data.shape[0], high=999999999).tolist()

        # -------------------------------------------------------------------
        # ValueError if True. (only_rows=True)
        # -------------------------------------------------------------------

        # Load the dataset
        with pytest.raises(ValueError):
            databases.load_full(source=source, only_rows=True, verbose=False)

        # -------------------------------------------------------------------
        # ValueError if used with only_index=True.
        # -------------------------------------------------------------------

        # Load the dataset
        with pytest.raises(ValueError):
            databases.load_full(
                source=source,
                only_index=True,
                only_rows=good_row,
                verbose=False,
            )

        # -------------------------------------------------------------------
        # Only the rows are loaded. (only_rows=<int>)
        # -------------------------------------------------------------------

        # Load the dataset
        data1 = databases.load_full(
            source=source, only_rows=good_row, verbose=False
        )

        # Check the values
        pd.testing.assert_series_equal(data.loc[good_row], data1.squeeze())

        # -------------------------------------------------------------------
        # The rows are loaded from the cache if saved.
        # -------------------------------------------------------------------

        # Check the dictionary of the indexes are empty
        assert databases._IN_MEMORY_INDEXES[source].empty
        assert databases._IN_MEMORY_DATASETS[source].empty
        assert not databases._IS_FULLY_STORED[source]

        # Load the dataset and store a row
        row1 = databases.load_full(
            source=source,
            only_rows=good_row,
            store=True,
            verbose=False,
            to_resokit=False,
        )

        # Check the stored dataset is not empty
        assert not databases._IN_MEMORY_DATASETS[source].empty
        assert not databases._IS_FULLY_STORED[source]

        # Check if properly stored
        pd.testing.assert_frame_equal(
            row1.dataset, databases._IN_MEMORY_DATASETS[source].dataset
        )

        # Mock the zip file
        mocker.patch("resokit.datasets.utils.ZIP_FILENAME", "wrong.zip")

        # Load with only_rows
        row2 = databases.load_full(
            source=source,
            only_rows=good_row,
            verbose=False,
            to_resokit=False,
        )

        # Check if it is the same as before
        pd.testing.assert_frame_equal(row1.dataset, row2.dataset)

        # -------------------------------------------------------------------
        # Empty df if the number of rows is greater than the dataset.
        # -------------------------------------------------------------------

        # Clear the memory
        databases.clear_memory(source=source, verbose=False)

        # Reset the zip file
        mocker.patch("resokit.datasets.utils.ZIP_FILENAME", "datasets.zip")

        # Load the dataset
        with pytest.raises(ValueError):
            databases.load_full(
                source=source, only_rows=bad_row, store=False, verbose=False
            )

        # Load the dataset without to_resokit
        with pytest.warns(UserWarning):
            data3 = databases.load_full(
                source=source,
                only_rows=bad_row,
                store=False,
                verbose=False,
                to_resokit=False,
            )

        # Check if the data is empty
        assert data3.empty
        assert data3.shape == (0, row2.shape[1])


class TestCheckFileAge:
    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_check_file_age(self, source: str):
        """Test the check_file_age function."""
        # Clear the memory
        databases.clear_memory(source=source, verbose=False)

        # Get the original ZIP path
        zip_path = databases.BASE_PATH / utils.ZIP_FILENAME

        # GEt the original filename
        filename = databases._DATASET_FILENAMES[source]

        # Check the file age
        age = databases._check_file_age(
            file_path=filename,
            zip_path=zip_path,
            verbose=False,
        )

        # Check if the age is a float
        assert isinstance(age, int)

        # Check if the age is greater than 0
        assert age > 0

        # Try to check the age of a non-existing file
        with pytest.raises(FileNotFoundError):
            databases._check_file_age(
                file_path="thisfiledoesnotexists.csv",
                zip_path=zip_path,
                verbose=False,
            )


class TestDownloadDataset:
    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_download_y_requests(self, source: str, db_temp_path: str):
        """Test the download function when requests is installed."""
        if not utils.requests_imported:
            pytest.skip("requests is not installed.")

        # Clear the memory
        databases.clear_memory(source=source, verbose=False)

        # If no destination, raise an error
        with pytest.raises(ValueError):
            databases.download(source=source, to_memory=False, verbose=False)

        # Load the dataset
        data = databases.load_full(
            source=source, store_index=False, verbose=False
        )

        # Get a temp file path
        tmp_path = db_temp_path(source)

        # Set destiny path
        path = Path(tmp_path)
        # path = databases.BASE_PATH / databases._DATASET_FILENAMES[source]

        # CHECK IF THE FILE DOES NOT EXISTS
        assert not path.exists()

        # Download the dataset
        data2 = databases.download(
            source=source,
            to_resokit=True,
            to_file=path,
            verbose=False,
            check_online=False,
        )

        # Check if the file is saved
        assert path.exists()

        # Can't download the dataset if the file already exists in memory
        # (overwrite=False) is the default.
        with pytest.raises(ValueError):
            databases.download(source=source, verbose=False)

        # Can't download the dataset if the file already exists in file
        # (overwrite=False) is the default.
        with pytest.raises(FileExistsError):
            databases.download(source=source, verbose=False, to_file=path)

        # Load the dataset (from the new file because zip is wrong)
        data2 = databases.load_full(
            source=source, store_index=False, verbose=False
        )

        # Check if the data is equal or longer than the original
        assert data2.shape[0] >= data.shape[0]
        assert data2.shape[1] == data.shape[1]
        assert data2.columns.tolist() == data.columns.tolist()

        # Remove the file
        path.unlink()

    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_download_n_requests(self, source: str, mocker):
        """Test the download function when requests is not installed."""

        # Clear the memory
        databases.clear_memory(source=source, verbose=False)

        # Mock the requests_imported variable of resokit
        mocker.patch("resokit.datasets.utils.requests_imported", False)

        # Mock the assert_module_imported function
        new_defauts = ("", False, None, None)  # retry=False
        mocker.patch.object(
            assert_module_imported,
            "__defaults__",
            new_defauts,
        )

        with pytest.raises(ImportError):
            databases.download(source=source, to_memory=True, verbose=False)
