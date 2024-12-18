#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2024, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================

import zipfile

# import filecmp

import numpy as np

import pandas as pd

import pytest

from resokit.datasets import databases

# ============================================================================
# TESTS
# ============================================================================


class TestPath:
    def test_zip_path(self):
        """Test the zip_path variable."""
        zip_path = databases.BASE_PATH / databases._ZIP_FILENAME
        # Check if the zip file exists
        assert zipfile.is_zipfile(zip_path)
        # Check if the zip file is not empty
        assert zip_path.stat().st_size > 0
        # Check the zipfile has both ("eu.csv", "nasa.csv") files
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            assert "exoplanet_eu.csv" in zip_ref.namelist()
            assert "nasa.csv" in zip_ref.namelist()


class TestLoadDataset:
    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_naive(self, zip_path: str, skip_rows: dict, source: str):
        """Test the load function with the naive approach."""
        # Load the dataset
        data = databases.load(source=source)

        # Check if the data is a pandas DataFrame
        assert isinstance(data, pd.DataFrame)

        # Load the dataset from the zip file with pandas
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            this_source = "exoplanet_" + source if source == "eu" else source
            with zip_ref.open(this_source + ".csv") as filestream:
                if source == "nasa":
                    with pytest.warns(pd.errors.DtypeWarning):
                        df = pd.read_csv(
                            filestream, skiprows=skip_rows[source]
                        )
                else:
                    df = pd.read_csv(filestream, skiprows=skip_rows[source])

        # Check if the data is equal to the DataFrame loaded from the zip file
        pd.testing.assert_frame_equal(data, df)

        # Check the shape
        if source == "nasa":
            assert data.shape == (36424, 288)
        else:
            assert data.shape == (7339, 98)

    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_store(self, index_cols: dict, source: str):
        """Test the load function with the in-memory cache features.

        - Test if nothing is saved. (store_index=False)
        - Test if the index is saved and the dataset is not. (Default)
        - Test if the index and dataset is saved. (store=True)
        - Test if the datasets is loaded from the cache if saved.
        """
        # Empty the dictionary of the indexes and datasets
        databases._IN_MEMORY_INDEXES[source] = None
        databases._IN_MEMORY_DATASETS[source] = pd.DataFrame()
        databases._IS_FULLY_STORED[source] = False

        # Ensure correct zip name
        databases._ZIP_FILENAME = "datasets.zip"

        # Index columns saved
        saved_index_cols = index_cols[source]

        # -------------------------------------------------------------------
        # Nothing is saved. (store_index=False)
        # -------------------------------------------------------------------

        # Load the dataset without saving
        data = databases.load(source=source, store_index=False)

        # Check the dictionary of the indexes if empty
        assert databases._IN_MEMORY_INDEXES[source] is None
        assert databases._IN_MEMORY_DATASETS[source].empty
        assert not databases._IS_FULLY_STORED[source]

        # -------------------------------------------------------------------
        # The index is saved and the dataset is not. (Default)
        # -------------------------------------------------------------------

        # Load the dataset again
        data = databases.load(source=source)

        # Check the dictionary of the indexes if filled
        assert databases._IN_MEMORY_INDEXES[source] is not None
        pd.testing.assert_frame_equal(
            data[saved_index_cols], databases._IN_MEMORY_INDEXES[source]
        )

        # Check the dictionary of the datasets is empty
        assert databases._IN_MEMORY_DATASETS[source].empty
        assert not databases._IS_FULLY_STORED[source]

        # -------------------------------------------------------------------
        # The index and dataset is saved. (store=True)
        # -------------------------------------------------------------------

        # Load the dataset again
        data = databases.load(source=source, store=True)

        # Check the dictionary of the indexes is not empty
        assert databases._IN_MEMORY_INDEXES[source] is not None

        # Check the dictionary of the datasets if filled
        pd.testing.assert_frame_equal(
            data, databases._IN_MEMORY_DATASETS[source]
        )

        # Check the variable _IS_FULLY_STORED
        assert databases._IS_FULLY_STORED[source]

        # -------------------------------------------------------------------
        # The datasets is loaded from the cache if saved
        # -------------------------------------------------------------------

        # Temporarily change the path of the zip file
        databases._ZIP_FILENAME = "wrong.zip"

        # Load the dataset again
        data2 = databases.load(source=source)

        # Check the dictionary of the indexes if filled
        assert databases._IN_MEMORY_INDEXES[source] is not None

        # Check if the data is equal to loaded from the original zip file
        pd.testing.assert_frame_equal(data, data2)

    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_only_index(self, index_cols: dict, source: str):
        """Test the load function with the only_index parameter.

        - Test if only the indexes are loaded. (only_index=True)
        - Test if the indexes are loaded from the cache if saved.
        """
        # Empty the dictionary of the indexes and datasets
        databases._IN_MEMORY_INDEXES[source] = None
        databases._IN_MEMORY_DATASETS[source] = pd.DataFrame()
        databases._IS_FULLY_STORED[source] = False

        # Ensure correct zip name
        databases._ZIP_FILENAME = "datasets.zip"

        # -------------------------------------------------------------------
        # Only the indexes are loaded. (only_index=True)
        # -------------------------------------------------------------------

        # Load the dataset
        index1 = databases.load(source=source, only_index=True)

        # Check if the data is a pandas DataFrame
        assert isinstance(index1, pd.DataFrame)

        # Check the index columns
        assert index1.columns.tolist() == index_cols[source]

        # Check the shape
        if source == "nasa":
            assert index1.shape == (36424, 2)
        else:
            assert index1.shape == (7339, 2)

        # -------------------------------------------------------------------
        # The indexes are loaded from the cache if saved
        # -------------------------------------------------------------------

        # Temporarily change the path of the zip file
        databases._ZIP_FILENAME = "wrong.zip"

        # Load the dataset again
        index2 = databases.load(source=source, only_index=True)

        # Check if the index dictionary is not empty
        assert databases._IN_MEMORY_INDEXES[source] is not None

        # Check if the data is equal to loaded from the original zip file
        pd.testing.assert_frame_equal(index1, index2)

    # @pytest.mark.parametrize("random_seed", random_int(2))
    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_only_rows(self, random_int_gen: int, source: str):
        """Test the load function with the only_rows parameter.

        - Test ValueError if True. (only_rows=True)
        - Test ValueError if used with only_index=True.
        - Test if only the rows are loaded. (only_rows=<int>)
        - Test if the rows are loaded from the cache if saved.
        - Test empty df if the number of rows is greater than the dataset.
        """
        # Empty the dictionary of the indexes and datasets
        databases._IN_MEMORY_INDEXES[source] = None
        databases._IN_MEMORY_DATASETS[source] = pd.DataFrame()
        databases._IS_FULLY_STORED[source] = False

        # Ensure correct zip name
        databases._ZIP_FILENAME = "datasets.zip"

        # Get the dataset
        data = databases.load(source=source, store_index=False, store=False)

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
            databases.load(source=source, only_rows=True)

        # -------------------------------------------------------------------
        # ValueError if used with only_index=True.
        # -------------------------------------------------------------------

        # Load the dataset
        with pytest.raises(ValueError):
            databases.load(source=source, only_index=True, only_rows=good_row)

        # -------------------------------------------------------------------
        # Only the rows are loaded. (only_rows=<int>)
        # -------------------------------------------------------------------

        # Load the dataset
        data1 = databases.load(source=source, only_rows=good_row)

        # Check if the data is a pandas DataFrame
        assert isinstance(data1, pd.DataFrame)

        # Check the shape
        if source == "nasa":
            assert data1.shape == (1, 288)
        else:
            assert data1.shape == (1, 98)

        # Check the values
        pd.testing.assert_series_equal(data.loc[good_row], data1.squeeze())

        # -------------------------------------------------------------------
        # The rows are loaded from the cache if saved.
        # -------------------------------------------------------------------

        # Check the dictionary of the indexes are empty
        assert databases._IN_MEMORY_INDEXES[source] is None
        assert databases._IN_MEMORY_DATASETS[source].empty
        assert not databases._IS_FULLY_STORED[source]

        # Load the dataset and store a row
        row1 = databases.load(source=source, only_rows=good_row, store=True)

        # Check the stored dataset is not empty
        assert not databases._IN_MEMORY_DATASETS[source].empty
        assert not databases._IS_FULLY_STORED[source]

        # Check if properly stored
        pd.testing.assert_frame_equal(
            row1, databases._IN_MEMORY_DATASETS[source]
        )

        # Temporarily change the path of the zip file
        databases._ZIP_FILENAME = "wrong.zip"

        # Load with only_rows
        row2 = databases.load(source=source, only_rows=good_row)

        # Check if it is the same as before
        pd.testing.assert_frame_equal(row1, row2)

        # -------------------------------------------------------------------
        # Empty df if the number of rows is greater than the dataset.
        # -------------------------------------------------------------------

        # Force empty the dictionaries of the datasets and indexes
        databases._IN_MEMORY_DATASETS[source] = pd.DataFrame()
        databases._IN_MEMORY_INDEXES[source] = None
        databases._IS_FULLY_STORED[source] = False

        # Ensure correct zip name
        databases._ZIP_FILENAME = "datasets.zip"

        # Load the dataset
        with pytest.warns(UserWarning):
            data3 = databases.load(
                source=source, only_rows=bad_row, store=False
            )

        # Check if the data is empty
        # assert data3.empty
        assert data3.empty
        assert data3.shape == (0, data.shape[1])

    # @pytest.mark.parametrize("source", ["eu", "nasa"])
    # def test_load_extract(self, source: str):
    #     """Test the load function with the extract parameter.

    #     - Test if the data is extracted. (extract=True)
    #     - Test if the data is not extracted when the file already exists.
    #     (extract=True, overwrite=False)
    #     - Test if the data is extracted when the file already exists.
    #     (extract=True, overwrite=True)
    #     - Test if the data is read from the file when the file already exists.
    #     """
    #     # Load the dataset
    #     data = databases.load(source=source)

    #     # Get the file path
    #     path = databases.BASE_PATH / databases._DATASET_FILENAMES[source]

    #     # Get the zip path
    #     zip_path = databases.BASE_PATH / databases._ZIP_FILENAME

    #     # Assert the file does not exist
    #     assert not path.exists()

    #     # -------------------------------------------------------------------
    #     # The data is extracted. (extract=True)
    #     # -------------------------------------------------------------------

    #     # Load the dataset and extract
    #     data1 = databases.load(source=source, extract=True)

    #     # Assert the file exists
    #     assert path.exists()

    #     # Check if the data is equal to the original
    #     pd.testing.assert_frame_equal(data, data1)

    #     # Check if the extracted file is the same as the zip file
    #     # Only check the first 3 rows
    #     with zipfile.ZipFile(zip_path, "r") as zip_ref:
    #         this_source = "exoplanet_" + source if source == "eu" else source
    #         with zip_ref.open(this_source + ".csv") as filestream:
    #             with open(path, "r") as file:
    #                 assert filecmp.cmp(filestream, file)

    #     # -------------------------------------------------------------------
    #     # The data is not extracted when the file already exists.
    #     # (extract=True, overwrite=False)
    #     # -------------------------------------------------------------------

    #     # Assert cant extract if the file already exists
    #     with pytest.raises(FileExistsError):
    #         databases.load(source=source, extract=True)

    #     # -------------------------------------------------------------------
    #     # The data is extracted when the file already exists.
    #     # (extract=True, overwrite=True)
    #     # -------------------------------------------------------------------

    #     # Wait 1 second and get the file age
    #     age2 = path.stat().st_mtime

    #     # Load the dataset and extract
    #     data3 = databases.load(
    #         source=source, extract=True, overwrite=True
    #     )

    #     # Get the new file age
    #     age3 = path.stat().st_mtime

    #     # Assert the file is updated
    #     assert age3 < age2

    #     # Check if the data is equal to the original
    #     pd.testing.assert_frame_equal(data, data3)

    #     # -------------------------------------------------------------------
    #     # The data is read from the file when the file already exists.
    #     # -------------------------------------------------------------------

    #     # Temporarily change the path of the zip file
    #     databases._ZIP_FILENAME = "wrong.zip"

    #     # Load the dataset
    #     data4 = databases.load(source=source)

    #     # Check if the data is equal to the original
    #     pd.testing.assert_frame_equal(data, data4)

    #     # Remove the file
    #     path.unlink()


class TestCheckFileAge:
    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_check_file_age(self, source: str):
        """Test the check_file_age function."""
        # Check the file age
        age = databases.check_file_age(source=source)

        # Check if the age is a float
        assert isinstance(age, int)

        # Check if the age is greater than 0
        assert age > 0

        # Try to check the age of a non-existing file
        with pytest.warns(UserWarning):
            databases.check_file_age(source=source, from_zip=False)


class TestDownloadDataset:
    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_download(self, source: str, has_requests: bool):
        """Test the download function."""
        # Check if not request, then nothing
        if not has_requests:
            with pytest.raises(ImportError):
                out = databases.download(source=source)

        else:
            # Load the dataset
            data = databases.load(source=source, store_index=False)

            # Get destiny path
            path = databases.BASE_PATH / databases._DATASET_FILENAMES[source]

            # CHECK IF THE FILE DOES NOT EXISTS
            assert not path.exists()

            # Cant download the dataset if the file already exists in zip
            with pytest.warns(UserWarning):
                out = databases.download(source=source)

            assert out is None

            # Temporary change the zip path
            databases._ZIP_FILENAME = "wrong.zip"

            # Download the dataset
            data2 = databases.download(source=source, return_data=True)

            # Check if the file is saved
            assert path.exists()

            # Can't download the dataset if the file already exists
            with pytest.raises(FileExistsError):
                out = databases.download(source=source)

            # Load the dataset (from the new file because zip is wrong)
            data2 = databases.load(source=source, store_index=False)

            # Check if the data is equal or longer than the original
            assert data2.shape[0] >= data.shape[0]
            assert data2.shape[1] == data.shape[1]
            assert data2.columns.tolist() == data.columns.tolist()

            # Remove the file
            path.unlink()

            # Correct the zip path
            databases._ZIP_FILENAME = "datasets.zip"

            # Download the dataset with overwrite
            data3 = databases.download(source=source, overwrite=True)

            # Check if the file is saved
            assert path.exists()

            # Check if the data is equal to the original
            pd.testing.assert_frame_equal(data2, data3)
