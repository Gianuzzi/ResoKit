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
        zip_path = databases.BASE_PATH / databases.ZIP_FILENAME
        # Check if the zip file exists
        assert zipfile.is_zipfile(zip_path)
        # Check if the zip file is not empty
        assert zip_path.stat().st_size > 0
        # Check the zipfile has both ("eu.csv", "nasa.csv") files
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            assert "exoplanet_eu.csv" in zip_ref.namelist()
            assert "nasa.csv" in zip_ref.namelist()


class TestDatabases:
    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_dataset_naive(
        self, zip_path: str, skip_rows: dict, source: str
    ):
        """Test the load_dataset function with the naive approach."""
        # Load the dataset
        data = databases.load_dataset(source=source)

        # Check if the data is a pandas DataFrame
        assert isinstance(data, pd.DataFrame)

        # Load the dataset from the zip file with pandas
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            this_source = "exoplanet_" + source if source == "eu" else source
            with zip_ref.open(this_source + ".csv") as filestream:
                df = pd.read_csv(filestream, skiprows=skip_rows[source])

        # Check if the data is equal to the DataFrame loaded from the zip file
        pd.testing.assert_frame_equal(data, df)

        # Check the shape
        if source == "nasa":
            assert data.shape == (36424, 288)
        else:
            assert data.shape == (7339, 98)

    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_dataset_store(self, index_cols: dict, source: str):
        """Test the load_dataset function with the in-memory cache features.

        - Test if nothing is saved. (store_index=False)
        - Test if the index is saved and the dataset is not. (Default)
        - Test if the index and dataset is saved. (store=True)
        - Test if the datasets is loaded from the cache if saved.
        """
        # Empty the dictionary of the indexes and datasets
        databases.IN_MEMORY_INDEXES[source] = None
        databases.IN_MEMORY_DATASETS[source] = None

        # Ensure correct zip name
        databases.ZIP_FILENAME = "datasets.zip"

        # Index columns saved
        saved_index_cols = index_cols[source]

        # -------------------------------------------------------------------
        # Nothing is saved. (store_index=False)
        # -------------------------------------------------------------------

        # Load the dataset without saving
        data = databases.load_dataset(source=source, store_index=False)

        # Check the dictionary of the indexes if empty
        assert databases.IN_MEMORY_INDEXES[source] is None
        assert databases.IN_MEMORY_DATASETS[source] is None

        # -------------------------------------------------------------------
        # The index is saved and the dataset is not. (Default)
        # -------------------------------------------------------------------

        # Load the dataset again
        data = databases.load_dataset(source=source)

        # Check the dictionary of the indexes if filled
        assert databases.IN_MEMORY_INDEXES[source] is not None
        pd.testing.assert_frame_equal(
            data[saved_index_cols], databases.IN_MEMORY_INDEXES[source]
        )

        # Check the dictionary of the datasets is empty
        assert databases.IN_MEMORY_DATASETS[source] is None

        # -------------------------------------------------------------------
        # The index and dataset is saved. (store=True)
        # -------------------------------------------------------------------

        # Load the dataset again
        data = databases.load_dataset(source=source, store=True)

        # Check the dictionary of the indexes is not empty
        assert databases.IN_MEMORY_INDEXES[source] is not None

        # Check the dictionary of the datasets if filled
        pd.testing.assert_frame_equal(
            data, databases.IN_MEMORY_DATASETS[source]
        )

        # -------------------------------------------------------------------
        # The datasets is loaded from the cache if saved
        # -------------------------------------------------------------------

        # Temporarily change the path of the zip file
        databases.ZIP_FILENAME = "wrong.zip"

        # Load the dataset again
        data2 = databases.load_dataset(source=source)

        # Check the dictionary of the indexes if filled
        assert databases.IN_MEMORY_INDEXES[source] is not None

        # Check if the data is equal to loaded from the original zip file
        pd.testing.assert_frame_equal(data, data2)

    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_dataset_only_index(self, index_cols: dict, source: str):
        """Test the load_dataset function with the only_index parameter.

        - Test if only the indexes are loaded. (only_index=True)
        - Test if the indexes are loaded from the cache if saved.
        """
        # Empty the dictionary of the indexes
        databases.IN_MEMORY_INDEXES[source] = None

        # Ensure correct zip name
        databases.ZIP_FILENAME = "datasets.zip"

        # -------------------------------------------------------------------
        # Only the indexes are loaded. (only_index=True)
        # -------------------------------------------------------------------

        # Load the dataset
        index1 = databases.load_dataset(source=source, only_index=True)

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
        databases.ZIP_FILENAME = "wrong.zip"

        # Load the dataset again
        index2 = databases.load_dataset(source=source, only_index=True)

        # Check if the index dictionary is not empty
        assert databases.IN_MEMORY_INDEXES[source] is not None

        # Check if the data is equal to loaded from the original zip file
        pd.testing.assert_frame_equal(index1, index2)

    # @pytest.mark.parametrize("random_seed", random_int(2))
    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_load_dataset_only_rows(self, random_int_gen: int, source: str):
        """Test the load_dataset function with the only_rows parameter.

        - Test ValueError if True. (only_rows=True)
        - Test ValueError if used with only_index=True.
        - Test if only the rows are loaded. (only_rows=<int>)
        -- Test if nothing is cached if only rows are loaded.
        - Test if only the rows are loaded. (only_rows=<list>)
        - Test if the rows are loaded from the cache if saved.
        - Test empty df if the number of rows is greater than the dataset.
        """
        # Get the dataset
        data = databases.load_dataset(
            source=source, store_index=False, store=False
        )

        # Force empty the dictionaries of the datasets and indexes
        databases.IN_MEMORY_DATASETS[source] = None
        databases.IN_MEMORY_INDEXES[source] = None

        # Ensure correct zip name
        databases.ZIP_FILENAME = "datasets.zip"

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
            databases.load_dataset(source=source, only_rows=True)

        # -------------------------------------------------------------------
        # ValueError if used with only_index=True.
        # -------------------------------------------------------------------

        # Load the dataset
        with pytest.raises(ValueError):
            databases.load_dataset(
                source=source, only_index=True, only_rows=good_row
            )

        # -------------------------------------------------------------------
        # Only the rows are loaded. (only_rows=<int>)
        # -------------------------------------------------------------------

        # Load the dataset
        data1 = databases.load_dataset(source=source, only_rows=good_row)

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
        # - Nothing is cached if only rows are loaded.
        # -------------------------------------------------------------------

        # Check the dictionary of the indexes are empty
        assert databases.IN_MEMORY_INDEXES[source] is None
        assert databases.IN_MEMORY_DATASETS[source] is None

        # -------------------------------------------------------------------
        # The rows are loaded from the cache if saved.
        # -------------------------------------------------------------------

        # Load and store the whole dataset
        databases.load_dataset(source=source, store=True)

        # Temporarily change the path of the zip file
        databases.ZIP_FILENAME = "wrong.zip"

        # Load with only_rows
        data2 = databases.load_dataset(source=source, only_rows=good_row)

        # Check if it is the same as before
        pd.testing.assert_frame_equal(data1, data2)

        # -------------------------------------------------------------------
        # Empty df if the number of rows is greater than the dataset.
        # -------------------------------------------------------------------

        # Force empty the dictionaries of the datasets and indexes
        databases.IN_MEMORY_DATASETS[source] = None
        databases.IN_MEMORY_INDEXES[source] = None

        # Ensure correct zip name
        databases.ZIP_FILENAME = "datasets.zip"

        # Load the dataset
        data3 = databases.load_dataset(
            source=source, only_rows=bad_row, store=False
        )

        # Check if the data is empty
        # assert data3.empty
        assert data3.shape == (0, data.shape[1])
