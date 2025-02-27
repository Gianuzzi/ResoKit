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
    @pytest.mark.parametrize(
        "source", ["eu", "nasa", "both", "p", "s", "binary", "all", "wrong"]
    )
    def test_clear_memory(self, source: str, mock_in_mem):
        """Test the clear_memory function."""
        # Mock the variables
        mock_in_mem(which="all")

        # Pre-check
        for eunasa in ["eu", "nasa"]:
            pd.testing.assert_series_equal(
                databases._IN_MEMORY_INDEXES[eunasa], default_ss
            )
            pd.testing.assert_frame_equal(
                databases._IN_MEMORY_DATASETS[eunasa], default_df
            )
            assert databases._IS_FULLY_STORED[eunasa]
        for ps in ["p", "s"]:
            pd.testing.assert_frame_equal(
                databases._IN_MEMORY_BINARIES[ps], default_df
            )
            assert databases._IN_MEMORY_BINARIES_HEADERS[ps] == "Header_" + ps

        # Setup
        if source == "all":
            eunasa = ["eu", "nasa"]
            other_en = []
            binary = ["p", "s"]
            other_bin = []
        elif source in ["eu", "nasa"]:
            eunasa = [source]
            other_en = ["nasa"] if source == "eu" else ["eu"]
            binary = []
            other_bin = ["p", "s"]
        elif source in ["p", "s"]:
            binary = [source]
            other_bin = ["s"] if source == "p" else ["p"]
            eunasa = []
            other_en = ["eu", "nasa"]
        elif source == "both":
            eunasa = ["eu", "nasa"]
            other_en = []
            binary = []
            other_bin = ["p", "s"]
        elif source == "binary":
            binary = ["p", "s"]
            other_bin = []
            eunasa = []
            other_en = ["eu", "nasa"]
        else:
            with pytest.raises(ValueError):
                databases.clear_memory(source=source, verbose=False)
            return

        # Clear the memory
        databases.clear_memory(source=source, verbose=False)

        # Check the results
        for en in eunasa:
            assert isinstance(
                databases._IN_MEMORY_INDEXES[en], databases.ResoKitDataset
            )
            assert isinstance(
                databases._IN_MEMORY_DATASETS[en], databases.ResoKitDataset
            )
            assert databases._IN_MEMORY_INDEXES[en].empty
            assert databases._IN_MEMORY_DATASETS[en].empty
            assert not databases._IS_FULLY_STORED[en]
        for oen in other_en:
            pd.testing.assert_series_equal(
                databases._IN_MEMORY_INDEXES[oen], default_ss
            )
            pd.testing.assert_frame_equal(
                databases._IN_MEMORY_DATASETS[oen], default_df
            )
            assert databases._IS_FULLY_STORED[oen]
        for bina in binary:
            assert databases._IN_MEMORY_BINARIES[bina].empty
            assert databases._IN_MEMORY_BINARIES_HEADERS[bina] == ""
        for obin in other_bin:
            pd.testing.assert_frame_equal(
                databases._IN_MEMORY_BINARIES[obin], default_df
            )
            assert (
                databases._IN_MEMORY_BINARIES_HEADERS[obin] == "Header_" + obin
            )

        return


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
    def test_load_full_basic(self, source: str):
        """Test the load_full function."""
        # Clear the memory
        databases.clear_memory(source=source, verbose=False)

        # Load the dataset
        if source == "nasa":
            data = databases.load_nasa(verbose=False)
        else:
            data = databases.load_eu(verbose=False)

        # Load the dataset
        data2 = databases.load_full(
            source=source,
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

        # Check all is empty
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

        # Check the dictionary of the datasets partially filled
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
    @pytest.mark.parametrize("to_resokit", [True, False])
    @pytest.mark.parametrize("to_df", [True, False])
    @pytest.mark.parametrize("check_age", [True, False])
    def test_load_full_full(
        self,
        source: str,
        to_resokit: bool,
        to_df: bool,
        check_age: bool,
        capfd,
    ):
        """Test the load_full function with all the parameters."""
        # Load the dataset
        data = databases.load_full(
            source=source,
            to_resokit=to_resokit,
            to_df=to_df,
            check_age=check_age,
            verbose=False,
            store=True,  # We can use store=True because we are testing
        )

        out, _ = capfd.readouterr()

        if to_df:
            # Check if the data is a DataFrame
            assert isinstance(data, pd.DataFrame)

            # Check it is not a ResokitDataset
            assert not isinstance(data, databases.ResoKitDataset)

            # Check if the data is not empty
            assert not data.empty
        else:
            # Check if a ResokitDataset
            assert isinstance(data, databases.ResoKitDataset)

            # Check if the data is not empty
            assert not data.empty

        # Check if check the age
        if check_age:
            if not to_df:  # If to_df, the age is not checked
                assert data.age > 0
            assert "Last modified: " in out
            assert "days ago" in out

        # Check to resokit
        if to_resokit:
            assert "star_radius_err_max" in data.columns
            assert "mass_err_min" in data.columns
        else:
            assert "star_radius_err_max" not in data.columns
            assert "mass_err_min" not in data.columns
            if source == "nasa":
                assert "st_raderr2" in data.columns
                assert "pl_massjerr1" in data.columns
            else:
                assert "star_radius_error_max" in data.columns
                assert "mass_error_min" in data.columns

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

        len_old = databases._IN_MEMORY_DATASETS[source].shape[0]

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

        # Check the length
        assert databases._IN_MEMORY_DATASETS[source].shape[0] == len_old

        # -------------------------------------------------------------------
        # Empty df if the number of rows is greater than the dataset.
        # -------------------------------------------------------------------

        # Clear the memory
        databases.clear_memory(source=source, verbose=False)

        # Reset the zip file
        mocker.patch("resokit.datasets.utils.ZIP_FILENAME", "datasets.zip")

        # Load the dataset
        with pytest.raises(ValueError), pytest.warns(UserWarning):
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

        # Load a dataset with good and bad rows
        with pytest.warns(UserWarning):
            data4 = databases.load_full(
                source=source,
                only_rows=[good_row] + [bad_row],
                store=False,
                verbose=False,
            )

        # Check if the data has the good rows
        pd.testing.assert_frame_equal(data1.dataset, data4.dataset)


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


class TestDatasetClass:
    dataset_eu = databases.load_eu(verbose=False)
    dataset_nasa = databases.load_nasa(verbose=False)
    dataset_dict = {"eu": dataset_eu, "nasa": dataset_nasa}

    def test_dataset_eq(self):
        """Test the __eq__ method of the ResoKitDataset class."""
        assert self.dataset_eu == self.dataset_eu
        assert self.dataset_nasa == self.dataset_nasa
        assert self.dataset_eu != self.dataset_nasa

        # Check with datasets and dfs
        assert self.dataset_eu == self.dataset_eu.dataset
        assert self.dataset_nasa == self.dataset_nasa.dataset
        assert self.dataset_eu != self.dataset_nasa

    @pytest.mark.parametrize("source", ["eu", "nasa"])
    def test_dataset_copy(self, source):
        """Test the copy method of the ResoKitDataset class."""
        data = self.dataset_dict[source]
        data2 = data.copy()

        # Check if the data is equal, but different id
        assert data == data2
        assert id(data) != id(data2)

    def test_to_dict(self):
        """Test the to_dict method of the ResoKitDataset class."""
        data_dict = self.dataset_eu.to_dict()

        # Assert is Metadata
        assert isinstance(data_dict, dict)

        # Assert the content
        assert data_dict["author_email"] == "egianuzzi@unc.edu.ar"
        assert data_dict["author"] == "Emmanuel Gianuzzi"

    @pytest.mark.parametrize(
        "item", ["mass", "radius", ["mass", "radius"], "bad_item"]
    )
    def test_getitem(self, item):
        """Test the __getitem__ method of the ResoKitDataset class."""
        if item == "bad_item":
            with pytest.raises(KeyError):
                data = self.dataset_eu[item]
            return

        data = self.dataset_eu[item]

        # Check if the data is a Series
        assert isinstance(data, databases.ResoKitDataset)

        assert len(data.dataset) == len(self.dataset_eu.dataset)

        # Check the length
        if isinstance(item, list):
            assert data.shape[1] == 2
            assert data.columns.tolist() == item
        else:
            assert data.shape[1] == 1
            assert data.columns[0] == item

    def test_dataset_eq_single_value(self):
        """Test the __eq__ method with a single value."""
        # get the known data
        disc_method = self.dataset_eu["disc_method"]
        transit = disc_method == "Primary Transit"
        amount = transit.sum().values[0]

        assert amount == 4507  # Known value


class TestDownloadDataset:
    def retrieve_data(self, source: str, zip_path: str):
        """Retrieve the data from the zip file."""
        if source == "eu":
            filename = "exoplanet_eu.csv"
        elif source == "p":
            filename = "plan_circ.txt"
        else:
            raise ValueError(f"source={source} not recognized.")

        with zipfile.ZipFile(zip_path, "r") as z:
            with z.open(f"{filename}", "r") as f:
                lines = f.read()

        return lines

    def test_download_y_requests(
        self, db_temp_path: str, capfd, mocker, zip_path
    ):
        """Test the download function when requests is installed."""
        if not utils.requests_imported:
            pytest.skip("requests is not installed.")

        # JUST FOR SOURCE = "EU"
        source = "eu"

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

        # CHECK IF THE FILE DOES NOT EXISTS
        assert not path.exists()

        # Get the corresponding lines
        lines = self.retrieve_data(source, zip_path)

        # MOCK THE DOWNLOAD FUNCTION
        mocker.patch(
            "resokit.datasets.databases.request_dataset", return_value=lines
        )
        mocker.patch(
            "resokit.datasets.databases.check_outdated", return_value=True
        )

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

        # Assert data2 is equal to data
        pd.testing.assert_frame_equal(data.dataset, data2.dataset)

        # Can't download the dataset if the data already exists in memory
        # (overwrite=False) is the default.
        assert databases.download(source=source, verbose=True) is None
        out, _ = capfd.readouterr()

        assert "Dataset is already fully stored." in out
        assert "Set overwrite=True to force the download." in out

        # Can't download the dataset if the file already exists
        # (overwrite=False) is the default.
        with pytest.raises(FileExistsError):
            databases.download(source=source, verbose=False, to_file=path)

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

    def test_download_y_requests_bina(
        self, db_temp_path: str, capfd, mocker, zip_path
    ):
        """Test the download function when requests is installed."""
        if not utils.requests_imported:
            pytest.skip("requests is not installed.")

        # JUST FOR SOURCE = "p"
        source = "p"

        # Clear the memory
        databases.clear_memory(source=source, verbose=False)

        # If no destination, raise an error
        with pytest.raises(ValueError):
            databases.download_binary(
                circumbinary=True, to_memory=False, verbose=False
            )

        # Load the dataset
        data = databases.load_binary(
            circumbinary=True,
            ret_header=False,
            verbose=False,
            clean=False,
        )

        # Get a temp file path
        tmp_path = db_temp_path(source, "txt")

        # Set destiny path
        path = Path(tmp_path)

        # CHECK IF THE FILE DOES NOT EXISTS
        assert not path.exists()

        # MOCK THE DOWNLOAD FUNCTION
        # Get the corresponding lines
        lines = self.retrieve_data(source, zip_path)

        # MOCK THE DOWNLOAD FUNCTION
        mocker.patch(
            "resokit.datasets.databases.request_dataset", return_value=lines
        )

        # Download the dataset
        data2 = databases.download_binary(
            circumbinary=True, to_file=path, verbose=False, return_data=True
        )

        # Check if the file is saved
        assert path.exists()

        # Assert data2 is equal to data
        pd.testing.assert_frame_equal(data, data2)

        # Can't download the dataset if the data already exists in memory
        # (overwrite=False) is the default.
        with pytest.raises(ValueError):
            databases.download_binary(
                circumbinary=True, verbose=True, return_data=False
            )
            _, err = capfd.readouterr()
            assert "Nothing to do. Dataset is already stored in memory" in err

        # Can't download the dataset if the file already exists
        # (overwrite=False) is the default.
        with pytest.raises(FileExistsError):
            databases.download_binary(
                circumbinary=True, verbose=False, to_file=path
            )

        # Remove the file
        path.unlink()

        # Set destiny path
        file_path = Path(tmp_path)

        # Write some content
        file_path.write_text("Old content")

        with pytest.raises(FileExistsError):
            databases.download_binary(
                circumbinary=True,
                to_file=file_path,
                overwrite=False,
                verbose=False,
            )

        # remove file
        file_path.unlink()
