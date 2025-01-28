#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2024, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Module ResoKit."""

# =============================================================================
# IMPORTS
# =============================================================================

import warnings
from collections.abc import Mapping
from typing import Iterable, List, Tuple, Union

import attrs

import matplotlib.pyplot as plt

from numpy import isnan, pi, sqrt

import pandas as pd

from resokit.units import Mj2Me, Me2Mj, Re2Rj, Rj2Re
from resokit.utils.mass_radius import estimate_mass, estimate_radius
from resokit.utils.utils import (
    DEFAULT_METADATA,
    MAPPINGS,
    RESO_DTYPES,
    RESO_OB_TYPES,
    RESO_PL_TYPES,
    RESO_SR_TYPES,
    calc_a_with_errors,
    calc_period_with_errors,
    float_to_fraction,
    hill_radius_with_errors,
    parse_to_iter,
)

# =============================================================================
# BASE CLASSES
# =============================================================================


@attrs.define(frozen=True, repr=False)
class MetaData(Mapping):
    """Implements an inmutable dict-like to store the metadata.

    Also provides attribute like access to the keys.

    Example
    -------
    >>> metadata = MetaData({"a": 12, "b": 2})
    >>> metadata.a
    12

    >>> metadata["a"]
    12
    """

    _data = attrs.field(converter=dict, factory=dict)

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""
        return f"Metadata({repr(self._data)})"

    def __getitem__(self, k):
        """x[k] <=> x.__getitem__(k)."""
        return self._data[k]

    def __iter__(self):
        """iter(x) <=> x.__iter__()."""
        return iter(self._data)

    def __len__(self):
        """len(x) <=> x.__len__()."""
        return len(self._data)

    def __getattr__(self, a):
        """getattr(x, y) <==> x.__getattr__(y) <==> getattr(x, y)."""
        return self[a]


@attrs.define(frozen=True, slots=True, repr=False)
class ResokitDataFrame:
    """Initialize a ResoKit DataFrame class.

    Parameters
    ----------
    data_df : pd.DataFrame or pd.Series
        DataFrame containing the data.
    source : str
        Source of the dataset. Either 'eu' or 'nasa' or 'user'.
    metadata : dict
        Metadata of the dataset.
    """

    data_df: Union[pd.DataFrame, pd.Series] = attrs.field(
        validator=attrs.validators.instance_of((pd.DataFrame, pd.Series)),
        converter=lambda df: df.squeeze(),  # Convert to Series if possible
    )
    source: str = attrs.field(
        validator=attrs.validators.in_({"eu", "nasa", "user"}),
        converter=str.lower,  # Convert to lowercase
    )
    metadata: dict = attrs.field(factory=MetaData, converter=MetaData)

    columns_: list = attrs.field(init=False)
    n_columns_: int = attrs.field(init=False)
    n_objects_: int = attrs.field(init=False)

    @columns_.default
    def _columns__default(self) -> list:
        """Set the default value for columns_."""
        return (
            self.data_df.index
            if isinstance(self.data_df, pd.Series)
            else self.data_df.columns
        ).to_list()

    @n_columns_.default
    def _n_columns__default(self) -> int:
        """Set the default value for n_columns_."""
        return len(self.columns_)

    @n_objects_.default
    def _n_objects__default(self) -> int:
        """Set the default value for n_objects_."""
        return (
            self.data_df.shape[0]
            if isinstance(self.data_df, pd.DataFrame)
            else 1
        )

    def __attrs_post_init__(self):
        """Post-initialization hook."""
        if self.data_df.empty:
            warnings.warn("Empty DataFrame.", stacklevel=2)

        if "name" not in self.columns_:
            warnings.warn(
                "Missing 'name' column in the DataFrame.",
                stacklevel=2,
            )

        if self.n_objects_ == 1 and not isinstance(self.data_df, pd.Series):
            raise TypeError(
                "With only one object, data_df must be a Series "
                + "(not a DataFrame)"
            )

        return

    def __len__(self):
        """len(x) <=> x.__len__()."""
        return self.n_objects_

    def __getitem__(self, key):
        """x[y] <==> x.__getitem__(y)."""
        if self.n_objects_ == 1:
            if isinstance(key, int):
                return self.data_df.iloc[key]

            if isinstance(key, list):
                if all(isinstance(i, int) for i in key):
                    return self.data_df.iloc[key]

            return self.data_df[key]

        return self.data_df.__getitem__(key)

    def __dir__(self):
        """dir(pdf) <==> pdf.__dir__()."""
        return super().__dir__() + dir(self.data_df)

    def __getattr__(self, a):
        """getattr(x, y) <==> x.__getattr__(y) <==> getattr(x, y)."""
        return getattr(self.data_df, a)

    def __repr__(self, prefoot=None):
        """repr(x) <=> x.__repr__()."""
        with pd.option_context("display.show_dimensions", False):
            df_body = repr(self.data_df).splitlines()

        rows = f"{self.n_objects_} row{'s' if self.n_objects_ > 1 else ''}"
        columns = f"{self.n_columns_} columns"
        if prefoot is None:
            prefoot = f"\n{type(self).__name__}"
        footer = f"{prefoot} - {rows} x {columns}"

        resokit_data_repr = "\n".join(df_body + [footer])

        return resokit_data_repr

    def _repr_html_(self, ad_id=None, prefoot=None, switch=False):
        """Return a HTML representation of the DataFrame."""
        if ad_id is None:
            ad_id = id(self)

        if switch:
            r = f"{self.n_objects_} column{'s' if self.n_objects_ > 1 else ''}"
            c = f"{self.n_columns_} row{'s' if self.n_columns_ > 1 else ''}"
        else:
            r = f"{self.n_objects_} row{'s' if self.n_objects_ > 1 else ''}"
            c = f"{self.n_columns_} column{'s' if self.n_columns_ > 1 else ''}"
        if prefoot is None:
            prefoot = f"\n{type(self).__name__}"
        footer = f"{prefoot} - {r} x {c}"

        with pd.option_context("display.show_dimensions", False):
            if self.n_objects_ > 1:  # It is a DataFrame
                df_html = self.data_df._repr_html_()
            else:  # It is a Series
                df_html = self.data_df.to_frame()._repr_html_()

        parts = [
            f'<div class="resokit-data-container" id={ad_id}>',
            df_html,
            footer,
            "</div>",
        ]

        html = "".join(parts)

        return html

    def to_dataframe(self, columns=None, copy=False) -> pd.DataFrame:
        """Convert data to pandas data frame.

        This method constructs a data frame with the data inside the
        data_df attribute.

        Parameters
        ----------
        columns : list, optional. Default: None.
            Specific columns to return.
            If `None`, return all columns.
        copy : bool, optional. Default: False.
            Whether to return a copy of the `DataFrame`, or the original.

        Returns
        -------
        df: DataFrame
            Data frame with the requested columns.
        """
        if columns is None:
            # my_cols = RESO_DTYPES.keys()
            # # Add columns in this df, but not in the default mapping
            # my_cols = my_cols | [
            #     col for col in self.columns_ if col not in my_cols
            # ]
            used_cols = list(self.columns_)
        else:
            used_cols = [col for col in list(columns) if col in self.columns_]

        df = self.data_df[used_cols]

        return df.copy(deep=True) if copy else df

    def to_dict(self) -> dict:
        """Convert metadata to a dictionary.

        This method constructs a dictionary with the data inside the
        metadata attribute.

        Returns
        -------
        metadata : dict
            Dictionary with the metadata.
        """
        return dict(self.metadata)

    def plot(
        self,
        x: str,
        y: str,
        error_x: bool = False,
        error_y: bool = False,
        ax: plt.Axes = None,
        label: str = "",
        **plot_kwargs,
    ) -> plt.Axes:
        """Plot the x vs y data of the :py:class:`ResokitDataFrame`.

        Parameters
        ----------
        x : str
            Name of the column to use as x-axis.
        y : str
            Name of the column to use as y-axis.
        error_x : bool, optional. Default: False.
            Whether to plot the x error bars.
        error_y : bool, optional. Default: False.
            Whether to plot the y error bars.
        ax : plt.Axes, optional. Default: None.
            `Matplotlib Axes` to plot on.
            If `None`, get and use the current `Axes`.
        label : str, optional. Default: "".
            Label for the data plotted.
        plot_kwargs : dict
            Additional keyword arguments for the :py:func:`plt.errorbar`
            function.

        Returns
        -------
        ax : Matplotlib Axes
            Matplotlib axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        x_data = self[x]
        y_data = self[y]

        if not isinstance(x_data, str) and isnan(x_data):
            return ax

        if not isinstance(y_data, str) and isnan(y_data):
            return ax

        if not isinstance(error_x, bool) or not isinstance(error_y, bool):
            raise TypeError("error_x and error_y must be booleans.")

        # Check error columns
        if error_x:
            try:
                xerr_min = self[f"{x}_err_min"]
                xerr_max = self[f"{x}_err_max"]
            except KeyError:
                error_x = False

        if error_y:
            try:
                yerr_min = self[f"{y}_err_min"]
                yerr_max = self[f"{y}_err_max"]
            except KeyError:
                error_y = False

        # Check label
        if label:
            label = str(label)
        else:
            label = None

        # Check fmt
        fmt = plot_kwargs.pop("fmt", "o")

        # Plot the data
        ax.errorbar(
            x_data,
            y_data,
            xerr=[[xerr_min], [xerr_max]] if error_x else None,
            yerr=[[yerr_min], [yerr_max]] if error_y else None,
            label=label,
            fmt=fmt,
            **plot_kwargs,
        )

        return ax

    def copy(self) -> "ResokitDataFrame":
        """Create and return copy of the :py:class:`ResokitDataFrame`.

        Returns
        -------
        ResokitDataFrame
            Copy of the ResokitDataFrame.
        """
        return ResokitDataFrame(
            data_df=self.data_df.copy(),
            source=self.source,
            metadata=self.metadata,
        )


# =============================================================================
# BASE FUNCTIONS
# =============================================================================


def df_to_resokit(
    df: pd.DataFrame,
    source: str,
    drop: bool = True,
    copy: bool = False,
    sort_by: Union[str, bool] = "P",
    return_df: bool = False,
    rename_index: bool = True,
    metadata: dict = None,
) -> ResokitDataFrame:
    """Convert ExoplanetEU or NASA data to :py:class:`ResokitDataFrame`.

    This function converts a DataFrame from ExoplanetEU or NASA to a
    :py:class:`ResokitDataFrame`. The columns are renamed according to the
    default mapping, and the DataFrame is sorted by the specified column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    source : str
        Source of the dataset. Either 'eu' or 'nasa'.
    drop : bool, optional. Default: True.
        Whether to drop columns not in the mapping.
    copy : bool, optional. Default: False.
        Whether to edit a copy of the DataFrame, instead of the original.
        Despite this, the output will be a :py:class:`ResokitDataFrame`,
        unless `return_df=True`.
    sort_by : str, bool, optional. Default: "P".
        Column to sort the data by.
        If `False` or `None`, do not sort the data.
        If `True`, sort by period ("P").
    return_df : bool, optional. Default: False.
        Whether to return the a pandas Data frame instead of the
        :py:class:`ResokitDataFrame`.
    rename_index : bool, optional. Default: True.
        Whether to rename the index column to "name" of the object/body.
    metadata : dict, optional. Default: None.
        Metadata to be added to the :py:class:`ResokitDataFrame`.

    Returns
    -------
    ResokitDataFrame
        DataFrame in :py:class:`ResokitDataFrame` format.
    """
    # Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a DataFrame. Got: {type(df)} instead.")

    # Get the new columns dictionary
    new_cols_dict = MAPPINGS[source]

    # Copy the DataFrame
    if copy:
        df = df.copy()

    # Rename columns
    df = df.rename(columns=new_cols_dict)

    # Drop columns not in the mapping
    if drop:
        df = df.drop(columns=set(df.columns) - set(new_cols_dict.values()))

    # Assert no empty DataFrame
    if df.empty:
        raise ValueError("Cannot create an empty ResokitDataFrame")

    # Add "n" column if not present
    if "P" in df.columns:
        if "P_err_min" in df.columns and "P_err_max" in df.columns:
            df["n_err_min"] = 2.0 * pi / df["P_err_max"]
            df["n_err_max"] = 2.0 * pi / df["P_err_min"]
        df["n"] = 2.0 * pi / df["P"]

    # Define all errors positive
    for col in df.columns:
        if col.endswith("_err_min") or col.endswith("_err_max"):
            df[col] = df[col].abs()

    # Sort by
    if sort_by and sort_by is not None:
        if sort_by is True:
            sort_by = "P"
        df = df.sort_values(by=sort_by, ascending=True)

    # Rename index if needed
    if rename_index and "name" in df.columns:
        df.reset_index(drop=True, inplace=True)
        df.set_index("name", inplace=True, drop=False)

    # Return DataFrame if needed
    if return_df:
        return df

    # Add metadata
    if metadata is None:
        metadata = dict(DEFAULT_METADATA)

    return ResokitDataFrame(data_df=df, source=source, metadata=metadata)


# =============================================================================
# STATIC CLASSES
# =============================================================================


@attrs.define(repr=False, frozen=True, slots=True)
class StaticPlanet(ResokitDataFrame):
    """StaticPlanet class representing a static planet.

    Attributes
    ----------
    data_df : pd.Series
        Pandas Series containing the data.
    source : str
        Source of the dataset.
        Either 'eu' or 'nasa' or 'user'.
    metadata : dict
        Metadata of the dataset.
    name : str
        Name of the planet.
    user_defined_ : bool
        Flag indicating if the planet is user-defined.
    suffix_ : str
        Suffix for the planet name.
    web_page : str
        Web page of the planet.
    """

    name: str = attrs.field(init=False)
    user_defined_: bool = attrs.field(init=False)
    suffix_: str = attrs.field(init=False)
    web_page: str = attrs.field(init=False)

    @name.default
    def _name_default(self):
        """Set the default value for name."""
        return self.data_df["name"]  # ["name"] because .name is a df method

    @user_defined_.default
    def _user_defined__default(self):
        """Set the default value for user_defined_."""
        return self.source not in ["eu", "nasa"]

    @suffix_.default
    def _suffix__default(self):
        """Set the default value for suffix_."""
        aux = self.data_df["name"].split(" ")[-1]
        if len(aux) == 1:
            return aux
        return aux[-1]

    @web_page.default
    def _web_page_default(self):
        """Set the default value for web_page."""
        if self.source == "eu":
            aux = (
                str(self.name).replace(" ", "_").lower()
                + "--"
                + str(self.metadata["eu_indexes"])
            )
            return "https://exoplanet.eu/catalog/" + aux + "/"
        if self.source == "nasa":
            aux = str(self.name).replace(" ", "%20")
            return (
                "https://exoplanetarchive.ipac.caltech.edu/overview/"
                + aux
                + "/"
            )
        return ""

    def __attrs_post_init__(self):
        """Post-initialization hook."""
        # Assert data_series is a Series and not DataFrame
        if not isinstance(self.data_df, pd.Series):
            raise TypeError(
                "StaticPlanet must have a pd.Series. "
                + f"Got: {type(self.data_df)} instead."
            )

        # Check if all columns are in the default mapping
        if not self.user_defined_:
            for col in self.data_df.index:
                if col not in RESO_PL_TYPES.keys() | RESO_OB_TYPES.keys() | {
                    "star_name",
                    "n",
                    "n_err_min",
                    "n_err_max",
                }:
                    warnings.warn(
                        "Found columns not in the default planet mapping.",
                        stacklevel=2,
                    )

    def __getitem__(self, key: Union[int, str, list]):
        """x[y] <==> x.__getitem__(y)."""
        key = parse_to_iter(key)

        if all(isinstance(i, int) for i in key):
            raise IndexError(
                "StaticPlanet does not support integer indexing. "
                + "Use the 'name' column instead."
            )
        elif any(isinstance(i, int) for i in key):
            raise NotImplementedError(
                "Mixed integer and string indexing not supported."
            )

        if len(key) == 1:
            return self.data_df[key[0]]

        return self.data_df[key]

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""
        text = f"StaticPlanet [{self.name}]"
        if not self.user_defined_:
            text += f" from {self.source} data source"
        else:
            text += " user defined"

        return text

    def _repr_html_(self):
        """Return a HTML representation of the StaticPlanet."""
        ad_id = id(self)
        prefoot = f"StaticPlanet [{self.name}]"
        if not self.user_defined_:
            prefoot += f" from {self.source} data source"
        else:
            prefoot += " user defined"

        return super()._repr_html_(ad_id=ad_id, prefoot=prefoot, switch=True)

    def get_item(
        self,
        items: Union[List[str], str],
        error: bool = False,
        silent: bool = False,
    ) -> pd.Series:
        """Return the specified items of the planet.

        Parameters
        ----------
        items : list, str
            Items to return.
        error : bool, optional. Default: False.
            Whether to return the error columns.
        silent : bool, optional. Default: False.
            Whether to suppress warnings for missing columns.

        Returns
        -------
        Series : pandas Series
            Series with the requested items.
        """
        items = parse_to_iter(items)

        vals = {}
        for item in items:
            vals[item] = self[item]
            if error:
                try:
                    vals[f"{item}_err_min"] = self[f"{item}_err_min"]
                    vals[f"{item}_err_max"] = self[f"{item}_err_max"]
                except KeyError:
                    if not silent:
                        warnings.warn(
                            f"Error columns not found for {item}.",
                            stacklevel=2,
                        )
                    pass

        return pd.Series(vals)

    def estimate_mass(
        self,
        **kwargs,
    ) -> float:
        r"""Calculate the mass of the planet using a power-law approximation.

        Equation:
            :math:`mass = \\frac{1}{C} \\times radius^{1/S}`

        Parameters
        ----------
        kwargs : dict
            Keyword arguments for the
            :py:func:`resokit.utils.mass_radius.estimate_mass_single`
            function.

        Returns
        -------
        mass, mass_err_min, mass_err_max : tuple[float, float, float]
            Estimated mass, and its minimum and maximum errors,
            in Jupiter masses.
            If `err_method=0`, the errors are 0.0.
            If `err_method=-1` (default), the errors are not returned.
        """
        # Get planet radius and convert to Earth radii
        radius = self["radius"] * Rj2Re

        radius_err_min = 0.0
        radius_err_max = 0.0
        # Get the errors and convert to Earth radii, if needed (and available)
        ret_err = True
        if kwargs.get("err_method", -1) in [1, 2]:
            radius_err_min = self["radius_err_min"] * Rj2Re
            radius_err_max = self["radius_err_max"] * Rj2Re
        elif kwargs.get("err_method", -1) == -1:
            ret_err = False
            kwargs["err_method"] = 0  # Set to 0 for the function

        # Remove radius and its errors from kwargs (just in case)
        kwargs.pop("radius", None)
        kwargs.pop("radius_err_min", None)
        kwargs.pop("radius_err_max", None)

        # Estimate the mass
        mass, mass_err_min, mass_err_max = estimate_mass(
            radius=radius,
            radius_err_min=radius_err_min,
            radius_err_max=radius_err_max,
            **kwargs,
        )

        # Convert mass to Jupiter masses
        mass = mass * Me2Mj

        # Return?
        if not ret_err:
            return mass

        # Convert errors to Jupiter masses
        mass_err_min = mass_err_min * Me2Mj
        mass_err_max = mass_err_max * Me2Mj

        return mass, mass_err_min, mass_err_max

    def estimate_radius(
        self,
        **kwargs,
    ) -> Tuple[float, float, float]:
        r"""Calculate the radius of a planet using a power-law approximation.

        Equation:
            :math:`radius = C \\times mass^S`

        Parameters
        ----------
        kwargs : dict
            Keyword arguments for the
            :py:func:`resokit.utils.mass_radius.estimate_radius_single`
            function.

        Returns
        -------
        radius : float
            Estimated radius in Jupiter radii.
        radius_err_min : float
            Minimum error in Jupiter radii. If `err_method=0`, the error is 0.0.
        radius_err_max : float
            Maximum error in Jupiter radii. If `err_method=0`, the error is 0.0.
        """
        # Get planet mass and convert to Earth masses
        mass = self["mass"] * Mj2Me

        # Get the errors and convert to Earth masses, if needed (and available)
        ret_err = True
        mass_err_min = 0.0
        mass_err_max = 0.0
        if kwargs.get("err_method", 0) in [1, 2]:
            mass_err_min = self["mass_err_min"] * Mj2Me
            mass_err_max = self["mass_err_max"] * Mj2Me
        elif kwargs.get("err_method", -1) == -1:
            ret_err = False
            kwargs["err_method"] = 0  # Set to 0 for the function

        # Remove mass and its errors from kwargs (just in case)
        kwargs.pop("mass", None)
        kwargs.pop("mass_err_min", None)
        kwargs.pop("mass_err_max", None)

        # Estimate the radius
        radius, radius_err_min, radius_err_max = estimate_radius(
            mass=mass,
            mass_err_min=mass_err_min,
            mass_err_max=mass_err_max,
            **kwargs,
        )

        # Convert radius to Jupiter radii
        radius = radius * Re2Rj

        # Return?
        if not ret_err:
            return radius

        # Convert errors to Jupiter radii
        radius_err_min = radius_err_min * Re2Rj
        radius_err_max = radius_err_max * Re2Rj

        return radius, radius_err_min, radius_err_max

    def plot(
        self,
        x: str,
        y: str,
        error_x: bool = False,
        error_y: bool = False,
        ax: plt.Axes = None,
        label: Union[bool, str] = True,
        **plot_kwargs: dict,
    ) -> plt.Axes:
        """Plot the x vs y data of the planet.

        Parameters
        ----------
        x : str
            Name of the column to use as x-axis.
        y : str
            Name of the column to use as y-axis.
        error_x : bool, optional. Default: False.
            Whether to plot the x error bars.
        error_y : bool, optional. Default: False.
            Whether to plot the y error bars.
        ax : plt.Axes, optional. Default: None.
            Matplotlib Axes to plot on.
            If None, get and use the current Axes.
        label : bool, str, optional. Default: True.
            String to use as the label.
            If True, use the planet name.
        plot_kwargs : dict
            Additional keyword arguments for the :py:func:`plt.errorbar`
            function.

        Returns
        -------
        ax : Matplotlib Axes
            `Matplotlib Axes` with the plot.
        """
        if label is True:
            label = self.name
        return super().plot(
            x=x,
            y=y,
            error_x=error_x,
            error_y=error_y,
            ax=ax,
            label=label,
            **plot_kwargs,
        )

    def copy(self) -> "StaticPlanet":
        """Return a copy of the StaticPlanet."""
        return StaticPlanet(
            data_df=self.data_df.copy(),
            source=self.source,
            metadata=self.metadata,
        )


@attrs.define(repr=False, frozen=True, slots=True)
class StaticStar(ResokitDataFrame):
    """StaticStar class representing a static star.

    Attributes
    ----------
    data_df : pd.Series
        Series containing the data.
    source : str
        Source of the dataset. Either 'eu' or 'nasa' or 'user'.
    metadata : dict
        Metadata of the dataset.
    name : str
        Name of the star.
    user_defined_ : bool
        Flag indicating if the star is user-defined.
    web_page : str
        Web page of the star.
    """

    name: str = attrs.field(init=False)
    user_defined_: bool = attrs.field(init=False)
    web_page: str = attrs.field(init=False)

    @name.default
    def _name_default(self):
        """Set the default value for name."""
        if "star_name" in self.data_df.index:
            return self.data_df["star_name"]

        return self.data_df["name"]  # ["name"] because .name is a df method

    @user_defined_.default
    def _user_defined__default(self):
        """Set the default value for user_defined_."""
        return self.source not in ["eu", "nasa"]

    @web_page.default
    def _web_page_default(self):
        """Set the default value for web_page."""
        if self.source == "nasa":
            aux = str(self.name).replace(" ", "%20")
            return (
                "https://exoplanetarchive.ipac.caltech.edu/overview/"
                + aux
                + "/"
            )
        return ""

    def __attrs_post_init__(self):
        """Post-initialization hook."""
        # Assert data_series is a Series
        if not isinstance(self.data_df, pd.Series):
            raise TypeError(
                "StaticStar must have a pd.Series. "
                + f"Got: {type(self.data_df)} instead."
            )

        # Check if all columns are in the default mapping
        if not self.user_defined_:
            aux_cols = {
                col.replace("star_", "") for col in RESO_SR_TYPES.keys()
            }
            for col in self.data_df.index:
                if col not in aux_cols | RESO_OB_TYPES.keys():
                    warnings.warn(
                        "Found columns not in the default star mapping.",
                        stacklevel=2,
                    )
                    print(col)

    def __getitem__(self, key: Union[int, str, list]):
        """x[y] <==> x.__getitem__(y)."""
        key = parse_to_iter(key)

        if any(isinstance(i, int) for i in key):
            raise IndexError(
                "StaticStar does not support integer indexing. "
                + "Use the 'name' column instead."
            )

        if len(key) == 1:
            return self.data_df[key[0]]

        return self.data_df[key]

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""
        text = f"StaticStar [{self.name}]"
        if not self.user_defined_:
            text += f" from {self.source} data source"
        else:
            text += " user defined"

        return text

    def _repr_html_(self):
        """Return a HTML representation of the StaticStar."""
        ad_id = id(self)
        prefoot = f"StaticStar [{self.name}]"
        if not self.user_defined_:
            prefoot += f" from {self.source} data source"
        else:
            prefoot += " user defined"

        return super()._repr_html_(ad_id=ad_id, prefoot=prefoot, switch=True)

    def plot(
        self,
        x: str,
        y: str,
        error_x: bool = False,
        error_y: bool = False,
        ax: plt.Axes = None,
        label: Union[bool, str] = True,
        **plot_kwargs: dict,
    ) -> plt.Axes:
        """Plot the x vs y data of the star.

        Parameters
        ----------
        x : str
            Name of the column to use as x-axis.
        y : str
            Name of the column to use as y-axis.
        error_x : bool, optional. Default: False.
            Whether to plot the x error bars.
        error_y : bool, optional. Default: False.
            Whether to plot the y error bars.
        ax : plt.Axes, optional. Default: None.
            Matplotlib Axes to plot on.
            If None, get and use the current Axes.
        label : bool, str, optional. Default: True.
            String to use as the label.
            If True, use the star name.
        plot_kwargs : dict
            Additional keyword arguments for the :py:func:`plt.errorbar`
            function.

        Returns
        -------
        ax : Matplotlib Axes
            `Matplotlib Axes` with the plot.
        """
        if label is True:
            label = self.name
        return super().plot(
            x=x,
            y=y,
            error_x=error_x,
            error_y=error_y,
            ax=ax,
            label=label,
            **plot_kwargs,
        )

    def copy(self) -> "StaticStar":
        """Return a copy of the StaticStar."""
        return StaticStar(
            data_df=self.data_df.copy(),
            source=self.source,
            metadata=self.metadata,
        )


@attrs.define(repr=False, frozen=True, slots=True)
class StaticSystem:
    """StaticSystem class representing a static system.

    Contains a star and a list of planets.

    Attributes
    ----------
    star : StaticStar
        StaticStar instance.
    planets : list[StaticPlanet], tuple[StaticPlanet], StaticPlanet
        List or tuple of StaticPlanet instances, or a single StaticPlanet.
    name : str
        Name of the system.
    metadata : dict
        Metadata of the dataset.
    web_page : list[str]
        Web page(s) of the system.
    n_planets_ : int
        Number of planets in this static system.
    source_ : str
        Source of the data.
    user_defined_ : bool
        Flag indicating if the system is user-defined.
    planet_names_ : list[str]
        List of planet names.
    """

    star: StaticStar = attrs.field(
        validator=attrs.validators.instance_of(StaticStar),
    )
    planets: Union[List[StaticPlanet], Tuple[StaticPlanet], StaticPlanet] = (
        attrs.field(
            validator=attrs.validators.instance_of(
                (list, tuple, StaticPlanet)
            ),
            converter=lambda x: (
                [x] if isinstance(x, StaticPlanet) else list(x)
            ),
        )
    )
    name: str = attrs.field(
        validator=attrs.validators.instance_of(str), default="unnamed"
    )
    metadata: dict = attrs.field(factory=MetaData, converter=MetaData)
    web_page: list = attrs.field(init=False)

    n_planets_: int = attrs.field(init=False)
    source_: str = attrs.field(init=False)
    user_defined_: bool = attrs.field(init=False)
    planet_names_: list = attrs.field(init=False)

    period_ratios_: Union[float, pd.DataFrame] = attrs.field(init=False)
    __error_ratios__: Union[float, pd.DataFrame] = attrs.field(init=False)

    @n_planets_.default
    def _n_planets__default(self):
        """Set the default value for n_planets_."""
        return len(self.planets)

    @source_.default
    def _source__default(self):
        """Set the default value for source_."""
        main_source = self.star.source

        return (
            main_source
            if all([planet.source == main_source for planet in self.planets])
            else "user"
        )

    @user_defined_.default
    def _user_defined__default(self):
        """Set the default value for user_defined_."""
        return self.source_ not in ["eu", "nasa"]

    @planet_names_.default
    def _planet_names__default(self):
        """Set the default value for planet_names_."""
        return [planet.name for planet in self.planets]

    @period_ratios_.default
    def _period_ratios__default(self):
        """Set the default value for period_ratios_."""
        if self.n_planets_ == 1:
            return None
        elif self.n_planets_ == 2:
            return self.planets[1].P / self.planets[0].P

        return pd.DataFrame()  # Empty mutable DataFrame

    @__error_ratios__.default
    def ___error_ratios__default(self):
        """Set the default value for __error_ratios__."""
        if self.n_planets_ == 1:
            return None
        elif self.n_planets_ == 2:
            error_0 = max(self.planets[0].P_err_min, self.planets[0].P_err_max)
            error_1 = max(self.planets[1].P_err_min, self.planets[1].P_err_max)
            return self.period_ratios_ * sqrt(
                (error_0 / self.planets[0].P) ** 2
                + (error_1 / self.planets[1].P) ** 2
            )

        return pd.DataFrame()  # Empty mutable DataFrame

    @web_page.default
    def _web_page_default(self):
        """Set the default value for web_page."""
        return [
            self.star.web_page,
            *[planet.web_page for planet in self.planets],
        ]

    def __attrs_post_init__(self):
        """Post-initialization hook."""
        star_name = self.star.name

        # Check if all planets are StaticPlanet instances
        # and if they have the same star name
        for planet in self.planets:
            if not isinstance(planet, StaticPlanet):
                raise TypeError(
                    " planets must be a StaticPlanet instance,"
                    + " or a list|tuple of StaticPlanet instances."
                    + f" Got: {type(planet)} instead"
                )
            if planet.star_name != star_name:
                warnings.warn(
                    f"Planet({planet.name}) star name({planet.star_name})"
                    + f" is different from Star({star_name}).",
                    stacklevel=2,
                )

        # Check if all planets have unique names
        if self.n_planets_ != len(set(self.planet_names_)):
            warnings.warn("Planets must have unique names.", stacklevel=2)

        return

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""
        star_msg = "\n Star:\n  " + f"{self.star.name}"

        planets_msg = (
            "\n"
            + f" Planet{'s' if self.n_planets_ > 1 else ''}:"
            + "\n  "
            + "\n  ".join(self.planet_names_)
        )

        return (
            "StaticSystem: "
            + f"{star_msg} "
            + f"{planets_msg}"
            + "\n"
            + f" from '{self.source_}' data source."
            if not self.user_defined_
            else ""
        )

    def __getitem__(self, key: Union[int, str]):
        """x[y] <==> x.__getitem__(y).

        Parameters
        ----------
        key : int, str
            Integer or list of integers to slice planets,
            or strings for attributes.

        Returns
        -------
        A sliced planet object or specific items of the system.
        """
        key = parse_to_iter(key)

        if all(isinstance(i, int) for i in key):
            return self.planet(key)
        elif any(isinstance(i, int) for i in key):
            raise NotImplementedError(
                "Mixed integer and string indexing not supported."
            )

        return self.get_item(key)

    def __len__(self):
        """len(x) <=> x.__len__()."""
        return self.n_planets_ + 1

    def planet(self, indices: Union[int, Iterable[int]]) -> StaticPlanet:
        """Slice the planets by given indices.

        Parameters
        ----------
        indices : int, Iterable[int]
            Indices for slicing planets.

        Returns
        -------
        planet : StaticPlanet or list[StaticPlanet]
            A copy of a system's planet :py:class:`StaticPlanet`
            or list of :py:class:`StaticPlanet` objects.
        """
        indices = parse_to_iter(indices, to=list)

        if not all(isinstance(i, int) for i in indices):
            raise TypeError("Indices must be integers.")

        if len(indices) == 1:
            return self.planets[indices[0]].copy()

        return [self.planets[i].copy() for i in indices]

    def _get_planets_items(
        self, items: Union[str, List[str]], return_values: bool = True
    ) -> Union[str, List[str]]:
        """Retrieve specific attributes of planets.

        Parameters
        ----------
        items : str, list[str]
            Names of planet attributes.
        return_values : bool, default=True
            Whether to return values or full objects.

        Returns
        -------
        items : list
            Values or full objects of the specified planet attributes.
        """
        data = [planet[items] for planet in self.planets]

        if return_values:
            try:
                return [item.values[0] for item in data]
            except AttributeError:
                pass  # Fall back to full objects

        return [item for item in data]

    def get_item(
        self, items: Union[str, List[str]], error: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """Retrieve specific attributes of the system (star and/or planets).

        Parameters
        ----------
        items : str, list[str]
            Names of the desired attributes.
        error : bool, optional. Default: False.
            Whether to return the error columns.
            Only available for standard ResokitDataFrame objects.

        Returns
        -------
        data : pandas series or pandas dataframe
            Pandas Series or DataFrame with the requested items.
        """
        items = parse_to_iter(items)

        if error:
            items_with_error = [
                item
                for item in items
                for item in (item, f"{item}_err_min", f"{item}_err_max")
            ]
            items = [
                item for item in items_with_error if item in RESO_DTYPES.keys()
            ]

        if len(items) == 1:
            return self._get_single_item(items[0])

        if self.n_planets_ == 1:
            return self._get_single_planet_items(items)

        return self._get_multiple_planet_items(items)

    def _get_single_item(self, item: str):
        """Handle retrieval when a single item is requested."""
        if item.startswith("star_"):
            return self.star[item.replace("star_", "")]

        if self.n_planets_ > 1:
            return pd.Series(
                self._get_planets_items(item),
                index=self.planet_names_,
                name=item,
            )

        return self.planets[0][item]

    def _get_single_planet_items(self, items: List[str]):
        """Retrieve attributes when there's only one planet."""
        return pd.Series(
            {
                item: (
                    self.star[item.replace("star_", "")]
                    if item.startswith("star_")
                    else self.planets[0][item]
                )
                for item in items
            },
            name=self.name,
        )

    def _get_multiple_planet_items(self, items: List[str]):
        """Retrieve attributes when there are multiple planets."""
        # Create a DataFrame with the requested items
        df = pd.DataFrame(
            {
                item: (
                    self._get_planets_items(item)
                    if not item.startswith("star_")
                    else [None] * self.n_planets_
                )
                for item in items
            },
            index=self.planet_names_,
        )

        # Add star attributes
        for col in df.columns:
            if col.startswith("star_"):
                df[col] = self.star[col.replace("star_", "")]

        return df

    def estimate_period(
        self, which: Union[str, int, List[int]] = "all", err_method: int = 0
    ) -> Union[Tuple[float, float, float], pd.DataFrame]:
        r"""Estimate the period of selected planets in the system.

        Calculate the period of the planet using the third Kepler's law.

        Parameters
        ----------
        which : str, int, list[int], optional. Default: 'all'.
            Which planets to estimate the period. Union[float, pd.Series]:
            If 'all', estimate all planets period.
            If an :py:class:`int`, estimate the period of the planet with the
            given index.
            For example:
            *0* will estimate the period of the first planet;
            *1* will estimate the period of the second planet.
            If a list of integers, estimate the period of the planets with the
            given indices.
        err_method : int, optional. Default: -1.
            Method to estimate the error.
            See py:func:`resokit.utils.calc_period_with_errors` for more
            details.
            *-1*: Nothing. Do not estimate the error.
            *0* : No propagation. Return both errors as 0.0.
            *1* : Extremes. Estimate the period at the extreme values of
            each parameter and retrieve the errors from the difference.
            *2* : Extended propagation. Assume each parameters follows a normal
            distribution with sigma = err_max.
            *3* : Centred propagation. Assume each parameters follows a normal
            distribution with sigma = (err_min + err_max) / 2.
            *4* : Deviated propagation. Assume each parameters follows a normal
            distribution with sigma = (err_max + err_min) / 2, but the
            mean is at ((val + err_min) + (val + err_max)) / 2.

        Returns
        -------
        period, period_err_min, period_err_max : tuple or DataFrame
            Estimated period, and its minimum and maximum errors,
            in days.

        """
        if which == "all":
            which = list(range(self.n_planets_))
        else:
            which = parse_to_iter(which)

        if all(isinstance(i, int) for i in which):
            df = pd.DataFrame()  # Create an empty DataFrame

            for i in which:  # Iterate over the planets
                pl = self.planets[i]
                per, per_err_min, per_err_max = calc_period_with_errors(
                    pl.a,
                    pl.a_err_min,
                    pl.a_err_max,
                    self.star.mass,
                    self.star.mass_err_min,
                    self.star.mass_err_max,
                    pl.mass,
                    pl.mass_err_min,
                    pl.mass_err_max,
                    err_method,
                )
                df[f"{pl.name}"] = [per, per_err_min, per_err_max]
            df.index = ["P", "P_err_min", "P_err_max"]

            if err_method == -1:  # No error requested
                return df.loc["P"]  # Return only the period

            return df.T  # Return the DataFrame

        raise ValueError("Invalid value for 'which'.")

    def estimate_semi_major_axis(
        self, which: Union[str, int, List[int]] = "all", err_method: int = 0
    ) -> Union[Tuple[float, float, float], pd.DataFrame]:
        r"""Estimate the semi-major axis of selected planets in the system.

        Parameters
        ----------
        which : str, int, list[int], optional. Default: 'all'.
            Which planets to estimate the semi-major axis.
            If 'all', estimate all planets semi-major axis.
            If an :py:class:`int`, estimate the semi-major axis of the planet
            with the given index.
            For example:
            *0* will estimate the semi-major axis of the first planet;
            *1* will estimate the semi-major axis of the second planet.
            If a list of integers, estimate the semi-major axis of the planets
            with the given indices.
        err_method : int, optional. Default: -1.
            Method to estimate the error.
            See py:func:`resokit.utils.calc_semi_major_axis_with_errors` for
            more details.
            *-1*: Nothing. Do not estimate the error.
            *0* : No propagation. Return both errors as 0.0.
            *1* : Extremes. Estimate the semi-major axis at the extreme
            values of each parameter and retrieve the errors from the
            difference.
            *2* : Extended propagation. Assume each parameters follows a normal
            distribution with sigma = err_max.
            *3* : Centred propagation. Assume each parameters follows a normal
            distribution with sigma = (err_min + err_max) / 2.
            *4* : Deviated propagation. Assume each parameters follows a normal
            distribution with sigma = (err_max + err_min) / 2, but the
            mean is at ((val + err_min) + (val + err_max)) / 2.

        Returns
        -------
        a, a_err_min, a_err_max : tuple or DataFrame
            Estimated semi-major axis, and its minimum and maximum errors,
            in AU.
        """
        if which == "all":
            which = list(range(self.n_planets_))
        else:
            which = parse_to_iter(which)

        if all(isinstance(i, int) for i in which):
            df = pd.DataFrame()  # Create an empty DataFrame

            for i in which:  # Iterate over the planets
                pl = self.planets[i]
                a, a_err_min, a_err_max = calc_a_with_errors(
                    pl.P,
                    pl.P_err_min,
                    pl.P_err_max,
                    self.star.mass,
                    self.star.mass_err_min,
                    self.star.mass_err_max,
                    pl.mass,
                    pl.mass_err_min,
                    pl.mass_err_max,
                    err_method,
                )
                df[f"{pl.name}"] = [
                    a,
                    a_err_min,
                    a_err_max,
                ]
            df.index = ["a", "a_err_min", "a_err_max"]

            if err_method == -1:  # No error requested
                return df.loc["a"]  # Return only the semi-major axis

            return df.T  # Return the DataFrame

        raise ValueError("Invalid value for 'which'.")

    def estimate_mass(
        self, which: Union[str, int, List[int]] = "all", **kwargs
    ) -> Union[Tuple[float, float, float], pd.DataFrame]:
        r"""Estimate the mass of selected planets in the system.

        Parameters
        ----------
        which : str, int, list[int], optional. Default: 'all'.
            Which planets to estimate the mass.
            If 'all', estimate all planets mass.
            If an :py:class:`int`, estimate the planet with the given index.
            For example:
            *0* will estimate the mass of the first planet;
            *1* will estimate the mass of the second planet.
            If a list of integers, estimate the mass of the planets with the
            given indices.
        **kwargs : dict
            Additional keyword arguments for the
            :py:func:`resokit.utils.mass_radius.estimate_mass` function.

        Note
        ----
        If `err_method=-1`, only the mass is returned. If `err_method=0`, the
        errors are 0.0.


        Returns
        -------
        mass, mass_err_min, mass_err_max : tuple or DataFrame
            Estimated mass, and its minimum and maximum errors,
            in Jupiter masses.
        """
        if which == "all":  # Estimate all planets
            which = list(range(self.n_planets_))
        else:
            which = parse_to_iter(which)  # Ensure it's an iterable

        if all(isinstance(i, int) for i in which):
            df = pd.DataFrame()  # Create an empty DataFrame

            # Define the error method
            ret_err = True
            if kwargs.get("err_method", -1) == -1:
                kwargs["err_method"] = 0  # Set to 0 for the function
                ret_err = False

            for i in which:  # Iterate over the planets
                mass, mass_err_min, mass_err_max = self.planets[
                    i
                ].estimate_mass(**kwargs)
                df[f"{self.planets[i].name}"] = [
                    mass,
                    mass_err_min,
                    mass_err_max,
                ]
            df.index = ["mass", "mass_err_min", "mass_err_max"]

            if not ret_err:  # No error requested
                return df.loc["mass"]  # Return only the mass

            return df.T  # Return the DataFrame

        raise ValueError("Invalid value for 'which'.")

    def estimate_radius(
        self, which: Union[str, int, List[int]] = "all", **kwargs
    ) -> Union[Tuple[float, float, float], pd.DataFrame]:
        r"""Estimate the radius of selected planets in the system.

        Parameters
        ----------
        which : str, int, list[int], optional. Default: 'all'.
            Which planets to estimate the radius.
            If 'all', estimate all planets radius.
            If an :py:class:`int`, estimate the planet with the given index.
            For example:
            *0* will estimate the radius of the first planet;
            *1* will estimate the radius of the second planet.
            If a list of integers, estimate the radius of the planets with the
            given indices.
        **kwargs : dict
            Additional keyword arguments for the
            :py:func:`resokit.utils.mass_radius.estimate_radius` function.

        Note
        ----
        If `err_method=-1`, only the radius is returned. If `err_method=0`, the
        errors are 0.0.

        Returns
        -------
        radius, radius_err_min, radius_err_max : tuple or DataFrame
            Estimated radius, and its minimum and maximum errors,
            in Earth radii.
        """
        if which == "all":  # Estimate all planets
            which = list(range(self.n_planets_))
        else:
            which = parse_to_iter(which)  # Ensure it's an iterable

        if all(isinstance(i, int) for i in which):
            df = pd.DataFrame()  # Create an empty DataFrame

            # Define the error method
            ret_err = True
            if kwargs.get("err_method", -1) == -1:
                kwargs["err_method"] = 0
                ret_err = False

            for i in which:
                radius, radius_err_min, radius_err_max = self.planets[
                    i
                ].estimate_radius(**kwargs)
                df[f"{self.planets[i].name}"] = [
                    radius,
                    radius_err_min,
                    radius_err_max,
                ]
            df.index = ["radius", "radius_err_min", "radius_err_max"]

            if not ret_err:  # No error requested
                return df.loc["radius"]  # Return only the radius

            return df.T  # Return the DataFrame

        raise ValueError("Invalid value for 'which'.")

    def estimate_hill_radius(
        self,
        which: Union[str, int, List[int]] = "all",
        err_method: int = 0,
    ) -> Union[Tuple[float, float, float], pd.DataFrame]:
        """Calculate the Hill radius of selected planets in the system.

        Parameters
        ----------
        which : str, int, list[int], optional. Default: 'all'.
            Which planets to calculate the Hill radius.
            If 'all', calculate the Hill radius of all planets.
            If an :py:class:`int`, calculate the Hill radius of the planet with
            the given index.
            For example:
            *0* will calculate the Hill radius of the first planet;
            *1* will calculate the Hill radius of the second planet.
            If a list of integers, calculate the Hill radius of the planets
            with the given indices.
        err_method : int, optional. Default: -1.
            Method to estimate the error.
            See py:func:`resokit.utils.hill_radius.hill_radius_with_errors` for
            more details.
            *-1*: Nothing. Do not estimate the error.
            *0* : No propagation. Return both errors as 0.0.
            *1* : Extremes. Estimate the semi-major axis at the extreme
            values of each parameter and retrieve the errors from the
            difference.
            *2* : Extended propagation. Assume each parameters follows a normal
            distribution with sigma = err_max.
            *3* : Centred propagation. Assume each parameters follows a normal
            distribution with sigma = (err_min + err_max) / 2.
            *4* : Deviated propagation. Assume each parameters follows a normal
            distribution with sigma = (err_max + err_min) / 2, but the
            mean is at ((val + err_min) + (val + err_max)) / 2.

        Returns
        -------
        rhill, rhill_err_min, rhill_err_max : tuple or DataFrame
            Hill radius, and its minimum and maximum errors,
            in AU.
        """
        if which == "all":
            which = list(range(self.n_planets_))
        else:
            which = parse_to_iter(which)

        if all(isinstance(i, int) for i in which):
            df = pd.DataFrame()  # Create an empty DataFrame

            for i in which:  # Iterate over the planets
                pl = self.planets[i]
                hill, hill_err_min, hill_err_max = hill_radius_with_errors(
                    pl.a,
                    pl.a_err_min,
                    pl.a_err_max,
                    pl.e,
                    pl.e_err_min,
                    pl.e_err_max,
                    self.star.mass,
                    self.star.mass_err_min,
                    self.star.mass_err_max,
                    pl.mass,
                    pl.mass_err_min,
                    pl.mass_err_max,
                    err_method,
                )
                df[f"{pl.name}"] = [
                    hill,
                    hill_err_min,
                    hill_err_max,
                ]
            df.index = ["hill", "hill_err_min", "hill_err_max"]

            if err_method == 0:  # No error
                return df.loc["hill"]  # Return only the mass

            return df.T  # Return the DataFrame

        raise ValueError("Invalid value for 'which'.")

    def plot(
        self,
        x: str,
        y: str,
        error_x: bool = False,
        error_y: bool = False,
        ax: plt.Axes = None,
        label: Union[bool, str, Iterable[str]] = True,
        plot_kwargs: dict = None,
    ) -> plt.Axes:
        """Plot the x vs y data of the system.

        Uses :py:func:`plt.errorbar` internally.

        Parameters
        ----------
        x : str
            Name of the column to use as x-axis.
        y : str
            Name of the column to use as y-axis.
        error_x : bool, optional. Default: False.
            Whether to plot the x error bars.
        error_y : bool, optional. Default: False.
            Whether to plot the y error bars.
        ax : plt.Axes, optional. Default: None.
            Matplotlib Axes to plot on.
            If None, get and use the current Axes.
        label : bool, str, Iterable, optional. Default: True.
            Whether to add a label with the planet (or star) names.
            If str, use the string as the label.
            If Iterable, use the list of strings as the label.
        plot_kwargs : dict
            Additional keyword arguments for the :py:func:`plt.errorbar`
            function.

        Returns
        -------
        ax : Matplotlib Axes
            `Matplotlib Axes` with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if x.startswith("star_") and y.startswith("star_"):
            star_plot = True
        elif x.startswith("star_") or y.startswith("star_"):
            raise ValueError("Both x and y must be star or planet attributes.")
        else:
            star_plot = False

        # Check plot_kwargs
        if plot_kwargs is None:
            plot_kwargs = {}

        if not star_plot:  # Planet plot

            # Check label
            if label is True:
                # True means use planet names
                label = [label] * self.n_planets_
            elif isinstance(label, str):
                # If label is a string, use it for the last planet
                label = [False] * (self.n_planets_ - 1) + [label]
            elif len(label) != self.n_planets_:
                raise ValueError(
                    "Length of planet_label must be equal "
                    + "to the number of planets."
                )

            # Plot planets
            for i, planet in enumerate(self.planets):
                ax = planet.plot(
                    x=x,
                    y=y,
                    error_x=error_x,
                    error_y=error_y,
                    ax=ax,
                    label=label[i],
                    **plot_kwargs,
                )

            return ax

        # Star plot
        x = x.replace("star_", "")
        y = y.replace("star_", "")

        return self.star.plot(x, y, error_x, error_y, ax, label, plot_kwargs)

    def plot_triplet(
        self,
        which: Union[str, int] = "all",
        error: bool = False,
        ax: plt.Axes = None,
        label: Union[str, list, bool] = True,
        **kwargs,
    ) -> plt.Axes:
        """Plot consecutive triplets of planets in the period ratio space.

        Systems triplets are shown in the plane
        :math:`P_{i+1}/P_i` vs. :math:`P_{i+2}/P_{i+1}`.

        Parameters
        ----------
        which : int, str, optional. Default: 'all'.
            Which triplets to plot.
            If 'all', plot all possible triplets.
            If an :py:class:`int`, plot the triplet with the given index.
            For example:
            *0* will plot the first triplet: (0, 1, 2);
            *1* will plot the second triplet: (1, 2, 3).
        error : bool, optional. Default: False.
            Whether to plot the error bars.
        ax : plt.Axes, optional. Default: None.
            Matplotlib Axes to plot on.
            If None, get and use the current Axes.
        label : str, list, bool, optional. Default: True.
            Label for the data plotted.
            If True, will (try to) concatenate each three planets suffixes to
            create triplets labels.
        **kwargs : dict
            Additional keyword arguments for the :py:func:`plt.errorbar`
            function.

        Returns
        -------
        ax : Matplotlib Axes
            `Matplotlib Axes` with the plot.
        """
        # Check if the system has at least 3 planets
        if self.n_planets_ < 3:
            raise ValueError("There must be at least 3 planets to compare.")

        # Get (all) period ratios
        period_ratios = self.period_ratios

        # Get (all) error ratios if needed
        if error:
            error_ratios = self.pair_ratio(error=True, verbose=False)

        # Check which triplets to plot. Remember they are consecutive.
        if which == "all":
            triplets = [(i, i + 1, i + 2) for i in range(self.n_planets_ - 2)]
        elif isinstance(which, int):
            if which < 0 or which >= self.n_planets_ - 2:
                raise ValueError("Index out of range.")
            triplets = [(which, which + 1, which + 2)]
        else:
            raise ValueError("Invalid value for 'which'.")

        # Create a new figure if ax is None
        if ax is None:
            ax = plt.gca()

        # Check label
        if label:
            if label is True:
                # For the label, use suffixes if they are unique
                suffixes = [planet.suffix_ for planet in self.planets]
                use_suffix = len(suffixes) == len(set(suffixes))
            elif isinstance(label, str):
                use_suffix = False
                label = [label] * len(triplets)
            elif isinstance(label, Iterable):
                use_suffix = False
                if len(label) != len(triplets):
                    raise ValueError(
                        "Length of label must be equal to the number of "
                        + "triplets to plot."
                    )
            else:
                raise ValueError("Invalid value for 'label'.")
        else:
            label_aux = False

        # Check plot_kwargs
        if kwargs is None:
            kwargs = {}

        # Extract the format from kwargs
        fmt = kwargs.pop("fmt", "o")

        # Plot each triplet
        for trip, (i, j, k) in enumerate(triplets):
            if label is True and not use_suffix:
                label_aux = "".join([str(i), str(j), str(k)])
            elif label is True and use_suffix:
                label_aux = "".join(
                    [self.planets[idx].suffix_ for idx in [i, j, k]]
                )
            elif label:
                label_aux = label[trip]
            x = period_ratios.iloc[i, j]
            y = period_ratios.iloc[j, k]
            err_x = error_ratios.iloc[i, j] if error else None
            err_y = error_ratios.iloc[j, k] if error else None
            ax.errorbar(
                x,
                y,
                xerr=err_x,
                yerr=err_y,
                label=label_aux,
                fmt=fmt,
                **kwargs,
            )

        return ax

    def remove_planet(
        self, index: Union[int, str], verbose: bool = True
    ) -> "StaticSystem":
        """Remove a planet from the system.

        Parameters
        ----------
        index : int, str
            Index or suffix (1 char) or name of the planet to remove.
        verbose : bool, optional. Default: True.
            Whether to print a message when removing the planet.

        Returns
        -------
        StaticSystem
            A new :py:class`StaticSystem` instance without the removed planet.
        """
        if isinstance(index, str):  # Remove by name or suffix

            if len(index) == 1:  # Remove by suffix
                indexes = [
                    planet.suffix_ for planet in self.planets
                ]  # Suffixes
                if len(indexes) != len(set(indexes)):
                    raise ValueError("Suffixes must be unique to remove.")
                if index not in indexes:
                    raise ValueError(f"Suffix '{index}' not found in planets.")
                index = indexes.index(index)  # Get index from suffix

            else:  # Remove by name
                if self.n_planets_ != len(set(self.planet_names_)):
                    raise ValueError(
                        "Planets must have unique names to remove."
                    )
                if index not in self.planet_names_:
                    raise ValueError(f"Name '{index}' not found in planets.")
                index = self.planet_names_.index(index)  # Get index from name

        if index < 0 or index >= self.n_planets_:
            raise IndexError("Index out of range.")

        # Create a new list of planets
        new_planets = [  # This way to avoid "index + 1 :" <BLACK>
            self.planets[i] for i in range(self.n_planets_) if i != index
        ]

        # Create a new metadata dictionary
        new_meta = self.to_dict()
        if "removed_planet" not in new_meta:
            new_meta["removed_planet"] = self.planets[index].name
        else:
            new_meta["removed_planet"] += f", {self.planets[index].name}"

        # Create a new StaticSystem instance
        ss = StaticSystem(
            star=self.star,
            planets=new_planets,
            name=self.name,
            metadata=new_meta,
        )

        # Print message
        if verbose:
            print(f"Planet {self.planets[index].name} [{index}] removed.")

        return ss

    def add_planet(
        self,
        planet: StaticPlanet,
        sort: Union[bool, str] = True,
        verbose: bool = True,
    ) -> "StaticSystem":
        """Add a planet to the system.

        Parameters
        ----------
        planet : StaticPlanet
            StaticPlanet instance to add.
        sort : bool, str, optional. Default: True.
            Whether to sort the planets by period.
            If str, sort by the specified column.
        verbose : bool, optional. Default: True.
            Whether to print a message when adding the planet.

        Returns
        -------
        StaticSystem
            A new :py:class`StaticSystem` instance.
        """
        if not isinstance(planet, StaticPlanet):
            raise TypeError(
                "planet must be a StaticPlanet instance."
                + f" Got: {type(planet)} instead."
            )

        # Create a new list of planets
        new_planets = self.planets + [planet]

        if sort:
            if sort is True:
                sort_col = "P"
            elif isinstance(sort, str):
                sort_col = sort
            else:
                raise ValueError("Invalid value for 'sort'.")
            new_planets = sorted(new_planets, key=lambda x: x[sort_col])

        # Create a new metadata dictionary
        new_meta = self.to_dict()
        if "added_planet" not in new_meta:
            new_meta["added_planet"] = planet.name
        else:
            new_meta["added_planet"] += f", {planet.name}"

        # Create a new StaticSystem instance
        ss = StaticSystem(
            star=self.star,
            planets=new_planets,
            name=self.name,
            metadata=new_meta,
        )

        # Print message
        if verbose:
            print(f"Planet {planet.name} added.")

        return ss

    @property
    def period_ratios(self) -> Union[float, pd.DataFrame]:
        """Calculate and return the period ratios of all the planets.

        Created after calling the period_ratios method.

        Returns
        -------
        float, pd.DataFrame
            Float with period ratio of the pair of planets, or DataFrame
            with all the period ratios.
        """
        if self.n_planets_ < 2:
            raise ValueError("There must be at least 2 planets to compare.")

        if self.n_planets_ == 2:
            return self.period_ratios_

        if not self.period_ratios_.empty:
            return self.period_ratios_

        return self.pair_ratio()

    def pair_ratio(
        self,
        *pair: Union[list, tuple, str],
        verbose: bool = True,
        error: bool = False,
        **fraction_kwargs: dict,
    ) -> Union[float, pd.DataFrame]:
        """Return the period ratio of the specified pair of planets.

        Parameters
        ----------
        pair : list, tuple, str, optional. Default: 'all'.
            Which pair of planets to consider.
            Either 'all' or a list/tuple of planet names/indexes.
            If *pair=(i,j)*, then the period ratio is :math:`P_j/P_i`, and
            remember that the first planet is 0.
        verbose : bool, optional. Default: False.
            Whether to print the steps of the calculation if a single pair,
            and fraction_arg is not 0.
        error : bool, optional. Default: False.
            Whether to return the error of the period ratio, instead of the
            period ratio itself.
        fraction_kwargs : dict, optional
            Keyword arguments for the float_to_fraction function.
            If None, no fraction conversion is done.
            See float_to_fraction for more information.

        Returns
        -------
        ratios : float, pd.DataFrame
            Float with period ratio of the pair of planets, or pandas Data frame
            with all the period ratios.
        """
        # Check if there are at least 2 planets
        if self.n_planets_ < 2:
            raise ValueError("There must be at least 2 planets to compare.")

        # Extract pair
        if not pair or pair == ("all",):
            pair = "all"
        elif len(pair) > 2:
            raise ValueError("Pair must have 2 elements.")
        elif len(pair) == 1:
            pair = pair[0]
            if not isinstance(pair, Iterable) or len(pair) != 2:
                raise ValueError("Pair must have 2 elements.")

        # If error is True, return the error of the period ratio
        if error:
            return self._pair_ratio_error(pair)

        # Add verbose to fraction_kwargs, if fraction_kwargs is not empty
        if fraction_kwargs:
            fraction_kwargs["verbose"] = verbose

        # This calculates all the period ratios
        if isinstance(pair, str):

            if not pair == "all":  # Check if it's 'all'
                raise ValueError("Invalid pair value.")

            if self.n_planets_ == 2:  # Only 2 planets
                if fraction_kwargs:  # Convert to fraction
                    return float_to_fraction(
                        self.period_ratios_,
                        **fraction_kwargs,
                    )
                return self.period_ratios_  # Already calculated

            if self.period_ratios_.empty:  # Calculate all the period ratios
                # Create a DataFrame with all the period ratios
                periods = self.get_item("P")
                df = pd.DataFrame(
                    [[p2 / p1 for p2 in periods] for p1 in periods],
                    index=periods.index,
                    columns=periods.index,
                )

                # Store the DataFrame
                self.period_ratios_[df.columns] = df

            if fraction_kwargs:
                return self.period_ratios_.map(
                    lambda x: float_to_fraction(
                        x,
                        **fraction_kwargs,
                    )
                )

            return self.period_ratios_

        # This is sigle pair

        idxs = []  # Indexes of the pair
        for idx in pair:

            if isinstance(idx, str):
                if len(idx) == 1:  # Suffix
                    idxs.append(
                        [planet.suffix_ for planet in self.planets].index(idx)
                    )
                else:  # Name
                    idxs.append(self.planet_names_.index(idx))
            elif isinstance(idx, int):  # Index
                idxs.append(idx)
            else:
                raise ValueError("Invalid pair value.")

        # Calculate the ratio
        if not self.period_ratios_.empty:
            ratio = self.period_ratios_.iloc[idxs[1], idxs[0]]
        else:
            ratio = self.planets[idxs[1]].P / self.planets[idxs[0]].P

        # Return the ratio
        if fraction_kwargs:
            return float_to_fraction(ratio, **fraction_kwargs)

        return ratio

    def _pair_ratio_error(
        self, *pair: Union[list, tuple, str]
    ) -> Union[float, pd.DataFrame]:
        """Return the period ratio error of the specified pair of planets.

        Parameters
        ----------
        pair : list, tuple, str, optional. Default: 'all'.
            Which pair of planets to consider.
            Either 'all' or a list/tuple of planet names/indexes.

        Returns
        -------
        float, pd.DataFrame
            Float with period ratio error of the pair of planets, or DataFrame
            with all the period ratio errors.
        """
        if self.n_planets_ <= 2:  # No error for 1 planet.
            return self.__error_ratios__  # Already calculated for 2 planets.

        # Extract pair ratio
        pair_ratio = self.pair_ratio(*pair, error=False)

        # Formula: sqrt((err1/P1)^2 + (err2/P2)^2) * ratio

        # If pair is all
        if isinstance(pair_ratio, pd.DataFrame):
            # Return the DataFrame if it's already calculated
            if not self.__error_ratios__.empty:
                return self.__error_ratios__
            # Create a DataFrame with all the period ratios
            sigma2 = pd.DataFrame(
                [
                    [
                        (
                            max(
                                abs(self.planets[i].P_err_min),
                                abs(self.planets[i].P_err_max),
                            )
                            / self.planets[i].P
                        )
                        ** 2
                        + (
                            max(
                                abs(self.planets[j].P_err_min),
                                abs(self.planets[j].P_err_max),
                            )
                            / self.planets[j].P
                        )
                        ** 2
                        for i in range(self.n_planets_)
                    ]
                    for j in range(self.n_planets_)
                ],
                index=pair_ratio.index,
                columns=pair_ratio.columns,
            )
            # Calculate the error
            df = pair_ratio * sqrt(sigma2)

            # Store the DataFrame
            self.__error_ratios__[df.columns] = df

            return df

        # If pair is a single pair

        idxs = []  # Indexes of the pair
        for idx in pair:

            if isinstance(idx, str):
                if len(idx) == 1:  # Suffix
                    idxs.append(
                        [planet.suffix_ for planet in self.planets].index(idx)
                    )
                else:  # Name
                    idxs.append(self.planet_names_.index(idx))
            elif isinstance(idx, int):  # Index
                idxs.append(idx)
            else:
                raise ValueError("Invalid pair value.")

        # Extract the indexes
        i, j = idxs

        # Calculate sigma2
        sigma2 = (
            max(self.planets[i].P_err_min, self.planets[i].P_err_max)
            / self.planets[i].P
        ) ** 2 + (
            max(self.planets[j].P_err_min, self.planets[j].P_err_max)
            / self.planets[j].P
        ) ** 2

        return pair_ratio * sqrt(sigma2)  # Return the error

    def to_dataframe(
        self, add_star: Union[bool, None] = True, columns: list = None
    ) -> pd.DataFrame:
        """Combine and return system objects data as a new pandas DataFrame.

        Parameters
        ----------
        add_star : bool, optional. Default: True.
            Whether to include the star data in the DataFrame as a row
            (same level as the planets).
            If False, the star data is included in the planets DataFrame
            as repeated rows.
            If None, do not include any star data.
        columns : list, optional. Default: None.
            Subset of columns to include in the DataFrame.

        Returns
        -------
        df : DataFrame
            Pandas Data frame with the data.
        """
        # Create a DataFrame with the planets data
        df = pd.DataFrame(
            {planet.name: planet.data_df for planet in self.planets}
        )

        if add_star is None:
            if columns is not None:
                used_cols = [col for col in columns if col in df.columns]
                df = df[used_cols]
            return df.T

        # Generate star data
        star_df = pd.Series(self.star.data_df).to_frame(self.star.name)

        # Drop RESO_OB_TYPES columns, as they are already in the planets
        drop2 = [col for col in RESO_OB_TYPES.keys() if col in star_df.index]
        star_df.drop(drop2, inplace=True)

        # Change star columns to inlclude "star_". Exclude RESO_OB_TYPES
        star_df = star_df.rename(lambda x: f"star_{x}")

        if add_star:
            # Concatenate star data
            df = pd.concat([star_df, df], axis=0)
        else:
            # Add the same star data for all planets
            vals = [val[0] for val in star_df.values]  # So messy
            new_rows = pd.DataFrame(
                {col: vals for col in df.columns},
                index=star_df.index,
            )
            df = pd.concat([new_rows, df])

        if columns is not None:
            used_cols = [col for col in columns if col in df.columns]
            df = df[used_cols]

        return df

    def to_dict(self) -> dict:
        """Return the metadata as a new dictionary."""
        return dict(self.metadata)

    def copy(self) -> "StaticSystem":
        """Return a copy of the :py:class:`StaticSystem`."""
        return StaticSystem(
            star=self.star.copy(),
            planets=[planet.copy() for planet in self.planets],
            name=self.name,
            metadata=self.metadata,
        )


# =============================================================================
# FUNCTIONS
# =============================================================================


def _create_static_system(
    star,
    planets,
    name,
    metadata=None,
) -> StaticSystem:
    """Create a :py:class:`StaticSystem` instance.

    Parameters
    ----------
    star : StaticStar
        StaticStar instance.
    planets : list, tuple, StaticPlanet
        List of StaticPlanet instances.
    name : str
        Name of the system.
    metadata : dict, optional. Default: {}.
        Metadata of the dataset.

    Returns
    -------
    StaticSystem
        A new StaticSystem instance.
    """
    if metadata is None:
        metadata = {}

    return StaticSystem(
        star=star,
        planets=planets,
        name=name,
        metadata=metadata,
    )


def _create_static_star(
    star_data,
    source="user",
    metadata=None,
) -> StaticStar:
    """Create a :py:class:`StaticStar` instance.

    Parameters
    ----------
    star_data : pd.Series
        Pandas Series with the star data.
    source : str, optional. Default: 'user'.
        Source of the data.
    metadata : dict, optional. Default: {}.
        Additional metadata about the star.

    Returns
    -------
    StaticStar
        A new StaticStar instance.
    """
    if metadata is None:
        metadata = {}

    return StaticStar(data_df=star_data, source=source, metadata=metadata)


def _create_static_planet(
    planet_data,
    source="user",
    metadata=None,
) -> StaticPlanet:
    """Create a :py:class:`StaticPlanet` instance.

    Parameters
    ----------
    planet_data : pd.Series
        Pandas Series with the planet data.
    source : str, optional. Default: 'user'.
        Source of the data.
    metadata : dict, optional. Default: {}.
        Additional metadata about the planet.

    Returns
    -------
    StaticPlanet
        A new StaticPlanet instance.
    """
    if metadata is None:
        metadata = {}

    return StaticPlanet(data_df=planet_data, source=source, metadata=metadata)


def resokit_to_system(
    resokit_data: ResokitDataFrame,
) -> StaticSystem:
    """Convert a :py:class:`ResokitDataFrame` to a :py:class:`StaticSystem`.

    Parameters
    ----------
    resokit_data : ResokitDataFrame
        ResokitDataFrame instance with the data.

    Returns
    -------
    StaticSystem
        :py:class:`StaticSystem` instance.
    """
    columns = resokit_data.columns_  # Columns of the data

    # Convert to DataFrame
    resokit_df = resokit_data.to_dataframe()

    # Stars
    aux_star_cols = RESO_SR_TYPES.keys() | RESO_OB_TYPES.keys()
    star_cols = list(set(aux_star_cols).intersection(columns))
    star_df = resokit_df[star_cols]

    # Planets
    aux_planet_cols = (
        RESO_PL_TYPES.keys()
        | RESO_OB_TYPES.keys()
        | {"star_name", "n", "n_err_min", "n_err_max"}
    )
    planet_cols = list(set(aux_planet_cols).intersection(columns))
    planet_df = resokit_df[planet_cols]

    # Clean data if more than 1 planet
    if resokit_data.n_objects_ > 1:
        # Assert unique star
        star_names = set(star_df["star_name"])
        if len(star_names) > 1:
            raise ValueError(
                "All planets must have the same star name."
                + f"Found {star_names} instead."
            )

        # Assert no duplicated planets
        planet_names = set(planet_df["name"])
        if len(planet_names) < len(planet_df):
            raise ValueError("Duplicated planet names found.")

        # If multiple lines (i.e. multiple planets), then:
        # Option1: create a star from the star_df line with less
        # null or NaN values
        # if len(star_df) > 1:
        #     star_df = star_df.loc[star_df.notnull().sum(axis=1).idxmax()]

        # Option2: preserve the row with most recent rowupdate column date
        # To get this, check the date from rowupdate column
        # and get the row with the most recent date
        rowupdate = pd.to_datetime(star_df["rowupdate"], errors="coerce")
        star_df = star_df.loc[rowupdate.idxmax()]

    # Redefine star columns to avoid "star_"
    star_df = star_df.rename(lambda x: str(x).replace("star_", ""))
    # EXTRA: Check if the df name is number (idx from db) or a name
    # If it's not a number, then the name is one of the planets names,
    # and we must change it to the star name
    if not str(star_df.name).isnumeric():
        star_df.name = star_df["name"]

    # Create star
    star = _create_static_star(
        star_data=star_df,
        source=resokit_data.source,
        metadata=resokit_data.metadata,
    )

    # Create Planets
    if resokit_data.n_objects_ > 1:  # Multiple planets
        new_metadata = resokit_data.to_dict()
        # Create planets list
        # Create planets list
        planets = [
            _create_static_planet(
                planet_data=planet,
                source=resokit_data.source,
                metadata={
                    **new_metadata,
                    f"{resokit_data.source}_indexes": idx,
                },
            )
            for idx, planet in planet_df.iterrows()
        ]
    else:  # Single planet
        planets = _create_static_planet(
            planet_data=planet_df,
            source=resokit_data.source,
            metadata=resokit_data.metadata,
        )

    return _create_static_system(
        star=star,
        planets=planets,
        name=star.name,
        metadata=resokit_data.metadata,
    )


# =============================================================================
# NEW STATIC CLASSES (TBD)
# =============================================================================


@attrs.define(repr=False, frozen=True, slots=True)
class StaticBinaryStar:
    """StaticBinaryStar class.

    Attributes
    ----------
    star1 : StaticStar
        StaticStar instance for the primary star.
    star2 : StaticStar
        StaticStar instance for the secondary star.
    name : str, optional. Default: 'unnamed'.
        Name of the binary system.
    alternative_name : str, optional. Default: 'unknown'.
        Alternative name of the binary system.
    detection_method : str, optional. Default: 'unknown'.
        Detection method of the binary system.
    distance : float, optional. Default: 0.0.
        Distance to the binary system, in parsecs.
    known_orbit : bool, optional. Default: False.
        Whether the orbit is known.
    a : float, optional. Default: 0.0.
        Semi-major axis of the binary system, in AU.
    e : float, optional. Default: 0.0.
        Eccentricity of the binary system.
    imut : float, optional. Default: 0.0.
        Inclination of the mutual orbit, in degrees.
    n_planets : int, optional. Default: 0.
        Number of planets in the binary system.
    metadata : dict, optional. Default: {}.
        Metadata of the dataset.
    """

    star1: StaticStar = attrs.field(
        validator=attrs.validators.instance_of(StaticStar)
    )
    star2: StaticStar = attrs.field(
        validator=attrs.validators.instance_of(StaticStar)
    )
    name: str = attrs.field(
        validator=attrs.validators.instance_of(str), default="unnamed"
    )
    alternative_name: str = attrs.field(
        validator=attrs.validators.instance_of(str), default="unknown"
    )
    detection_method: str = attrs.field(
        validator=attrs.validators.instance_of(str), default="unknown"
    )
    distance: float = attrs.field(
        validator=attrs.validators.instance_of(float), default=0.0
    )
    known_orbit: bool = attrs.field(
        validator=attrs.validators.instance_of(bool), default=False
    )
    a: float = attrs.field(
        validator=attrs.validators.instance_of(float), default=0.0
    )
    e: float = attrs.field(
        validator=attrs.validators.instance_of(float), default=0.0
    )
    imut: float = attrs.field(
        validator=attrs.validators.instance_of(float), default=0.0
    )
    n_planets: int = attrs.field(
        validator=attrs.validators.instance_of(int), default=0
    )
    metadata: dict = attrs.field(factory=MetaData, converter=MetaData)

    def __attrs_post_init__(self):
        """Post-init method."""
        pass

    def __repr__(self):
        """Return a string representation of the StaticBinaryStar."""
        return (
            f"StaticBinaryStar(star1={self.star1}, star2={self.star2}, "
            + f"name='{self.name}', metadata={self.metadata})"
        )

    def to_dict(self) -> dict:
        """Return the metadata as a new dictionary."""
        return dict(self.metadata)

    def copy(self) -> "StaticBinaryStar":
        """Return a copy of the :py:class:`StaticBinaryStar`."""
        return StaticBinaryStar(
            star1=self.star1.copy(),
            star2=self.star2.copy(),
            name=self.name,
            metadata=self.metadata,
        )


@attrs.define(repr=False, frozen=True, slots=True)
class StaticBinarySystem:
    """StaticBinarySystem class.

    Attributes
    ----------
    binary_star : StaticBinaryStar
        StaticBinaryStar instance for the binary system.
    planets : list, tuple, StaticPlanet
        List of StaticPlanet instances.
    name : str, optional. Default: 'unnamed'.
        Name of the system.
    metadata : dict, optional. Default: {}.
        Metadata of the dataset.
    """

    binary_star: StaticBinaryStar = attrs.field(
        validator=attrs.validators.instance_of(StaticBinaryStar)
    )
    planets: List[StaticPlanet] = attrs.field(
        validator=attrs.validators.deep_iterable(
            member_validator=attrs.validators.instance_of(StaticPlanet)
        )
    )
    name: str = attrs.field(
        validator=attrs.validators.instance_of(str), default="unnamed"
    )
    metadata: dict = attrs.field(factory=MetaData, converter=MetaData)

    def __attrs_post_init__(self):
        """Post-init method."""
        pass

    def __repr__(self):
        """Return a string representation of the StaticBinarySystem."""
        return (
            f"StaticBinarySystem(binary_star={self.binary_star}, "
            + f"planets={self.planets}, "
            + f"name='{self.name}', "
            + f"metadata={self.metadata})"
        )

    def to_dict(self) -> dict:
        """Return the metadata as a new dictionary."""
        return dict(self.metadata)

    def copy(self) -> "StaticBinarySystem":
        """Return a copy of the :py:class:`StaticBinarySystem`."""
        return StaticBinarySystem(
            binary_star=self.binary_star.copy(),
            planets=[planet.copy() for planet in self.planets],
            name=self.name,
            metadata=self.metadata,
        )
