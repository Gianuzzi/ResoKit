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
from typing import Iterable, Union

import attrs

import matplotlib.pyplot as plt

from numpy import isnan, pi, sqrt

import pandas as pd


from resokit.utils.utils import (
    MAPPINGS,
    RESO_DTYPES,
    RESO_OB_TYPES,
    RESO_PL_TYPES,
    RESO_SR_TYPES,
    float_to_fraction,
    parse_to_iter,
)

# =============================================================================
# CONSTANTS
# =============================================================================

# Default attributes for float fields
DEFAULT_FLOAT_ATTRS = {
    "validator": attrs.validators.instance_of((int, float, type(None))),
    "default": None,
}

# =============================================================================
# CLASSES
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
        """Repr(x) <=> x.__repr__()."""
        return f"Metadata({repr(self._data)})"

    def __getitem__(self, k):
        """X[k] <=> x.__getitem__(k)."""
        return self._data[k]

    def __iter__(self):
        """Iter(x) <=> x.__iter__()."""
        return iter(self._data)

    def __len__(self):
        """Len(x) <=> x.__len__()."""
        return len(self._data)

    def __getattr__(self, a):
        """Getattr(x, y) <==> x.__getattr__(y) <==> getattr(x, y)."""
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

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""
        with pd.option_context("display.show_dimensions", False):
            df_body = repr(self.data_df).splitlines()

        if self.n_objects_ > 1:
            sdf_dim = f"{self.n_objects_} rows x {self.n_columns_} columns"
            fotter = f"\nResokitDataFrame - {sdf_dim}"
        else:
            sdf_dim = f"1 row x {self.n_columns_} columns"
            fotter = f"\nResokitSeries - {sdf_dim}"

        resokit_data_repr = "\n".join(df_body + [fotter])

        return resokit_data_repr

    def _repr_html_(self):
        """Return a HTML representation of the DataFrame."""
        ad_id = id(self)

        with pd.option_context("display.show_dimensions", False):
            df_html = self.data_df._repr_html_()

        if self.n_objects_ > 1:
            rows = f"{self.n_objects_} rows"
            columns = f"{self.n_columns_} columns"
            footer = f"ResokitDataFrame - {rows} x {columns}"
        else:
            rows = "1 row"
            columns = f"{self.n_columns_} columns"

        parts = [
            f'<div class="resokit-data-container" id={ad_id}>',
            df_html,
            footer,
            "</div>",
        ]

        html = "".join(parts)

        return html

    def to_dataframe(self, columns=None, copy=False):
        """Return the data_df as a new DataFrame.

        Parameters
        ----------
        columns : list, optional. Default: None.
            Columns to return.
        copy : bool, optional. Default: False.
            Whether to return a copy of the DataFrame.
        """
        if columns is not None:
            used_cols = [col for col in list(columns) if col in self.columns_]
            df = self.data_df[used_cols]
        else:
            df = self.data_df

        return df.copy() if copy else df

    def to_dict(self):
        """Return a copy of the metadata as a dictionary."""
        return dict(self.metadata)

    def plot(
        self,
        x: str,
        y: str,
        error_x: bool = False,
        error_y: bool = False,
        ax: plt.Axes = None,
        label: str = "",
        plot_kwargs: dict = None,
    ):
        """Plot the x vs y data of the ResokitDataFrame.

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
        label : str, optional. Default: "".
            Label for the data plotted.
        plot_kwargs : dict
            Additional keyword arguments for the plot function.

        Returns
        -------
        plt.Axes
            Matplotlib Axes with the plot.
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

        # Check plot_kwargs
        if plot_kwargs is None:
            plot_kwargs = {}

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

    def copy(self):
        """Return a copy of the ResokitDataFrame."""
        return ResokitDataFrame(
            data_df=self.data_df.copy(),
            source=self.source,
            metadata=self.metadata,
        )


def df_to_resokit(
    df: pd.DataFrame,
    source: str,
    drop: bool = True,
    copy: bool = False,
    metadata: dict = None,
) -> ResokitDataFrame:
    """Convert ExoplanetEU or NASA dataset to ResoKit format.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    source : str
        Source of the dataset. Either 'eu' or 'nasa'.
    drop : bool, optional. Default: True.
        Whether to drop columns not in the mapping.
    copy : bool, optional. Default: False.
        Whether to return a copy of the DataFrame.
    metadata : dict, optional. Default: {}.
        Metadata of the dataset.

    Returns
    -------
    ResokitDataFrame
        DataFrame in ResoKit format.
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
        # Sort by period
        df = df.sort_values(by="P", ascending=True)

    # Add metadata
    if metadata is None:
        metadata = {}

    return ResokitDataFrame(data_df=df, source=source, metadata=metadata)


@attrs.define(repr=False, frozen=True, slots=True)
class StaticPlanet(ResokitDataFrame):
    """StaticPlanet class representing a static planet.

    Attributes
    ----------
    data_df : pd.Series
        Series containing the data.
    source : str
        Source of the dataset. Either 'eu' or 'nasa' or 'user'.
    metadata : dict
        Metadata of the dataset.
    name : str
        Name of the planet.
    user_defined_ : bool
        Flag indicating if the planet is user-defined.
    suffix_ : str
        Suffix for the planet name.
    """

    name: str = attrs.field(init=False)
    user_defined_: bool = attrs.field(init=False)
    suffix_: str = attrs.field(init=False)

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
        return self.data_df["name"].split(" ")[-1]

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
        return (
            f"StaticPlanet [{self.data_df['name']}]"
            + f" from {self.source} data source."
            if not self.user_defined_
            else "user defined."
        )

    def get_item(
        self,
        items: Union[list[str], str],
        error: bool = False,
        silent: bool = False,
    ):
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
        pd.Series
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

    def plot(
        self,
        x: str,
        y: str,
        error_x: bool = False,
        error_y: bool = False,
        ax: plt.Axes = None,
        planet_label: bool = True,
        plot_kwargs: dict = None,
    ):
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
        planet_label : bool, optional. Default: True.
            Whether to add a label with the planet name.
        plot_kwargs : dict
            Additional keyword arguments for the plot function.

        Returns
        -------
        plt.Axes
            Matplotlib Axes with the plot.
        """
        return super().plot(
            x=x,
            y=y,
            error_x=error_x,
            error_y=error_y,
            ax=ax,
            label=planet_label,
            plot_kwargs=plot_kwargs,
        )

    def copy(self):
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
    """

    name: str = attrs.field(init=False)
    user_defined_: bool = attrs.field(init=False)

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
        return (
            f"StaticStar [{self.name}]" + f" from {self.source} data source."
            if not self.user_defined_
            else "user defined."
        )

    def plot(
        self,
        x: str,
        y: str,
        error_x: bool = False,
        error_y: bool = False,
        ax: plt.Axes = None,
        star_label: bool = True,
        plot_kwargs: dict = None,
    ):
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
        star_label : bool, optional. Default: True.
            Whether to add a label with the star name.
        plot_kwargs : dict
            Additional keyword arguments for the plot function.

        Returns
        -------
        plt.Axes
            Matplotlib Axes with the plot.
        """
        return super().plot(
            x=x,
            y=y,
            error_x=error_x,
            error_y=error_y,
            ax=ax,
            label=star_label,
            plot_kwargs=plot_kwargs,
        )

    def copy(self):
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
    planets : list, tuple, StaticPlanet
        List of StaticPlanet instances.
    name : str
        Name of the system.
    metadata : dict
        Metadata of the dataset.
    n_planets_ : int
        Number of planets in this static system.
    source_ : str
        Source of the data.
    user_defined_ : bool
        Flag indicating if the system is user-defined.
    planet_names_ : list
        List of planet names.
    period_ratios : float, pd.DataFrame
        Period ratios of the planets.
        Created after calling the period_ratios method.
    """

    star: StaticStar = attrs.field(
        validator=attrs.validators.instance_of(StaticStar),
    )
    planets: Union[list[StaticPlanet], tuple[StaticPlanet], StaticPlanet] = (
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
        indices : int or Iterable[int]
            Indices for slicing planets.

        Returns
        -------
        A copy of an existing StaticPlanet or list of StaticPlanet objects.
        """
        indices = parse_to_iter(indices, to=list)

        if not all(isinstance(i, int) for i in indices):
            raise TypeError("Indices must be integers.")

        if len(indices) == 1:
            return self.planets[indices[0]].copy()

        return [self.planets[i].copy() for i in indices]

    def _get_planets_items(
        self, items: Union[str, list[str]], return_values: bool = True
    ):
        """Retrieve specific attributes of planets.

        Parameters
        ----------
        items : str, list[str]
            Names of planet attributes.
        return_values : bool, default=True
            Whether to return values or full objects.

        Returns
        -------
        list
            Values or full objects of the specified planet attributes.
        """
        data = [planet[items] for planet in self.planets]

        if return_values:
            try:
                return [item.values[0] for item in data]
            except AttributeError:
                pass  # Fall back to full objects

        return [item for item in data]

    def get_item(self, items: Union[str, list[str]], error: bool = False):
        """Retrieve specific attributes of the system (star/planets).

        Parameters
        ----------
        items : str or list[str]
            Names of the desired attributes.
        error : bool, optional. Default: False.
            Whether to return the error columns.
            Only available for standard ResokitDataFrame objects.

        Returns
        -------
        pd.Series or pd.DataFrame
            Series or DataFrame with the requested items.
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

    def _get_single_planet_items(self, items: list[str]):
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

    def _get_multiple_planet_items(self, items: list[str]):
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

    def plot(
        self,
        x: str,
        y: str,
        error_x: bool = False,
        error_y: bool = False,
        ax: plt.Axes = None,
        label: Union[bool, str, Iterable[str]] = True,
        plot_kwargs: dict = None,
    ):
        """Plot the x vs y data of the system. Uses plt.errorbar internally.

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
        label : bool, str, Iterable, optional. Default: True.
            Whether to add a label with the planet (or star) names.
            If str, use the string as the label.
            If Iterable, use the list of strings as the label.
        plot_kwargs : dict
            Additional keyword arguments for the plt.errorbar function.
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
            if (label is True) or isinstance(label, str):
                label = [label] * self.n_planets_  # Anything is True
            elif len(label) != self.n_planets_:
                raise ValueError(
                    "Length of planet_label must be equal "
                    + "to the number of planets."
                )

            # Plot planets
            for i, planet in enumerate(self.planets):
                ax = planet.plot(
                    x,
                    y,
                    error_x,
                    error_y,
                    ax,
                    label[i],
                    plot_kwargs,
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
        label: Union[str, list] = "",
        **kwargs,
    ):
        """Plot each CONSECUTIVE triplet of planets in the period ratio space.

        Systems triplets are shown in the plane P_{i+1}/P_i vs P_{i+2}/P_{i+1}.

        Parameters
        ----------
        which : int, str, optional. Default: 'all'.
            Which triplets to plot.
            If 'all', plot all possible triplets.
            If int, plot the triplet with the given index.
            For example:
            - 0 will plot the first triplet: (0, 1, 2).
            - 1 will plot the second triplet: (1, 2, 3).
        error : bool, optional. Default: False.
            Whether to plot the error bars.
        ax : plt.Axes, optional. Default: None.
            Matplotlib Axes to plot on.
        label : str, list, optional. Default: "".
            Label for the data plotted.
        **kwargs : dict
            Additional keyword arguments for the plt.errorbar function.

        Returns
        -------
        plt.Axes
            Matplotlib Axes with the plot.
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
                        "Length of label must be equal to the number of triplets to plot."
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

    def remove_planet(self, index: Union[int, str], verbose: bool = True):
        """Remove a planet from the system.

        Parameters
        ----------
        index : int, str
            Index or suffix (1 char) or name of the planet to remove.
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
        self, planet: StaticPlanet, sort: bool = True, verbose: bool = True
    ):
        """Add a planet to the system.

        Parameters
        ----------
        planet : StaticPlanet
            StaticPlanet instance to add.
        sort : bool, optional. Default: True.
            Whether to sort the planets by period.

        Returns
        -------
        StaticSystem
            A new StaticSystem instance.
        """
        if not isinstance(planet, StaticPlanet):
            raise TypeError(
                "planet must be a StaticPlanet instance."
                + f" Got: {type(planet)} instead."
            )

        # Create a new list of planets
        new_planets = self.planets + [planet]

        if sort:
            new_planets = sorted(new_planets, key=lambda x: x["P"])

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
    def period_ratios(self):
        """Return the period ratios of all the planets."""
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
        fraction_kwargs: dict = None,
        error: bool = False,
    ) -> Union[float, pd.DataFrame]:
        """Return the period ratio of the specified pair of planets.

        Parameters
        ----------
        pair : list, tuple, str, optional. Default: 'all'.
            Which pair of planets to consider.
            Either 'all' or a list/tuple of planet names/indexes.
            If pair=(i,j), then the period ratio is P_j/P_i, and
            remember that the first planet is 0.
        verbose : bool, optional. Default: False.
            Whether to print the steps of the calculation if a single pair,
            and fraction_arg is not 0.
        fraction_kwargs : dict, optional. Default: {}.
            Keyword arguments for the float_to_fraction function.
            See float_to_fraction for more information.
        error : bool, optional. Default: False.
            Whether to return the error of the period ratio.

        Returns
        -------
        float, pd.DataFrame
            Float with period ratio of the pair of planets, or DataFrame
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

        # Check fraction_kwargs
        if fraction_kwargs is None:
            fraction_kwargs = {}

        # This calculates all the period ratios
        if isinstance(pair, str):

            if not pair == "all":
                raise ValueError("Invalid pair value.")
            if self.n_planets_ == 2:
                return self.period_ratios_  # Already calculated

            if not self.period_ratios_.empty:
                if fraction_kwargs:
                    return self.period_ratios_.map(
                        lambda x: float_to_fraction(
                            x,
                            **fraction_kwargs,
                            verbose=False,
                        )
                    )
                return self.period_ratios

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
                return df.map(
                    lambda x: float_to_fraction(
                        x, **fraction_kwargs, verbose=False
                    )
                )

            return df

        # This is sigle pair

        if fraction_kwargs:
            fraction_kwargs["verbose"] = verbose

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

    def to_dataframe(self, columns: list = None) -> pd.DataFrame:
        """Return data_df as a new DataFrame.

        Parameters
        ----------
        columns : list, optional. Default: None.
            Columns to return.
        """
        # Create a DataFrame with the planets data
        df = pd.DataFrame()
        for planet in self.planets:
            df = df.append(planet.data_df)

        # Add star data
        df = pd.concat([self.star.data_df, df], axis=0)

        if columns is not None:
            used_cols = [col for col in columns if col in df.columns]
            df = df[used_cols]

        return df

    def to_dict(self) -> dict:
        """Return the metadata as a new dictionary."""
        return dict(self.metadata)

    def copy(self):
        """Return a copy of the StaticSystem."""
        return StaticSystem(
            star=self.star.copy(),
            planets=[planet.copy() for planet in self.planets],
            name=self.name,
            metadata=self.metadata,
        )


# =============================================================================
# NEW FUNCTIONS
# =============================================================================


def _create_static_system(
    star,
    planets,
    name,
    metadata=None,
) -> StaticSystem:
    """Create a StaticSystem instance.

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
    """Create a StaticStar instance.

    Parameters
    ----------
    star_data : pd.Series
        Series with the star data.
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
    """Create a StaticPlanet instance.

    Parameters
    ----------
    planet_data : pd.Series
        Series with the planet data.
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
    """Convert a ResokitDataFrame to a StaticSystem instance.

    Parameters
    ----------
    resokit_data : ResokitDataFrame
        ResokitDataFrame instance.

    Returns
    -------
    StaticSystem
        StaticSystem instance.
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

    # Create star
    star = _create_static_star(
        star_data=star_df,
        source=resokit_data.source,
        metadata=resokit_data.metadata,
    )

    # Create Planets
    if resokit_data.n_objects_ > 1:  # Multiple planets
        planets = [
            _create_static_planet(
                planet_data=planet,
                source=resokit_data.source,
                metadata=resokit_data.metadata,
            )
            for _, planet in planet_df.iterrows()
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
