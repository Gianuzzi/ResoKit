#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2024, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

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
from numpy.random import default_rng

import pandas as pd

from resokit.units import (
    AU,
    DEG2RAD,
    M_JUP,
    M_SUN,
    Me2Mj,
    Mj2Me,
    Mj2Ms,
    R_JUP,
    R_SUN,
    Re2Rj,
    Rj2Re,
)
from resokit.utils.mass_radius import estimate_mass, estimate_radius
from resokit.utils.parser import (
    DEFAULT_METADATA,
    MAPPINGS,
    RESO_DTYPES,
    RESO_OB_TYPES,
    RESO_PL_TYPES,
    RESO_SR_TYPES,
    assert_module_imported,
    parse_name,
    parse_to_iter,
)
from resokit.utils.utils import (
    calc_a_with_errors,
    calc_hill_radius_with_errors,
    calc_period_with_errors,
    calc_sum_with_errors,
    float_to_fraction,
)

try:
    from rebound import Simulation

    rebound_imported = True
except ImportError:
    rebound_imported = False
    Simulation = None

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


@attrs.define(on_setattr=attrs.setters.frozen, slots=True, repr=False)
class ResokitDataFrame:
    """Initialize a ResoKit DataFrame class.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        DataFrame containing the data.
    source : str
        Source of the dataset. Either 'eu' or 'nasa' or 'binary' or 'user'.
    metadata : dict
        Metadata of the dataset.
    """

    data: Union[pd.DataFrame, pd.Series] = attrs.field(
        validator=attrs.validators.instance_of((pd.DataFrame, pd.Series)),
        converter=lambda df: df.squeeze(),  # Convert to Series if possible
    )
    source: str = attrs.field(
        validator=attrs.validators.in_({"eu", "nasa", "binary", "user"}),
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
            self.data.index
            if isinstance(self.data, pd.Series)
            else self.data.columns
        ).to_list()

    @n_columns_.default
    def _n_columns__default(self) -> int:
        """Set the default value for n_columns_."""
        return len(self.columns_)

    @n_objects_.default
    def _n_objects__default(self) -> int:
        """Set the default value for n_objects_."""
        return self.data.shape[0] if isinstance(self.data, pd.DataFrame) else 1

    def __attrs_pre_init__(self):
        """Pre-initialization hook."""
        pass

    def __attrs_post_init__(self):
        """Post-initialization hook."""
        if self.data.empty:
            warnings.warn("Empty DataFrame.", stacklevel=2)

        if "name" not in self.columns_:
            warnings.warn(
                "Missing 'name' column in the DataFrame.",
                stacklevel=2,
            )

        if self.n_objects_ == 1 and not isinstance(self.data, pd.Series):
            raise TypeError(
                "With only one object, data must be a Series "
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
                return self.data.iloc[key]

            if isinstance(key, list):
                if all(isinstance(i, int) for i in key):
                    return self.data.iloc[key]

            return self.data[key]

        return self.data.__getitem__(key)

    def __dir__(self):
        """dir(pdf) <==> pdf.__dir__()."""
        return super().__dir__() + dir(self.data)

    def __getattr__(self, a):
        """getattr(x, y) <==> x.__getattr__(y) <==> getattr(x, y)."""
        return getattr(self.data, a)  # Get attribute from data

    def __repr__(self, prefoot=None):
        """repr(x) <=> x.__repr__()."""
        with pd.option_context("display.show_dimensions", False):
            df_body = repr(self.data).splitlines()

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
                df_html = self.data._repr_html_()
            else:  # It is a Series
                df_html = self.data.to_frame()._repr_html_()

        parts = [
            f'<div class="resokit-data-container" id={ad_id}>',
            df_html,
            footer,
            "</div>",
        ]

        html = "".join(parts)

        return html

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

    def to_dataframe(self, columns=None, copy=False) -> pd.DataFrame:
        """Convert data to pandas data frame.

        This method constructs a data frame with the data inside the
        data attribute.

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

        df = self.data[used_cols]

        return df.copy(deep=True) if copy else df

    def copy(self) -> "ResokitDataFrame":
        """Create and return copy of the :py:class:`ResokitDataFrame`.

        Returns
        -------
        ResokitDataFrame
            Copy of the ResokitDataFrame.
        """
        return attrs.evolve(self)


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
        df["n"] = 2.0 * pi / df["P"]
        if "P_err_min" in df.columns and "P_err_max" in df.columns:
            df["n_err_min"] = 2.0 * pi / df["P_err_max"]
            df["n_err_max"] = 2.0 * pi / df["P_err_min"]

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

    return ResokitDataFrame(data=df, source=source, metadata=metadata)


rng = default_rng(seed=42)


# =============================================================================
# VALIDATORS
# =============================================================================


def static_binary_star_data_validator(
    instance, attribute, data: Union[pd.Series, pd.DataFrame]
):
    """Validate the data for a StaticBinaryStar."""
    # Must be Series or Dataframe
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(
            "StaticBinaryStar must have a pd.Series or pd.DataFrame. "
            + f"Got: {type(data)} instead."
        )

    # Must have a, e, and P
    if not all(col in data.index for col in ["a", "e", "P"]):
        raise ValueError(
            "StaticBinaryStar must have 'a', 'e', and 'P' columns."
        )
    return True


# =============================================================================
# STATIC CLASSES
# =============================================================================


@attrs.define(repr=False, on_setattr=attrs.setters.frozen, slots=True)
class StaticBody(ResokitDataFrame):
    """StaticBody class.

    This class defines the basic object structure; for
    a star or a planet. This class should not be instantiated
    directly.

    Attributes
    ----------
    data : pd.Series
        Series containing the data.
        Inherited from ResokitDataFrame.
    source : str
        Source of the dataset.
        Either 'eu' or 'nasa' or 'binary' or 'user'.
        Inherited from ResokitDataFrame.
    metadata : dict
        Metadata of the dataset.
        Inherited from ResokitDataFrame.
    is_star : bool
        Flag indicating if the object is a star.
    name : str
        Name of the object.
    web_page : str
        Web page of the object.
    """

    is_star: bool = attrs.field(
        init=True,
        kw_only=True,
    )  # Must be set in the subclass

    name: str = attrs.field(init=False)
    suffix_: str = attrs.field(init=False)
    web_page: str = attrs.field(init=False)
    user_defined_: bool = attrs.field(init=False)

    @name.default
    def _name_default(self):
        """Set the default value for name."""
        if self.is_star and "star_name" in self.data.index:
            return self.data["star_name"]
        return self.data["name"]  # ["name"] because .name is a df method

    @suffix_.default
    def _suffix_default(self):
        """Set the default value for suffix_."""
        aux = self.data["name"].split(" ")[-1]
        if len(aux) == 1:
            return aux
        elif len(aux) == 2 and aux[0] in ["A", "B"]:
            return aux[1]
        if self.is_star:  # No suffix if star then
            return ""
        aux = self.data["name"].split(")")[-1]
        # Check if the suffix is a letter
        if len(aux) == 1:
            return aux
        return ""

    @web_page.default
    def _web_page_default(self):
        """Set the default value for web_page."""
        if not self.is_star and self.source == "eu":
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

    @user_defined_.default
    def _user_defined_default(self):
        """Set the default value for user_defined_."""
        return self.source not in ["eu", "nasa"]

    def __attrs_pre_init__(self):
        """Pre-initialization hook."""
        pass

    def __attrs_post_init__(self):
        """Post-initialization hook."""
        # Assert data is a Series
        if not isinstance(self.data, pd.Series):
            raise TypeError(
                "StaticBody must have a pd.Series. "
                + f"Got: {type(self.data)} instead."
            )

    def __dir__(self):
        """Return the attributes of both the superclass and this instance."""
        instance_attrs = list(self.__dict__.keys())  # Get self attributes
        parent_attrs = super().__dir__()  # Get superclass attributes
        return sorted(set(instance_attrs + parent_attrs))  # Remove duplicates

    def __repr__(self, cls_name: str = "StaticBody"):
        """repr(x) <=> x.__repr__()."""
        text = f"{cls_name} '{self.name}'"
        if not self.user_defined_:
            text += f" from {self.source} data source"
        else:
            text += " user defined"

        return text

    def _repr_html_(self, ad_id=None, switch=False, cls_name="StaticBody"):
        """Return a HTML representation of the StaticBody."""
        if ad_id is None:
            ad_id = id(self)
        prefoot = f"{cls_name} '{self.name}''"
        if not self.user_defined_:
            prefoot += f" from {self.source} data source"
        else:
            prefoot += " user defined"

        return super()._repr_html_(ad_id=ad_id, prefoot=prefoot, switch=switch)

    def __getitem__(self, key: Union[int, str, list]):
        """x[y] <==> x.__getitem__(y)."""
        key = parse_to_iter(key)

        if any(isinstance(i, int) for i in key):
            raise IndexError(
                "StaticBody does not support integer indexing. "
                + "Use the 'name' column instead."
            )

        if len(key) == 1:
            return self.data[key[0]]

        return self.data[key]

    def change_value(self, attr: str, value: any) -> "StaticBody":
        """Set an attribute of the StaticBody and return a new instance.

        Parameters
        ----------
        attr : str
            Attribute to set.
        value : any
            Value to set.

        Returns
        -------
        StaticBody
            A new StaticBody with the updated attribute.
        """
        # Can not change is_star
        if attr == "is_star":
            raise ValueError("Cannot change 'is_star' attribute.")

        # Modify the metadata
        new_metadata = dict(self.metadata)
        msg = f"Changed {attr} to {value}."
        new_metadata["history"] = new_metadata.get("history", []) + [msg]

        # If name, then change data.name too
        if attr == "name":
            new_data = self.data.copy()
            new_data.name = value
            new_data["name"] = value
            return attrs.evolve(
                self,
                data=new_data,
                source="user",
                metadata=new_metadata,
            )
        else:
            # Check if the attribute is in the data
            if attr not in self.data.index:
                return attrs.evolve(self, **{attr: value})

        # Copy and modify the data dictionary directly
        new_data = self.data.copy()
        new_data[attr] = value

        return attrs.evolve(
            self, data=new_data, metadata=new_metadata, source="user"
        )


@attrs.define(repr=False, on_setattr=attrs.setters.frozen, slots=True)
class StaticPlanet(StaticBody):
    """StaticPlanet class representing a static planet.

    Attributes
    ----------
    data : pd.Series
        Pandas Series containing the data.
    source : str
        Source of the dataset.
        Either 'eu' or 'nasa' or 'user'.
    metadata : dict
        Metadata of the dataset.
    name : str
        Name of the planet.
    web_page : str
        Web page of the planet.
    user_defined_ : bool
        Flag indicating if the planet is user-defined.
    suffix_ : str
        Suffix for the planet name.
    """

    is_star: bool = attrs.field(init=False, on_setattr=attrs.setters.NO_OP)

    def __attrs_pre_init__(self):
        """Pre-initialization hook."""
        # Set is_star to False
        self.is_star = False

    def __attrs_post_init__(self):
        """Post-initialization hook."""
        # Check if all columns are in the default mapping
        if not self.user_defined_:
            for col in self.data.index:
                if col not in RESO_PL_TYPES.keys() | RESO_OB_TYPES.keys() | {
                    "star_name",
                    "n",
                    "n_err_min",
                    "n_err_max",
                    "binary",
                }:
                    warnings.warn(
                        "Found columns not in the default planet mapping.",
                        stacklevel=2,
                    )

    def _repr_html_(self):
        """Return a HTML representation of the StaticPlanet."""
        return super()._repr_html_(ad_id=id(self), cls_name="StaticPlanet")

    def __dir__(self):
        """Return the attributes of both the superclass and this instance."""
        instance_attrs = list(self.__dict__.keys())  # Get self attributes
        parent_attrs = super().__dir__()  # Get superclass attributes
        return sorted(set(instance_attrs + parent_attrs))  # Remove duplicates

    def __repr__(self, cls_name="StaticPlanet"):
        """repr(x) <=> x.__repr__()."""
        return super().__repr__(cls_name)

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
            :math:`mass = \dfrac{1}{C} \times radius^{1/S}`

        Note
        ----
        To use the errors, set `err_method` to 1 or 2. Be aware that the
        planet must have `radius_err_min` and `radius_err_max` columns.

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
            :math:`radius = C \times mass^S`

        Note
        ----
        To use the errors, set `err_method` to 1 or 2. Be aware that the
        planet must have `mass_err_min` and `mass_err_max` columns.

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

        Note
        ----
        The parameters `error_x` and `error_y` link each error to the
        input parameter `x` and `y`, respectively. The error columns must be
        named as `x_err_min`, `x_err_max`, `y_err_min`, and `y_err_max`.

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


@attrs.define(repr=False, on_setattr=attrs.setters.frozen, slots=True)
class StaticStar(StaticBody):
    """StaticStar class representing a static star.

    Attributes
    ----------
    data : pd.Series
        Series containing the data.
    source : str
        Source of the dataset. Either 'eu' or 'nasa' or 'binary' or 'user'.
    metadata : dict
        Metadata of the dataset.
    name : str
        Name of the star.
    web_page : str
        Web page of the star.
    user_defined_ : bool
        Flag indicating if the star is user-defined.
    """

    is_star: bool = attrs.field(init=False, on_setattr=attrs.setters.NO_OP)

    def __attrs_pre_init__(self):
        """Pre-initialization hook."""
        # Set is_star to True
        self.is_star = True

    def __attrs_post_init__(self):
        """Post-initialization hook."""
        # Check if all columns are in the default mapping
        if not self.user_defined_:
            aux_cols = {
                col.replace("star_", "") for col in RESO_SR_TYPES.keys()
            }
            for col in self.data.index:
                if (
                    col not in aux_cols | RESO_OB_TYPES.keys()
                    and self.source != "binary"
                ):
                    warnings.warn(
                        "Found columns not in the default star mapping.",
                        stacklevel=2,
                    )
                    print(col)

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""
        return super().__repr__(cls_name="StaticStar")

    def _repr_html_(self):
        """Return a HTML representation of the StaticStar."""
        return super()._repr_html_(ad_id=id(self), cls_name="StaticStar")

    def __dir__(self):
        """Return the attributes of both the superclass and this instance."""
        instance_attrs = list(self.__dict__.keys())  # Get self attributes
        parent_attrs = super().__dir__()  # Get superclass attributes
        return sorted(set(instance_attrs + parent_attrs))  # Remove duplicates

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


# --------------------------- System of bodies --------------------------------


@attrs.define(repr=False, on_setattr=attrs.setters.frozen, slots=True)
class StaticBinaryStar:
    """StaticBinaryStar class.

    Attributes
    ----------
    star0 : StaticStar. Mandatory.
        StaticStar instance for the primary star.
    star1 : StaticStar. Mandatory.
        StaticStar instance for the secondary star.
    data : Union[pd.DataFrame, pd.Series]. Mandatory.
        Data of the binary system.
    name : str, optional. Default: 'unnamed'.
        Name of the binary system.
    metadata : dict, optional. Default: {}.
        Metadata of the dataset.
    web_page : str
        Web page of the binary system.
    suffix_ : str
        Suffix for the binary system name.
    alternate_name : str
        Alternative name of the binary system.
    disc_method : str
        Detection method of the binary system.
    dist : float
        Distance to the binary system, in parsecs.
    a : float
        Semi-major axis of the binary system, in AU.
    e : float
        Eccentricity of the binary system.
    imut : float
        Inclination of the mutual orbit, in degrees.
    nplanets : int
        Number of planets in the binary system.
    planet_HW_crit : float
        Hill-Waterworth criterion for the binary system.
    total_mass_ : float
        Total mass of the binary system, in solar masses.
    known_orbit_ : bool
        Whether the orbit is known.
    """

    star0: StaticStar = attrs.field(
        validator=attrs.validators.instance_of(StaticStar)
    )
    star1: StaticStar = attrs.field(
        validator=attrs.validators.instance_of(StaticStar)
    )
    data: Union[pd.DataFrame, pd.Series] = attrs.field(
        validator=static_binary_star_data_validator,
        converter=lambda df: df.squeeze(),  # Convert to Series if possible
    )

    name: str = attrs.field(
        validator=attrs.validators.instance_of(str), default="unnamed"
    )

    metadata: dict = attrs.field(factory=MetaData, converter=MetaData)

    web_page: str = attrs.field(init=False)

    source_: str = attrs.field(init=False)
    user_defined_: bool = attrs.field(init=False)

    suffix_: str = attrs.field(init=False)

    is_star: bool = attrs.field(init=False, on_setattr=attrs.setters.NO_OP)

    total_mass_: float = attrs.field(init=False)

    known_orbit_: bool = attrs.field(init=False)

    @web_page.default
    def _web_page_default(self):
        """Set the default value for web_page."""
        return [self.star0.web_page, self.star1.web_page]

    @source_.default
    def _source__default(self):
        """Set the default value for source_."""
        main_source = self.star0.source

        return main_source if main_source == self.star1.source else "user"

    @user_defined_.default
    def _user_defined__default(self):
        """Set the default value for user_defined_."""
        return self.source_ != "binary"

    @suffix_.default
    def _suffix__default(self):
        """Set the default value for suffix_."""
        return [self.star0.suffix_, self.star1.suffix_]

    @total_mass_.default
    def _total_mass__default(self):
        """Set the default value for total_mass_."""
        return self.star0.mass + self.star1.mass

    @known_orbit_.default
    def _known_orbit__default(self):
        """Set the default value for known_orbit_."""
        return self.data["e"] < 1.0

    def __attrs_pre_init__(self):
        """Pre-init method."""
        # Set is_star to True
        self.is_star = True

    def __attrs_post_init__(self):
        """Post-init method."""
        pass

    def __len__(self):
        """len(x) <=> x.__len__()."""
        return 2

    def __getitem__(self, key):
        """x[y] <==> x.__getitem__(y)."""
        return self.data.__getitem__(key)

    def __dir__(self):
        """dir(pdf) <==> pdf.__dir__()."""
        return super().__dir__() + dir(self.data)

    def __getattr__(self, a):
        """getattr(x, y) <==> x.__getattr__(y) <==> getattr(x, y)."""
        return getattr(self.data, a)

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""
        star0 = f"\n Star 1:\n  {self.star0.name}"
        star1 = f"\n Star 2:\n  {self.star1.name}"

        return (
            f"StaticBinaryStar: '{self.name}'"
            + f"{star0} "
            + f"{star1}"
            + "\n"
            + " from binary data source."
            if not self.user_defined_
            else ""
        )

    def _repr_html_(self):
        """Return a HTML representation of the DataFrame."""
        if self.data.empty:
            return self.__repr__()

        ad_id = id(self)

        header = (
            f" StaticBinaryStar: '{self.name}'"
            + f"[{self.star0.name} - {self.star1.name}]"
        )
        footer = "Binary data"
        if not self.user_defined_:
            footer += " from binary data source."

        with pd.option_context("display.show_dimensions", False):
            df_html = self.data.to_frame()._repr_html_()

        parts = [
            f'<div class="resokit-data-container" id={ad_id}>',
            header,
            df_html,
            footer,
            "</div>",
        ]

        html = "".join(parts)

        return html

    def __iter__(self):
        """Iterate over the stars."""
        return iter([self.star0, self.star1])

    def star(
        self, indices: Union[int, str, List[Union[int, str]]]
    ) -> StaticStar:
        """Return the star(s) of the binary system, by given index or name.

        Parameters
        ----------
        indices : int, str, list
            Which star(s) to return.
            If 'all', return both stars.
            If an :py:class:`int`, return the star with the given index.
            If a :py:class:`str`, return the star with the given name.
            If a list, return the stars with the given indexes or names.

        Returns
        -------
        star : StaticStar, list
            StaticStar instance for the star, or list of stars.
        """
        # Check if indices is int
        if isinstance(indices, int):
            if indices == 0:
                return self.star0
            if indices == 1:
                return self.star1
            raise ValueError(
                "Invalid int value for 'indices'. Must be 0 or 1."
            )
        # Check if indices is str
        elif isinstance(indices, str):
            if indices == "all":
                return [self.star0, self.star1]
            elif indices == self.star0.name:
                return self.star0
            elif indices == self.star1.name:
                return self.star1
            raise ValueError("Invalid str value for 'indices'.")

        # Parse indices to list
        indices = parse_to_iter(indices)

        # Check if indices are int or str
        if not all(isinstance(i, (int, str)) for i in indices):
            raise TypeError("Indices must be integers or strings.")

        # Check not "all" inside
        if "all" in indices:
            raise ValueError("Cannot mix 'all' with other indices.")

        return [self.star(idx) for idx in indices]

    def change_value(
        self, attr: str, value: any, in_star: Union[None, int] = None
    ) -> "StaticBinaryStar":
        """Set an attribute of the StaticBinaryStar and return a new instance.

        Parameters
        ----------
        attr : str
            Attribute to set.
        value : any
            Value to set.
        in_star : int, optional (default: None)
            Index of the star to modify (0 or 1).
            If None, modifies the binary system.

        Returns
        -------
        StaticBinaryStar
            A new instance with the updated attribute.
        """
        # Check if in_star is None
        if in_star is None:
            return self._change_binary_attribute(attr, value)
        elif in_star == 0:
            return self._change_star_attribute(0, attr, value)
        elif in_star == 1:
            return self._change_star_attribute(1, attr, value)

        raise ValueError("Invalid value for 'in_star'. Must be 0, 1, or None.")

    def _change_binary_attribute(
        self, attr: str, value: any
    ) -> "StaticBinaryStar":
        """Modify an attribute of the binary system (not an individual star)."""
        # Check if the attribute is in the data
        if attr not in self.data.index:
            return attrs.evolve(self, **{attr: value})

        # Modify the data dictionary
        new_data = self.data.copy()
        new_data[attr] = value

        # Modify the metadata
        new_metadata = dict(self.metadata)
        msg = f"Changed {attr} to {value}."
        new_metadata["history"] = new_metadata.get("history", []) + [msg]

        return attrs.evolve(self, data=new_data, metadata=new_metadata)

    def _change_star_attribute(
        self, star_index: int, attr: str, value: any
    ) -> "StaticBinaryStar":
        """Modify an attribute of one of the stars (star0 or star1)."""
        # Select and modify the star
        stars = [self.star0, self.star1]
        stars[star_index] = stars[star_index].change_value(attr, value)

        # Also, change the star source
        stars[star_index] = stars[star_index].change_value("source", "user")

        # And modify the metadata too
        new_metadata = dict(stars[star_index].metadata)
        msg = f"Changed {attr} to {value}."
        new_metadata["history"] = new_metadata.get("history", []) + [msg]
        stars[star_index] = stars[star_index].change_value(
            "metadata", new_metadata
        )

        # Also, the binary metadata
        new_metadata = dict(self.metadata)
        # Add a message in "notes" if not already there
        msg = f" Changed {attr} in star {star_index} to {value}."
        new_metadata["history"] = new_metadata.get("history", []) + [msg]

        return attrs.evolve(
            self, star0=stars[0], star1=stars[1], metadata=new_metadata
        )

    def to_dict(self) -> dict:
        """Return the metadata as a new dictionary."""
        return dict(self.metadata)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the binary data as a new DataFrame."""
        return self.data.to_frame(name=self.name)

    def copy(self) -> "StaticBinaryStar":
        """Return a copy of the :py:class:`StaticBinaryStar`."""
        return attrs.evolve(self)


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

    star: Union[StaticStar, StaticBinaryStar] = attrs.field(
        validator=attrs.validators.instance_of((StaticStar, StaticBinaryStar))
    )
    planets: Union[List[StaticPlanet], Tuple[StaticPlanet], StaticPlanet] = (
        attrs.field(
            validator=attrs.validators.instance_of(
                (list, tuple, StaticPlanet)
            ),
            converter=parse_to_iter,
        )
    )

    name: str = attrs.field(
        validator=attrs.validators.instance_of(str), default="unnamed"
    )

    metadata: dict = attrs.field(factory=MetaData, converter=MetaData)

    is_circumbinary: bool = attrs.field(
        validator=attrs.validators.instance_of(bool), default=False
    )  # This is for the user

    is_binary_: bool = attrs.field(init=False)  # This is for the code

    bodies_: List[Union[StaticStar, StaticPlanet]] = attrs.field(init=False)

    web_page: list = attrs.field(init=False)

    n_stars_: int = attrs.field(init=False)
    n_planets_: int = attrs.field(init=False)

    source_: str = attrs.field(init=False)
    user_defined_: bool = attrs.field(init=False)

    star_names_: list = attrs.field(init=False)
    planet_names_: list = attrs.field(init=False)
    suffixes_: list = attrs.field(init=False)

    period_ratios_: Union[float, pd.DataFrame] = attrs.field(init=False)
    __error_ratios__: Union[float, pd.DataFrame] = attrs.field(init=False)

    mass_accum_: pd.Series = attrs.field(init=False)
    total_mass_: float = attrs.field(init=False)

    @is_binary_.default
    def _is_binary__default(self):
        """Set the default value for is_binary_."""
        return isinstance(self.star, StaticBinaryStar)

    @bodies_.default
    def _bodies__default(self):
        """Set the default value for bodies_."""
        if not self.is_binary_:
            return [self.star, *self.planets]
        if self.is_circumbinary:
            return [self.star.star0, self.star.star1, *self.planets]
        return [self.star.star0, *self.planets, self.star.star1]

    @web_page.default
    def _web_page_default(self):
        """Set the default value for web_page."""
        return [body.web_page for body in self.bodies_]

    @n_stars_.default
    def _n_stars__default(self):
        """Set the default value for n_stars_."""
        return 2 if self.is_binary_ else 1

    @n_planets_.default
    def _n_planets__default(self):
        """Set the default value for n_planets_."""
        return len(self.planets)

    @source_.default
    def _source__default(self):
        """Set the default value for source_."""
        # Check if not binary
        if not self.is_binary_:
            main_source = self.star.source
        else:
            if self.star.star0.source == self.star.star1.source == "binary":
                main_source = "eu"
            else:
                main_source = "user"

        return (
            main_source
            if all([planet.source == main_source for planet in self.planets])
            else "user"
        )

    @user_defined_.default
    def _user_defined__default(self):
        """Set the default value for user_defined_."""
        return self.source_ not in ["eu", "nasa"]

    @star_names_.default
    def _star_names__default(self):
        """Set the default value for star_names_."""
        if not self.is_binary_:
            return self.star.name
        return [self.star.star0.name, self.star.star1.name]

    @planet_names_.default
    def _planet_names__default(self):
        """Set the default value for planet_names_."""
        return [planet.name for planet in self.planets]

    @suffixes_.default
    def _suffixes__default(self):
        """Set the default value for suffixes_."""
        return [body.suffix_ for body in self.bodies_]

    @period_ratios_.default
    def _period_ratios__default(self):
        """Set the default value for period_ratios_."""
        # If binary system...
        if self.is_binary_:
            bin_per = self.star.data.P
            if self.n_planets_ == 1 and self.is_circumbinary:  # Circumbinary
                return self.planets[0].P / bin_per
            elif self.n_planets_ == 1 and not self.is_circumbinary:
                return bin_per / self.planets[0].P
            return self.pair_ratio(verbose=False, use_binary=True)

        # If not a binary system...
        if self.n_planets_ == 1:  # Single planet
            return None
        elif self.n_planets_ == 2:  # Two planets
            return self.planets[1].P / self.planets[0].P

        return self.pair_ratio(verbose=False)  # More than two planets

    @__error_ratios__.default
    def ___error_ratios__default(self):
        """Set the default value for __error_ratios__."""
        # If binary system...
        if self.is_binary_:
            # Binary has no error
            if self.n_planets_ == 1:  # Return None
                return None
            return pd.DataFrame()  # Empty mutable DataFrame

        # If not a binary system...
        if self.n_planets_ == 1:  # Single planet
            return None
        elif self.n_planets_ == 2:  # Two planets
            error_0 = max(self.planets[0].P_err_min, self.planets[0].P_err_max)
            error_1 = max(self.planets[1].P_err_min, self.planets[1].P_err_max)
            return self.period_ratios_ * sqrt(
                (error_0 / self.planets[0].P) ** 2
                + (error_1 / self.planets[1].P) ** 2
            )

        return pd.DataFrame()  # Empty mutable DataFrame

    @mass_accum_.default
    def _mass_accum__default(self):
        """Set the default value for mass_accum_."""
        # Calculate the accumulated mass inside each planet orbit
        # (and the star(s) too).
        # The series will have n_stars_ + n_planets_ elements.
        in_masses = [self.bodies_[0].mass]
        for i in range(1, self.n_stars_ + self.n_planets_):
            in_masses.append(in_masses[i - 1] + self.bodies_[i].mass)

        return pd.Series(
            in_masses,
            index=[body.name for body in self.bodies_],
            name="mass_accum",
        )

    @total_mass_.default
    def _total_mass__default(self):
        """Set the default value for total_mass_."""
        return self.mass_accum_.iloc[-1]

    def __attrs_pre_init__(self):
        """Pre-initialization hook."""
        pass

    def __attrs_post_init__(self):
        """Post-initialization hook."""
        parsed_star_names = [parse_name(self.star.name)]
        # Check if not binary system
        if self.is_binary_:
            parsed_star_names += [parse_name(star.name) for star in self.star]

        # Check if all planets have the same star name
        for planet in self.planets:
            if parse_name(planet.star_name) not in parsed_star_names:
                warnings.warn(
                    f"Planet({planet.name}) parsed "
                    + f"star name({planet.star_name})"
                    + f" is not in the parsed star names({parsed_star_names}).",
                    stacklevel=2,
                )

        # Check if all planets have unique names
        if self.n_planets_ != len(set(self.planet_names_)):
            warnings.warn("Planets must have unique names.", stacklevel=2)

        return

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""
        # Planets message
        planets_msg = (
            "\n"
            + f" Planet{'s' if self.n_planets_ > 1 else ''}:"
            + "\n  "
            + "\n  ".join(self.planet_names_)
        )

        # Star message
        if not self.is_binary_:  # Single star
            star_msg1 = "\n Star:\n  " + f"{self.star.name}"
            star_msg2 = ""
        else:  # Binary star
            star_msg1 = "\n Star 1:\n  " + f"{self.star.star0.name}"
            if self.is_circumbinary:  # Circumbinary
                star_msg1 += "\n Star 2:\n  " + f"{self.star.star1.name}"
                star_msg2 = "\n [circumbinary system]"
            else:  # Normal binary
                star_msg2 = (
                    "\n Star 2:\n  "
                    + f"{self.star.star1.name}"
                    + "\n [binary system]"
                )

        return (
            f"StaticSystem: '{self.name}'"
            + f"{star_msg1} "
            + f"{planets_msg}"
            + f"{star_msg2}"
            + (
                f"\nfrom '{self.source_}' data source."
                if not self.user_defined_
                else ""
            )
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
        return self.n_stars_ + self.n_planets_

    def __iter__(self):
        """Iterate over the stars and planets."""
        return iter(self.bodies_)

    def __contains__(self, item: Union[str, StaticStar, StaticPlanet]) -> bool:
        """Check if the item is in the system."""
        # Check by name
        if isinstance(item, str):
            if self.is_binary_ and item == self.star.name:  # If binary
                return True
            return item in self.star_names_ or item in self.planet_names_
        # Check by object (star)
        if isinstance(item, StaticStar) and not self.is_binary_:
            return item == self.star
        # Check by object (binary star)
        if isinstance(item, StaticBinaryStar) and self.is_binary_:
            return item == self.star
        # Check by object (planet)
        if isinstance(item, StaticPlanet):
            return item in self.planets
        return False

    def body(
        self,
        indices: Union[int, str, Iterable[Union[int, str]]],
        only_index: bool = False,
    ) -> Union[
        StaticStar,
        StaticPlanet,
        StaticBinaryStar,
        List[Union[StaticStar, StaticPlanet, StaticBinaryStar]],
    ]:
        """Return the star (or binary) and/or planets by given indices.

        For this function, the star (or binary) is index 0,
        and the planets starts from 1.

        Parameters
        ----------
        indices : int, Iterable[int], Iterable[str], str
            Indices for slicing star and planets.
            If "all", return all bodies.
            Instead of index numbers, the bodies names can be used.
        only_index : bool, default=False
            Return only the index of the bodies.

        Returns
        -------
        bodies : StaticStar or StaticPlanet or StaticBinaryStar or list
            A copy of a system's star :py:class:`StaticStar` or
            planet :py:class:`StaticPlanet` or
            binary star :py:class:`StaticBinaryStar` or list of them.
        """
        # Define possible extra in in case circum-binary
        extra = 2 if self.is_binary_ and self.is_circumbinary else 1
        # Check if indices is an integer
        if isinstance(indices, int):
            if not 0 <= indices < self.n_planets_ + extra:
                raise IndexError(
                    "Index out of range. "
                    + f"Expected: 0 to {self.n_planets_ + extra - 1}. "
                    + f"Got: {indices}."
                )
            return indices if only_index else self.bodies_[indices]
        # Check if indices is a string
        elif isinstance(indices, str):
            if indices == "all":  # All bodies
                if only_index:
                    return list(range(self.n_stars_ + self.n_planets_))
                return [self.star] + self.planets
            if indices == "star" or indices == self.star.name:  # Star
                return self.star
            if self.is_binary_:
                if indices == self.star.star0.name or indices == "star0":
                    return self.star.star0
                if indices == self.star.star1.name or indices == "star1":
                    return self.star.star1
            if indices in self.planet_names_:  # Planet name
                if only_index:
                    return self.planet_names_.index(indices)
                return self.planets[self.planet_names_.index(indices)]
            if len(indices) == 1:  # Check suffixes
                # Assert no repeated suffixes
                if len(self.suffixes_) != len(set(self.suffixes_)):
                    raise ValueError(
                        "Cannot slice by suffix. "
                        + "Star and Planets have repeated suffixes."
                    )
                if indices not in self.suffixes_:  # Not in suffixes
                    raise ValueError(
                        f"Suffix '{indices}' not found in star or planets."
                    )
                # Get planet from int index from suffixes
                indices = self.suffixes_.index(indices)
                return indices if only_index else self.bodies_[indices]
            raise ValueError(
                "Invalid index. "
                + "Expected: 'all', 'star', or planet name. "
                + f"Got: {indices}."
            )

        # Parse indices to list
        indices = parse_to_iter(indices)

        # Check if indices are integers
        if not all(isinstance(i, (int, str)) for i in indices):
            raise TypeError("Indices must be integers or strings.")

        # Check not "all" inside
        if "all" in indices:
            raise ValueError("Cannot mix 'all' with other indices.")

        return [self.body(i) for i in indices]

    def planet(
        self,
        indices: Union[int, str, Iterable[Union[int, str]]],
        only_index: bool = False,
    ) -> Union[StaticPlanet, List[StaticPlanet]]:
        """Slice the planets by given indices.

        The planets are indexed from 0 to n_planets - 1.

        Parameters
        ----------
        indices : int, Iterable[int], Iterable[str], str
            Indices for slicing planets.
            If "all", return all planets.
            Instead of index numbers, the planets names can be used.
        only_index : bool, default=False
            Return only the index of the planets.

        Returns
        -------
        planet : StaticPlanet or list[StaticPlanet]
            A copy of a system's planet :py:class:`StaticPlanet`
            or list of :py:class:`StaticPlanet` objects.
        """
        # Define possible extra in in case circum-binary
        extra = 2 if self.is_binary_ and self.is_circumbinary else 1
        # parse indices to iter
        indices = parse_to_iter(indices)
        for idx, num in enumerate(indices):
            if isinstance(num, int):
                if not 0 <= idx <= self.n_planets_ - 1:
                    raise IndexError(
                        "Index out of range. "
                        + f"Expected: 0 to {self.n_planets_ - 1}. Got: {idx}."
                    )
                indices[idx] = num + extra

        # extract in case of only one index
        if len(indices) == 1:
            indices = indices[0]

        # Check in case "all" planets is requested
        if indices == "all":
            if only_index:
                return list(range(self.n_planets_))
            return self.planets

        # Get planets
        bodies = self.body(indices, only_index=only_index)

        # Check if bodies are planets
        if not only_index and not all(
            isinstance(b, StaticPlanet) for b in parse_to_iter(bodies)
        ):
            raise ValueError("Not all bodies are planets.")
        elif only_index:  # remember to remove extra
            if isinstance(bodies, list):
                return [b - extra for b in bodies]
            return bodies - extra

        return bodies

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

    def _get_single_item(self, item: str):
        """Handle retrieval when a single item is requested."""
        if item.startswith("star_") or item.startswith("binary_"):
            try:
                # If binary, attempt for a binary attribute
                return self.star[
                    item.replace("star_", "").replace("binary_", "")
                ]
            except KeyError:
                raise KeyError(f"Attribute '{item}' not found in binary star.")

        # If binary and star0 or star1
        if (
            self.is_binary_
            and item.startswith("star0_")
            or item.startswith("star1_")
        ):
            if item.startswith("star0_"):
                return self.star.star0[item.replace("star0_", "")]
            if item.startswith("star1_"):
                return self.star.star1[item.replace("star1_", "")]

        if self.n_planets_ > 1:
            return pd.Series(
                self._get_planets_items(item),
                index=self.planet_names_,
                name=item,
            )

        return self.planets[0][item]

    def get_item(
        self, items: Union[str, List[str]], error: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """Retrieve specific attributes of the system (star(s) and/or planets).

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
        # Parse items to list
        items = parse_to_iter(items)

        # Check if error is requested
        if error:
            items_with_error = [
                item
                for item in items
                for item in (item, f"{item}_err_min", f"{item}_err_max")
            ]
            items = [
                item for item in items_with_error if item in RESO_DTYPES.keys()
            ]

        # Get single item
        if len(items) == 1:
            return self._get_single_item(items[0])

        # Retrieve attributes when there are multiple items
        dicc = {}  # Dictionary to store the items and their values
        for item in items:
            dicc[item] = self._get_single_item(item)

        # Check if all scalar values
        if all(isinstance(val, (int, float, str)) for val in dicc.values()):
            return pd.Series(dicc)
        # Create DataFrame
        df = pd.DataFrame(dicc)

        # Return Series if only one column
        return df.squeeze() if len(df.columns) == 1 else df

    def __estimate_period_or_a_or_hill(
        self,
        p_a_h: int,
        which: Union[str, int, List[int]] = "all",
        err_method: int = -1,
        jacobi: bool = False,
        deep_estimate: bool = False,
        force: bool = False,
        circular: bool = False,
    ) -> Union[Tuple[float, float, float], pd.DataFrame]:
        r"""Estimate the 'period' or 'a' of selected planets in the system.

        Calculate the period or semimajor axis of the planet using the
        third Kepler's law.

        Equations:
            :math:`P = 2 \pi \sqrt{\dfrac{a^3}{G (m_\star + m_p)}}`
            :math:`a = \left(\dfrac{G (m_\star + m_p)}
            {4 \pi^2 P^2}\right)^{1/3}`

        Parameters
        ----------
        p_a_h : int
            0 for period, 1 for semimajor axis, 2 for Hill radius.
        which : str, int, list[int], optional. Default: 'all'.
            Which planets to estimate the parameter.
            If 'all', estimate all planets parameter.
            If an :py:class:`int`, estimate the parameter of the planet
            with the given index.
            For example:
            *0* will estimate the parameter of the first planet;
            *1* will estimate the parameter of the second planet.
            If a list of integers, estimate the parameter of the planets with
            the given indices.
            If str, estimate the parameter of the planet with the given
            name or given suffix.
        err_method : int, optional. Default: -1.
            Method to estimate the error.
            See py:func:`resokit.utils.calc_period_with_errors` or
            py:func:`resokit.utils.calc_a_with_errors` for more
            details. The planet must have "a"|"period" and "mass" errors"
            columns, and the star must have "mass" errors columns.
            The options are:
            *-1*: Nothing. Do not estimate the error.
            *0* : No propagation. Return both errors as 0.0.
            *1* : Extremes. Estimate the parameter at the extreme values of
            each parameter and retrieve the errors from the difference.
            *2* : Extended propagation. Assume each parameters follows a normal
            distribution with sigma = err_max.
            *3* : Centred propagation. Assume each parameters follows a normal
            distribution with sigma = (err_min + err_max) / 2.
            *4* : Deviated propagation. Assume each parameters follows a normal
            distribution with sigma = (err_max + err_min) / 2, but the
            mean is at ((val + err_min) + (val + err_max)) / 2.
        jacobi : bool, optional. Default: False.
            Whether to use the Jacobi criterion to estimate the parameter.
            Involves using the accumulated inner mass (star + inner planets)
            instead of just the star mass. Doesn't involve considering the
            semi-major axis of the planets from another reference frame.
        deep_estimate : bool, optional. Default: False.
            Whether to estimate the missing parameters (masses) to calculate
            the parameter.
        force : bool, optional. Default: False.
            Whether to force the estimation of the parameter, even if it
            already exists.
            If `False`, return the existing parameter if it exists.
        circular : bool, optional. Default: False.
            Whether to assume the unknown eccentricities are 0.0.
            Only available for Hill radius estimation.

        Returns
        -------
        parameter, parameter_err_min, parameter_err_max : tuple or DataFrame
            Estimated parameter, and its minimum and maximum errors,
            in days.

        """
        # Check which
        if p_a_h not in [0, 1, 2]:
            raise ValueError("Invalid parameter. Expected 0, 1, or 2.")
        is_period = p_a_h == 0
        is_hill = p_a_h == 2

        # Define parameters
        if is_period:  # Period
            param = "P"
            func = calc_period_with_errors
            anti_param = "a"
        elif is_hill:  # Hill radius
            param = "hill"
            func = calc_hill_radius_with_errors
            anti_param = "P"
        else:  # Semimajor axis
            param = "a"
            func = calc_a_with_errors
            anti_param = "P"

        # Get a list of the planets to use
        planets = parse_to_iter(self.planet(which))

        # Create an empty DataFrame
        df = pd.DataFrame()

        # A simple aux to see if working just with the first planet
        only_first = len(planets) == 1 and planets[0] in [self.planets[0]]
        # Redefine if there is only one planet, or if only the first
        jacobi = jacobi and self.n_planets_ > 1 and not only_first

        # If deep_estimate is requested, estimate the missing parameters
        if deep_estimate:
            used_mass_table = self.estimate_mass(
                which="all",
                force=False,  # Do not force the estimation
                err_method=err_method,
            )
        else:
            used_mass_table = self.get_item("mass", error=err_method > 0)

        # Iterate over the planets
        for pl in planets:
            i = self.planet_names_.index(pl.name)

            # Parameter already exists
            if not is_hill and not isnan(pl[param]) and not force:
                df[f"{pl.name}"] = pl.get_item(param, error=True)
                continue

            # Define the used (this) planet masses
            if err_method > 0:
                pl_mass = used_mass_table.iloc[i, 0]
                pl_mass_err_min = used_mass_table.iloc[i, 1]
                pl_mass_err_max = used_mass_table.iloc[i, 2]
            else:
                pl_mass = used_mass_table.iloc[i]
                pl_mass_err_min = 0
                pl_mass_err_max = 0

            # Check if binary star
            if self.is_binary_:
                # Check if circumbinary
                if self.is_circumbinary:
                    in_m = self.star.total_mass_
                else:  # Normal binary
                    in_m = self.star.star0.mass
                # No errors
                in_m_err_min = 0.0
                in_m_err_max = 0.0
            else:  # Single star
                in_m = self.star.mass
                in_m_err_min = self.star.mass_err_min
                in_m_err_max = self.star.mass_err_max

            # Jacobi criterion with errors
            if jacobi and err_method > 0:
                # Check if not the first planet
                if i > 0:
                    # Keep only the masses of the planets before this one
                    used_mass_table = used_mass_table.iloc[: i - 1]
                    # Create a tuple with the masses and their errors. Also,
                    # convert to solar masses for total mass sumamtion
                    tup = (used_mass_table * Mj2Ms).itertuples(
                        index=False, name=None
                    )
                    # Re Calculate the inner mass and error
                    in_m, in_m_err_min, in_m_err_max = calc_sum_with_errors(
                        (
                            *(
                                in_m,
                                in_m_err_min,
                                in_m_err_max,
                            ),
                            *tup,
                        ),
                    )
            # Jacobi criterion without errors
            elif jacobi:
                # in_m will be the star mass, plus the inner planets mass
                # up to the previous planet
                # Check if not the first planet
                if i > 0:
                    in_m = in_m + used_mass_table.cumsum().iloc[i - 1]
            # No Jacobi criterion. Heliocentric
            else:
                # Already defined
                pass

            # Calculate the param and its errors if not hill radius
            if not is_hill:
                # Calculate the param and its errors
                # We use the anti_param to get the other parameter
                antipar, antipar_err_min, antipar_err_max = pl.get_item(
                    anti_param, error=True
                )
                par, par_err_min, par_err_max = func(
                    antipar,
                    antipar_err_min,
                    antipar_err_max,
                    in_m,
                    in_m_err_min,
                    in_m_err_max,
                    pl_mass,
                    pl_mass_err_min,
                    pl_mass_err_max,
                    err_method,
                )
            # Calculate the Hill radius and its errors
            else:
                # The little trick here, is that if the semi-major axis is
                # not available, we use the period to calculate it, in case
                # deep_estimate is requested
                # First we get the a values
                pl_a, pl_a_err_min, pl_a_err_max = pl.get_item("a", error=True)
                # If the semi-major axis is not available, we use the period
                if isnan(pl_a) and deep_estimate:  # force = False implícito
                    pl_a, pl_a_err_min, pl_a_err_max = calc_a_with_errors(
                        pl.P,
                        pl.P_err_min,
                        pl.P_err_max,
                        in_m,
                        in_m_err_min,
                        in_m_err_max,
                        pl_mass,
                        pl_mass_err_min,
                        pl_mass_err_max,
                        err_method,
                    )
                # Check if the eccentricities can be assumed to be 0.0
                if isnan(pl.e) and circular:
                    pl_e = 0.0
                    pl_e_err_min = 0.0
                    pl_e_err_max = 0.0
                else:
                    pl_e = pl.e
                    pl_e_err_min = pl.e_err_min
                    pl_e_err_max = pl.e_err_max

                # Calculate the Hill radius and its errors
                par, par_err_min, par_err_max = func(
                    pl_a,
                    pl_a_err_min,
                    pl_a_err_max,
                    pl_e,
                    pl_e_err_min,
                    pl_e_err_max,
                    in_m,
                    in_m_err_min,
                    in_m_err_max,
                    pl_mass,
                    pl_mass_err_min,
                    pl_mass_err_max,
                    err_method,
                )

            # Fill the DataFrame
            df[f"{pl.name}"] = [par, par_err_min, par_err_max]

        # Set the index
        df.index = [param, f"{param}_err_min", f"{param}_err_max"]

        # Check if no error requested
        if err_method == -1:  # No error requested
            return df.loc[param]  # Return only the parameter

        return df.T  # Return the DataFrame

    def estimate_period(
        self,
        which: Union[str, int, List[int]] = "all",
        err_method: int = -1,
        jacobi: bool = False,
        deep_estimate: bool = False,
        force: bool = False,
    ) -> Union[Tuple[float, float, float], pd.DataFrame]:
        r"""Estimate the period of selected planets in the system.

        Calculate the period of the planet using the third Kepler's law.

        Equation:
            :math:`P = 2 \pi \sqrt{\dfrac{a^3}{G (m_\star + m_p)}}`

        Parameters
        ----------
        which : str, int, list[int], optional. Default: 'all'.
            Which planets to estimate the period.
            If 'all', estimate all planets period.
            If an :py:class:`int`, estimate the period of the planet with the
            given index.
            For example:
            *0* will estimate the period of the first planet;
            *1* will estimate the period of the second planet.
            If a list of integers, estimate the period of the planets with the
            given indices.
            If str, estimate the period of the planet with the given name or
            given suffix.
        err_method : int, optional. Default: -1.
            Method to estimate the error.
            See py:func:`resokit.utils.calc_period_with_errors` for more
            details. The planet must have "a" and "mass" errors" columns, and
            the star must have "mass" errors columns.
            The options are:
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
        jacobi : bool, optional. Default: False.
            Whether to use the Jacobi criterion to estimate the period.
            Involves using the accumulated inner mass (star + inner planets)
            instead of just the star mass. Doesn't involve considering the
            semi-major axis of the planets from another reference frame.
        deep_estimate : bool, optional. Default: False.
            Whether to estimate the missing parameters (masses) to calculate
            the period.
        force : bool, optional. Default: False.
            Whether to force the estimation of the period, even if it already
            exists.
            If `False`, return the existing period if it exists.

        Returns
        -------
        period, period_err_min, period_err_max : tuple or DataFrame
            Estimated period, and its minimum and maximum errors,
            in days.

        """
        return self.__estimate_period_or_a_or_hill(
            p_a_h=0,
            which=which,
            err_method=err_method,
            jacobi=jacobi,
            deep_estimate=deep_estimate,
            force=force,
        )

    def estimate_semi_major_axis(
        self,
        which: Union[str, int, List[int]] = "all",
        err_method: int = -1,
        jacobi: bool = False,
        deep_estimate: bool = False,
        force: bool = False,
    ) -> Union[Tuple[float, float, float], pd.DataFrame]:
        r"""Estimate the semi-major axis of selected planets in the system.

        Equation:
            :math:`a = \left(\dfrac{G (m_\star + m_p)}
            {4 \pi^2 P^2}\right)^{1/3}`

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
            If str, estimate the semi-major axis of the planet with the given
            name or given suffix.
        err_method : int, optional. Default: -1.
            Method to estimate the error.
            See py:func:`resokit.utils.calc_semi_major_axis_with_errors` for
            more details. The planet must have "P" and "mass" errors" columns,
            and the star must have "mass" errors columns.
            The options are:
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
        jacobi : bool, optional. Default: False.
            Whether to use the Jacobi criterion to estimate the semi-major axis.
            Involves using the accumulated inner mass (star + inner planets)
            instead of just the star mass. Doesn't involve considering the
            semi-major axis of the planets from another reference frame.
        deep_estimate : bool, optional. Default: False.
            Whether to estimate the missing parameters (masses) to calculate
            the semi-major axis.
        force : bool, optional. Default: False.
            Whether to force the estimation of the semi-major axis, even if it
            already exists.
            If `False`, return the existing semi-major axis if it exists.

        Returns
        -------
        a, a_err_min, a_err_max : tuple or DataFrame
            Estimated semi-major axis, and its minimum and maximum errors,
            in AU.
        """
        return self.__estimate_period_or_a_or_hill(
            p_a_h=1,
            which=which,
            err_method=err_method,
            jacobi=jacobi,
            deep_estimate=deep_estimate,
            force=force,
        )

    def estimate_mass(
        self,
        which: Union[str, int, List[int]] = "all",
        force: bool = False,
        **kwargs,
    ) -> Union[Tuple[float, float, float], pd.DataFrame]:
        r"""Estimate the mass of selected planets in the system.

        Equation:
            :math:`mass = \frac{1}{C} \times radius^{1/S}`

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
            If str, estimate the mass of the planet with the given name or
            given suffix.
        force : bool, optional. Default: False.
            Whether to force the estimation of the mass, even if it already
            exists.
            If `False`, return the existing mass if it exists.
        model : str, optional. Default: "ck17"
            Model to use for the mass-radius power-law relation.
            'ck17': Chen & Kipping (2017) [trivariate]
            'o20': Otegi et al. (2020) [density|bivariate]
            'e23': Edmondson et al. (2023)
            'm24': Müller et al. (2024) [bivariate]
        multivariate : float, tuple, optional. Default: 0.5
            Probability of using the (first, second, ...) branch if the
            estimation falls in a multivariate region.
            For bivariate models ('o20', 'm24'), it must be a float between
            0 and 1.
            For trivariate model "ck17", it must be a tuple of two floats
            between 0 and 1, where the sum of them must be lower equal than 1.
        err_method : int, optional. Default: -1
            Which method implement for error calculation.
            Method -1: Do not calculate errors. Return only the mass.
            Method 0: Do not calculate errors. Return both as 0.0.
            Method 1: (Naive) Error propagation with the power-law
            approximation, using the radius error as the maximum of the
            two extremes.
            Warning: May return excessively large errors for multivariate
            sections.
            Method 2: Evaluate the radius extremes and calculate each mass
            extreme with the power-law approximation.
            Method 3: Returns the approximate model error as value errors.
        **kwargs : dict
            Additional keyword arguments for the
            :py:func:`resokit.utils.mass_radius.estimate_mass` function.

        Note
        ----
        If `err_method=-1`, only the mass is returned. If `err_method=0`, the
        errors are 0.0.

        Note
        ----
        To use the errors, the planet must have "radius" error columns.


        Returns
        -------
        mass, mass_err_min, mass_err_max : tuple or DataFrame
            Estimated mass, and its minimum and maximum errors,
            in Jupiter masses.
        """
        # Get a list of the planets to use
        planets = parse_to_iter(self.planet(which))

        # Create an empty DataFrame
        df = pd.DataFrame()  # Create an empty DataFrame

        # Define the error method
        ret_err = True
        if kwargs.get("err_method", -1) == -1:
            kwargs["err_method"] = 0  # Set to 0 for the function
            ret_err = False

        # Iterate over the planets
        for pl in planets:
            # Check if the mass already exists
            if not isnan(pl.mass) and not force:
                df[f"{pl.name}"] = [pl.mass, pl.mass_err_min, pl.mass_err_max]
                continue
            mass, mass_err_min, mass_err_max = pl.estimate_mass(**kwargs)
            df[f"{pl.name}"] = [
                mass,
                mass_err_min,
                mass_err_max,
            ]
        df.index = ["mass", "mass_err_min", "mass_err_max"]

        if not ret_err:  # No error requested
            return df.loc["mass"]  # Return only the mass

        return df.T  # Return the DataFrame

    def estimate_radius(
        self,
        which: Union[str, int, List[int]] = "all",
        force: bool = False,
        **kwargs,
    ) -> Union[Tuple[float, float, float], pd.DataFrame]:
        r"""Estimate the radius of selected planets in the system.

        Equation:
            :math:`radius = C \times mass^S`

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
            If str, estimate the radius of the planet with the given name or
            given suffix.
        force : bool, optional. Default: False.
            Whether to force the estimation of the radius, even if it already
            exists.
            If `False`, return the existing radius if it exists.
        model : str, optional. Default: "ck17"
            Model to use for the mass-radius power-law relation.
            'ck17': Chen & Kipping (2017)
            'o20': Otegi et al. (2020) [density|bivariate]
            'e23': Edmondson et al. (2023)
            'm24': Müller et al. (2024)
        bivariate : float, optional. Default: 0.5
            Probability of using the lower branch if the estimation falls in a
            bivariate region. Must be a number between 0 and 1.
            Only used if model is 'o20'.
        err_method : int, optional. Default: -1
            Which method implement for error calculation.
            Method -1: Do not calculate errors. Return only the radius.
            Method 0: Do not calculate errors. Return both as 0.0.
            Method 1: (Naive) Error propagation with the power-law
            approximation, using the mass error as the maximum of the
            two extremes.
            Method 2: Evalaute the mass extremes and calculate each
            radius extreme.
            Method 3: Returns the approximate model error as value errors.
        **kwargs : dict
            Additional keyword arguments for the
            :py:func:`resokit.utils.mass_radius.estimate_radius` function.

        Note
        ----
        If `err_method=-1`, only the radius is returned. If `err_method=0`, the
        errors are 0.0.

        Note
        ----
        To use the errors, the planet must have "radius" error columns.


        Returns
        -------
        radius, radius_err_min, radius_err_max : tuple or DataFrame
            Estimated radius, and its minimum and maximum errors,
            in Jupiter radii.
        """
        # Get a list of the planets to use
        planets = parse_to_iter(self.planet(which))

        # Create an empty DataFrame
        df = pd.DataFrame()

        # Define the error method
        ret_err = True
        if kwargs.get("err_method", -1) == -1:
            kwargs["err_method"] = 0
            ret_err = False

        # Iterate over the planets
        for pl in planets:
            # Check if the radius already exists
            if not isnan(pl.radius) and not force:
                df[f"{pl.name}"] = [
                    pl.radius,
                    pl.radius_err_min,
                    pl.radius_err_max,
                ]
                continue
            radius, radius_err_min, radius_err_max = pl.estimate_radius(
                **kwargs
            )
            df[f"{pl.name}"] = [
                radius,
                radius_err_min,
                radius_err_max,
            ]
        df.index = ["radius", "radius_err_min", "radius_err_max"]

        if not ret_err:  # No error requested
            return df.loc["radius"]  # Return only the radius

        return df.T  # Return the DataFrame

    def estimate_hill_radius(
        self,
        which: Union[str, int, List[int]] = "all",
        err_method: int = -1,
        jacobi: bool = False,
        deep_estimate: bool = False,
        circular: bool = False,
    ) -> Union[Tuple[float, float, float], pd.DataFrame]:
        r"""Calculate the Hill radius of selected planets in the system.

        Equation:
            :math:`r_H = a (1 - e) \left(\dfrac{m_p}
            {3 (m_\star + m_p)}\right)^{1/3}`

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
            If str, estimate the period of the planet with the given name or
            given suffix.
        err_method : int, optional. Default: -1.
            Method to estimate the error.
            See py:func:`resokit.utils.hill_radius.calc_hill_radius_with_errors`
            for more details. The planet must have "a", "e" and "mass" error
            columns, and the star must have "mass" errors columns.
            The options are:
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
        jacobi : bool, optional. Default: False.
            Whether to use the Jacobi criterion to estimate the Hill radius.
            Involves using the accumulated inner mass (star + inner planets)
            instead of just the star mass. Doesn't involve considering the
            semi-major axis of the planets from another reference frame.
        deep_estimate : bool, optional. Default: False.
            Whether to estimate the missing parameters (masses and semi-major
            axis) to calculate the Hill radius.
            If the mass and semi-major axis are missing, they will be estimated;
            otherwise, the existing values will be used.
        circular : bool, optional. Default: False.
            Whether to assume the unknown eccentricities are 0.0.

        Returns
        -------
        rhill, rhill_err_min, rhill_err_max : tuple or DataFrame
            Hill radius, and its minimum and maximum errors,
            in AU.
        """
        return self.__estimate_period_or_a_or_hill(
            p_a_h=2,
            which=which,
            err_method=err_method,
            jacobi=jacobi,
            deep_estimate=deep_estimate,
            circular=circular,
        )

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

        Note
        ----
        Crossed attributes (e.g., x='star_mass', y='mass') are not allowed yet.

        Note
        ----
        To use the error bars, the planets (star) must have the corresponding
        error columns.

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
            raise NotImplementedError(
                "Crossed attributes are not allowed. "
                + "Use either both star attributes or both planet attributes."
            )
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
        if self.is_binary_:
            raise NotImplementedError(
                "Binary stars are not supported yet. "
                + "Use the planets instead."
            )

        # Remove the 'star_' prefix
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
        r"""Plot consecutive triplets of planets in the period ratio space.

        Systems triplets are shown in the plane
            :math:`P_{i+1}/P_i` vs. :math:`P_{i+2}/P_{i+1}`.

        Note
        ----
        Only available for planets yet.

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
            Only available if the planets have period ratio error columns.
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

        # Get (all) error ratios if needed
        if error:
            error_ratios = self.pair_ratio(
                error=True, verbose=False, use_binary=False
            )

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
            x = self.pair_ratio(j, i, verbose=False, use_binary=False)
            y = self.pair_ratio(k, j, verbose=False, use_binary=False)
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
            body = self.body(index)
            if isinstance(body, StaticStar):
                raise ValueError("Cannot remove the star.")
            index = self.planet_names_.index(body.name)

        if index < 0 or index >= self.n_planets_:
            raise IndexError("Index out of range.")

        # Create a new list of planets
        new_planets = [  # This way to avoid "index + 1 :" <BLACK>
            self.planets[i] for i in range(self.n_planets_) if i != index
        ]

        # Create a new metadata dictionary
        new_meta = dict(self.metadata)
        new_meta["removed_planet"] = new_meta.get("removed_planet", []) + [
            self.planets[index].name
        ]

        # Create a new StaticSystem instance
        new_ss = attrs.evolve(self, planets=new_planets, metadata=new_meta)

        # Print message
        if verbose:
            print(f"Planet {self.planets[index].name} [{index}] removed.")

        return new_ss

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
        new_meta = dict(self.metadata)
        new_meta["added_planet"] = new_meta.get("added_planet", []) + [
            planet.name
        ]

        # Create a new StaticSystem instance
        new_ss = attrs.evolve(self, planets=new_planets, metadata=new_meta)

        # Print message
        if verbose:
            print(f"Planet {planet.name} added.")

        return new_ss

    def swap_planets(
        self,
        i: Union[int, str],
        j: Union[int, str],
        verbose: bool = True,
    ) -> "StaticSystem":
        """Switch the position of two planets in the system.

        Parameters
        ----------
        i : int, str
            Index or suffix (1 char) or name of the first planet to switch.
        j : int, str
            Index or suffix (1 char) or name of the second planet to switch.
        verbose : bool, optional. Default: True.
            Whether to print a message when switching the planets.

        Returns
        -------
        StaticSystem
            A new :py:class`StaticSystem` instance with the switched planets.
        """
        # Get the indexes of the planets
        i = self.planet(i, only_index=True)
        j = self.planet(j, only_index=True)

        # Check if the planets are the same
        if i == j:
            if verbose:
                print("The planets are the same. Nothing to do.")
            return self

        # Create a new list of planets
        new_planets = self.planets.copy()
        new_planets[i], new_planets[j] = new_planets[j], new_planets[i]

        # Create a new StaticSystem instance
        new_ss = attrs.evolve(self, planets=new_planets)

        # Print message
        if verbose:
            print(
                f"Planets {self.planets[i].name} [{i}] and "
                + f"{self.planets[j].name} [{j}] switched."
            )

        return new_ss

    def replace_planet(
        self,
        index: Union[int, str],
        planet: StaticPlanet,
        verbose: bool = True,
    ) -> "StaticSystem":
        """Replace a planet in the system with a new one.

        Parameters
        ----------
        index : int, str
            Index or suffix (1 char) or name of the planet to change.
        planet : StaticPlanet
            StaticPlanet instance to add.
        verbose : bool, optional. Default: True.
            Whether to print a message when changing the planet.

        Returns
        -------
        StaticSystem
            A new :py:class`StaticSystem` instance with the changed planet.
        """
        # Get the index of the planet
        index = self.planet(index, only_index=True)
        old_name = self.planets[index].name

        # Edit the source of the new planet
        planet = planet.change_value("source", "user")

        # Create a new list of planets
        new_planets = self.planets.copy()
        new_planets[index] = planet

        # Modify the metadata
        new_meta = dict(self.metadata)
        new_meta["changed_planet"] = new_meta.get("changed_planet", []) + [
            planet.name
        ]

        # Create a new StaticSystem instance
        new_ss = attrs.evolve(self, planets=new_planets, metadata=new_meta)

        # Print message
        if verbose:
            print(f"Planet {old_name} [{index}] replaced by {planet.name}.")

        return new_ss

    def pair_ratio(
        self,
        *pair: Union[int, list, tuple, str],
        verbose: bool = True,
        error: bool = False,
        use_binary: bool = True,
        **fraction_kwargs: dict,
    ) -> Union[float, pd.DataFrame]:
        r"""Return the period ratio of the specified pair of planets.

        Parameters
        ----------
        pair : int, list, tuple, str, optional. Default: 'all'.
            Which pair of planets to consider.
            Either 'all' or a list/tuple of planet names/indexes.
            If *pair=(i,j)*, then the period ratio is :math:`P_j/P_i`, and
            remember that the first planet is 0.
        verbose : bool, optional. Default: False.
            Whether to print the steps of the calculation if a single pair,
            and fraction_arg is not 0.
        error : bool, optional. Default: False.
            Whether to return the error of the period ratio, instead of the
            period ratio itself. Only meaningful if there are errors.
        use_binary : bool, optional. Default: True.
            Whether to use the binary period ratio (include the star).
            If False, only the planets are considered.
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
        # Redefine use_binary if needed
        if not self.is_binary_:
            use_binary = False
        # Check if there are at least 3 bodies
        if self.n_planets_ + (2 if use_binary else 1) < 3:
            raise ValueError("There must be at least 3 bodies to compare.")

        # Extract pair
        if not pair or pair == ("all",):
            pair = "all"
        elif len(pair) > 2:
            raise ValueError("Pair must have 2 elements.")
        elif len(pair) == 1:
            pair = pair[0]
            if isinstance(pair, int):
                return self.pair_ratio(
                    pair,
                    pair + 1,
                    verbose=verbose,
                    error=error,
                    use_binary=use_binary,
                    **fraction_kwargs,
                )
            if not isinstance(pair, Iterable) or len(pair) != 2:
                raise ValueError("Pair must have 2 elements.")

        # Add verbose to fraction_kwargs, if fraction_kwargs is not empty
        if fraction_kwargs:
            fraction_kwargs["verbose"] = verbose

        # This calculates all the period ratios
        if isinstance(pair, str) and not error:

            if not pair == "all":  # Check if it's 'all'
                raise ValueError("Invalid pair value.")

            if (self.n_planets_ == 2 and not use_binary) or (
                self.n_planets_ == 1 and use_binary
            ):  # Only 3 bodies
                # Already calculated!
                if fraction_kwargs:  # Convert to fraction if needed
                    return float_to_fraction(
                        self.period_ratios_,
                        **fraction_kwargs,
                    )
                return self.period_ratios_

            # Get all the period ratios
            periods = self.get_item("P")
            # Create a Series in case a single object is returned
            if self.n_planets_ == 1:
                periods = pd.Series(periods, index=self.planet_names)
            # Check if the star has to be included
            if use_binary:
                if not self.is_circumbinary:  # Add at the end
                    periods.loc[self.star_names_[1]] = self.star.P
                else:  # Add at the beginning
                    periods = pd.concat(
                        [
                            pd.Series(
                                self.star.P, index=[self.star_names_[1]]
                            ),
                            periods,
                        ]
                    )
            df = pd.DataFrame(
                [[p2 / p1 for p2 in periods] for p1 in periods],
                index=periods.index,
                columns=periods.index,
            )

            if fraction_kwargs:
                return df.map(
                    lambda x: float_to_fraction(
                        x,
                        **fraction_kwargs,
                    )
                )

            return df

        # If pair is a single pair
        if not pair == "all":
            idxs = []  # Indexes of the pair
            for idx in pair:
                # If using integer
                if not isinstance(idx, int):
                    pl = self.planet(idx)
                    idx = self.planet_names_.index(pl.name)
                if not use_binary and self.is_circumbinary:
                    idx += 1
                idxs.append(idx)
        elif error:
            idxs = "all"
        else:
            raise ValueError("Invalid pair value.")

        # If error is True, return the error of the period ratio
        if error:
            err_df = self._pair_ratio_error(idxs)
            # Check if remove binary
            if idxs == "all" and not use_binary and self.is_binary_:
                if self.is_circumbinary:
                    err_df = err_df.iloc[1:, 1:]
                else:
                    err_df = err_df.iloc[:-1, :-1]
            return err_df

        # This is sigle pair
        if (self.n_planets_ == 2 and not use_binary) or (
            self.n_planets_ == 1 and use_binary
        ):  # Only 3 bodies
            # Already calculated!
            if not (idxs == [0, 1] or idxs == [1, 0]):
                raise ValueError("Invalid pair value. Must be 0, 1.")
            # Return the ratio
            if fraction_kwargs:
                return float_to_fraction(
                    self.period_ratios_,
                    **fraction_kwargs,
                )

            return self.period_ratios_

        # Obtain the ratio
        ratio = self.period_ratios_.iloc[idxs[1], idxs[0]]

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
        if (
            self.n_planets_ + (2 if self.is_binary_ else 1) <= 3
        ):  # No error for 2 bodies
            return self.__error_ratios__  # Already calculated for 3 bodies

        # Extract pair ratio
        pair_ratio = self.pair_ratio(*pair, error=False, use_binary=False)

        # Formula: sqrt((err1/P1)^2 + (err2/P2)^2) * ratio

        # If pair is all. First call will be this one
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

            # Add None Error to binary if needed
            if self.is_binary_:
                df[self.star_names_[1]] = None
                df.loc[self.star_names_[1]] = None
                if (
                    self.is_circumbinary
                ):  # Reorder to get the star at the beginning
                    df = df[[self.star_names_[1]] + self.planet_names_]
                    df = df.reindex([self.star_names_[1]] + self.planet_names_)

            # Store the DataFrame
            self.__error_ratios__[df.columns] = df

            return df

        # Be sure that self.__error_ratios__ is not empty
        if self.__error_ratios__.empty:
            self.pair_ratio("all", error=True)

        # If pair is a single pair. Assume already parsed by pair_ratio
        # If pair is something like ([i,j],), then pair[0] is the pair
        if len(pair) == 1:
            pair = pair[0]

        # Get the indexes
        i, j = pair

        # Get the error from calculated data
        return self.__error_ratios__.iloc[j, i]

    def change_value(
        self,
        attr: str,
        value: any,
        in_star: Union[None, str, int] = None,
        in_planet: Union[None, int, str] = None,
        in_binary: Union[None, bool] = None,
        verbose: bool = True,
    ) -> "StaticSystem":
        """Set an attribute of the StaticSystem and return a new instance.

        Note
        ----
        If in_star, in_planet, and in_binary are all None, the attribute of the
        system is changed.

        Parameters
        ----------
        attr : str
            Attribute to set.
        value : any
            Value to set.
        in_star : bool, int, optional (default: None)
            If not None, modify the attribute of the star.
            If the system has a binary star, the star can be selected with
            the index (0, 1) or the name.
        in_planet : int, str, optional (default: None)
            If not None, modify the attribute of the planet.
            The planet can be selected with the index or the name or suffix.
        in_binary : bool, optional (default: None)
            If not None, modify the attribute of the binary system.
        verbose : bool, optional (default: True)
            Whether to print a message when changing the attribute.

        Returns
        -------
        StaticSystem
            A new instance with the updated attribute.
        """
        # Default_to change
        to_change = dict()
        # Check if any of the in_* is not None
        if in_star is not None:
            if in_planet is not None or in_binary is not None:
                raise ValueError(
                    "Cannot set in_star with in_planet or in_binary."
                )
            if not self.is_binary_:
                to_change = dict({"star": self.star.change_value(attr, value)})
            else:
                # Here, "star" is a binary star
                to_change = dict(
                    {
                        "star": self.star.change_value(
                            attr, value, in_star=in_star
                        )
                    }
                )
        elif in_planet is not None:
            if in_binary is not None:
                raise ValueError("Cannot set in_planet with in_binary.")
            planet = self.planet(in_planet)
            new_planet = planet.change_value(attr, value)
            # Update list of planets
            new_planets = self.planets.copy()
            new_planets[self.planet_names_.index(planet.name)] = new_planet
            to_change = dict({"planets": new_planets})
        elif in_binary is not None:
            if not self.is_binary_:
                raise ValueError(
                    "Cannot set in_binary in a single star system."
                )
            to_change = dict(
                {"star": self.star.change_value(attr, value, in_star=None)}
            )
        else:
            to_change = dict({attr: value})

        # Modify the metadata
        new_metadata = dict(self.metadata)
        msg = f"Changed {attr} to {value}."
        new_metadata["history"] = new_metadata.get("history", []) + [msg]

        # Update to_change with the attribute and metadata
        to_change["metadata"] = new_metadata

        # Create a new StaticSystem instance
        new_ss = attrs.evolve(self, **to_change)

        # Message
        if verbose:
            msg = msg[:-1] + f" in {self.name}."
            print(msg)

        return new_ss

    def to_dict(self) -> dict:
        """Return the metadata as a new dictionary."""
        return dict(self.metadata)

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
            {planet.name: planet.data for planet in self.planets}
        )

        if add_star is None:
            if columns is not None:
                used_cols = [col for col in columns if col in df.columns]
                df = df[used_cols]
            return df

        # Generate star data
        # Check if binary
        if self.is_binary_:  # Binary
            star0_df = pd.Series(self.star.star0.data).to_frame(
                self.star.star0.name
            )
            star1_df = pd.Series(self.star.star1.data).to_frame(
                self.star.star1.name
            )
            star_df = pd.concat([star0_df, star1_df], axis=1)
        # Single star
        else:
            star_df = pd.Series(self.star.data).to_frame(self.star.name)

        # Drop RESO_OB_TYPES columns, as they are already in the planets
        drop2 = [col for col in RESO_OB_TYPES.keys() if col in star_df.index]
        star_df.drop(drop2, inplace=True)

        # Change star columns to inlclude "star_". Exclude RESO_OB_TYPES
        star_df = star_df.rename(lambda x: f"star_{x}")

        if add_star:
            # Concatenate star data
            # Check if not simple binary
            if not self.is_binary_ or self.is_circumbinary:
                df = pd.concat([star_df, df], axis=0)
            else:  # Simple binary
                # Add the planets data between the stars data
                # So: Star1, Planet1, Planet2, ..., Star2
                df = pd.concat(
                    [star_df.iloc[:, 0], df, star_df.iloc[:, 1]], axis=1
                )
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

    def to_rebound(
        self,
        sim: Simulation = None,
        fillna: bool = True,
        units: Union[bool, Tuple[float, float]] = True,
        random_m_angle: bool = False,
        verbose: bool = True,
    ) -> "Simulation":
        """Return a REBOUND simulation with the system data.

        Note
        ----
            The REBOUND simulation is created with the heliocentric orbits.

        Note
        ----
            The simulation units are not changed. The user has to ensure the
            consistency of the units after the simulation is created.
            (eg. if the user provides the units in [Kg, m], then
            sim.units = ('kg', 'm', <time_unit>) has to be set after the
            simulation is created).

        Parameters
        ----------
        sim : rebound.Simulation, optional. Default: None.
            REBOUND simulation to add the system data.
            If None, create a new simulation.
        fillna : bool, optional. Default: True.
            Whether to fill missing data with estimations.
        units : bool, tuple, optional. Default: True.
            Whether to use AU, Msun units.
            If False, use m, Kg units.
            If tuple, use the given units. The tuple must be
            (unit_mass, unit_length). The user has to ensure the
            consistency of the units after the simulation is created.
        random_m_angle : bool, optional. Default: False.
            Whether to use random values for the mean anomaly, in
            case the planets do not have it.
            If False, use 0.0.
        verbose : bool, optional. Default: True.
            Whether to print a message when creating the simulation.

        """
        # Check if REBOUND is imported
        assert_module_imported(rebound_imported, "rebound")

        # Create a new simulation if not provided
        if sim is None:
            sim = Simulation()
            if verbose:
                print("New REBOUND simulation created.")

        # Define units
        if isinstance(units, bool):
            if units:  # AU, Msun
                units = (M_SUN, AU)
                if verbose:
                    print(" Using AU, Msun units.")
            else:  # m, Kg
                units = (1.0, 1.0)
                if verbose:
                    print(" Using m, Kg units.")
        elif verbose:
            print(" Using user-defined units:", units)
        # If not defined, the user provides (unit_mass, unit_length) in [Kg, m]

        # Add the central star
        if self.is_binary_:  # Binary
            sim.add(
                m=self.star.star0.mass * M_SUN / units[0],
                r=self.star.star0.radius * R_SUN / units[1],
                hash=self.star.star0.name,
            )
            if self.is_circumbinary:  # Circumbinary
                sim.add(
                    m=self.star.star1.mass * M_SUN / units[0],
                    r=self.star.star1.radius * R_SUN / units[1],
                    hash=self.star.star1.name,
                )
        else:  # Single star
            sim.add(
                m=self.star.mass * M_SUN / units[0],
                r=self.star.radius * R_SUN / units[1],
                hash=self.star.name,
            )
        # Print message
        if verbose and not self.is_binary_:
            print(f" Star {self.star.name} added.")
        elif verbose and self.is_binary_ and self.is_circumbinary:
            print(
                f" Stars {self.star.star0.name} "
                + f"and {self.star.star1.name} added."
            )
        elif verbose and self.is_binary_:
            print(f" Star {self.star.star0.name} added.")

        # Add the planets. Loop
        for planet in self.planets:
            # Fill mass or radius with estimation if not given
            pmass = (
                planet.mass
                if planet.mass > 0
                else planet.estimate_mass(err_method=-1) if fillna else 0.0
            )
            pradius = (
                planet.radius
                if planet.radius > 0
                else planet.estimate_radius(err_method=-1) if fillna else 0.0
            )
            pa = (
                planet.a
                if planet.a > 0
                else (
                    self.estimate_semi_major_axis(
                        which=planet.name,
                        err_method=-1,  # No error for rebound
                        jacobi=False,  # Heliocentric orbit
                    )
                    if fillna
                    else 0.0
                )
            )
            sim.add(
                m=pmass * M_JUP / units[0],
                r=pradius * R_JUP / units[1],
                a=pa * AU / units[1],
                e=planet.e,
                inc=planet.inc * DEG2RAD if planet.inc > 0 else 0.0,
                omega=planet.w * DEG2RAD if planet.w > 0 else 0.0,
                M=rng.uniform(0, 2 * pi) if random_m_angle else 0.0,
                hash=planet.name,
                primary=sim.particles[self.star.name],  # Star
            )
        if verbose:
            print(f" {self.n_planets_} planets added.")

        # Check if final binary
        if self.is_binary_ and not self.is_circumbinary:
            sim.add(
                m=self.star.star1.mass * M_SUN / units[0],
                r=self.star.star1.radius * R_SUN / units[1],
                hash=self.star.star1.name,
            )
            if verbose:
                print(f" Star {self.star.star1.name} added.")

        return sim

    def copy(self) -> "StaticSystem":
        """Return a copy of the :py:class:`StaticSystem`."""
        return attrs.evolve(self)


# =============================================================================
# CREATION FUNCTIONS
# =============================================================================


def _create_static_planet(
    planet_data: pd.Series,
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

    return StaticPlanet(data=planet_data, source=source, metadata=metadata)


def _create_static_star(
    star_data: pd.Series,
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

    return StaticStar(data=star_data, source=source, metadata=metadata)


def _create_static_system(
    star: Union[StaticStar, StaticBinaryStar],
    planets: List[StaticPlanet],
    name: str,
    metadata=None,
    circumbinary: bool = False,
) -> StaticSystem:
    """Create a :py:class:`StaticSystem` instance.

    Parameters
    ----------
    star : StaticStar, StaticBinaryStar
        StaticStar or StaticBinaryStar instance.
    planets : list, tuple, StaticPlanet
        List of StaticPlanet instances.
    name : str
        Name of the system.
    metadata : dict, optional. Default: {}.
        Metadata of the dataset.
    circumbinary : bool, optional. Default: False.
        Whether the system is circumbinary.

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
        is_circumbinary=circumbinary,
    )


def _create_static_binary_star_from_binaries(
    star0: StaticStar,
    star1: StaticStar,
    binary_row: pd.Series,
    name: str,
    metadata=None,
) -> StaticBinaryStar:
    """Create a :py:class:`StaticBinaryStar` instance.

    Parameters
    ----------
    star0 : StaticStar
        StaticStar instance for the primary star.
    star1 : StaticStar
        StaticStar instance for the secondary star.
    binary_row : pd.Series
        Pandas Series with the binary data.
    name : str
        Name of the binary system.
    metadata : dict, optional. Default: {}.
        Additional metadata about the star.

    Returns
    -------
    StaticBinaryStar
        A new StaticBinaryStar instance.
    """
    if metadata is None:
        metadata = {}

    # Create the StaticBinaryStar instance
    return StaticBinaryStar(
        star0=star0,
        star1=star1,
        data=binary_row.to_frame(name=name).T,
        name=name,
        metadata=metadata,
    )


def resokit_to_system(
    resokit_data: ResokitDataFrame,
    binary_star: Union[None, StaticBinaryStar] = None,
    circumbinary: bool = False,
    verbose: bool = False,
) -> StaticSystem:
    """Convert a :py:class:`ResokitDataFrame` to a :py:class:`StaticSystem`.

    Parameters
    ----------
    resokit_data : ResokitDataFrame
        ResokitDataFrame instance with the data.
    binary_star : StaticBinaryStar, optional. Default: None.
        StaticBinaryStar instance to use as the binary star.
    circumbinary : bool, optional. Default: False.
        Whether the system is circumbinary.
    verbose : bool, optional. Default: True.
        Whether to print a message when creating the system.

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
        | {"star_name", "n", "n_err_min", "n_err_max", "binary"}
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

    # Message
    if verbose:
        print(f"Creating system {star_df.name}.")

    # Create star
    star = _create_static_star(
        star_data=star_df,
        source=resokit_data.source if binary_star is None else "binary",
        metadata=resokit_data.metadata,
    )
    # Create binary star only if needed
    if binary_star is None:
        if verbose:
            print(f" Star {star.name} created.")
    else:
        binary = _create_static_binary_star_from_binaries(
            star0=star,  # Star0 is the one from the Resokit data
            star1=binary_star.star1,
            binary_row=binary_star.data,
            name=star_df.name,
            metadata=resokit_data.metadata,
        )
        star = binary
        if verbose:
            print(f" Using binary star {star.name}.")

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

    # Message
    if verbose:
        print(f" {resokit_data.n_objects_} planets created.")

    return _create_static_system(
        star=star,
        planets=planets,
        name=star.name,
        metadata=resokit_data.metadata,
        circumbinary=circumbinary,
    )


def binary_row_to_binary_star(
    binary_row: pd.Series,
    source="user",
    metadata=None,
) -> StaticBinaryStar:
    """Convert a binary row to a :py:class:`StaticBinaryStar`.

    Parameters
    ----------
    binary_row : pd.Series
        Pandas Series with the binary data.
    source : str, optional. Default: 'user'.
        Source of the data.
    metadata : dict, optional. Default: {}.
        Additional metadata about the star.

    Returns
    -------
    StaticBinaryStar
        :py:class:`StaticBinaryStar` instance.
    """
    # Get the systems name
    name = binary_row["star0_name"]
    star0_name = name + " A"
    star1_name = name + " B"

    # Create the necessary series for the StaticStar instances
    # First, the shared parameters we usually get in a StaticStar
    shared_resokit = binary_row[["dist", "disc_method"]].copy()
    shared_resokit.columns = ["star_dist", "disc_method"]  # Same as Resokit
    # Then, the parameters for each star
    star0_resokit = pd.Series(
        {
            "star_name": star0_name,
            "star_mass": binary_row["star0_mass"],
        }
    )
    star1_resokit = pd.Series(
        {
            "star_name": star1_name,
            "star_mass": binary_row["star1_mass"],
        }
    )

    # Create the StaticStar dfs
    star0_df = pd.concat([shared_resokit.squeeze(), star0_resokit], axis=0)
    star1_df = pd.concat([shared_resokit.squeeze(), star1_resokit], axis=0)

    # Rename the columns. If the column starts with "star_" we remove it.
    star0_df.rename(lambda x: str(x).replace("star_", ""), inplace=True)
    star1_df.rename(lambda x: str(x).replace("star_", ""), inplace=True)

    # Set the dataframe names
    star0_df.name = star0_name
    star1_df.name = star1_name

    # Create the StaticStar instances
    star0 = _create_static_star(star0_df, source=source, metadata=metadata)
    star1 = _create_static_star(star1_df, source=source, metadata=metadata)

    # Add the total binary mass to binary_row
    binary_row["mass"] = binary_row["star0_mass"] + binary_row["star1_mass"]

    return _create_static_binary_star_from_binaries(
        star0=star0,
        star1=star1,
        binary_row=binary_row,
        name=name,
        metadata=metadata,
    )
