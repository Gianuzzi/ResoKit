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

""" Module ResoKit. """

# =============================================================================
# IMPORTS
# =============================================================================

import warnings
from collections.abc import Mapping
from typing import Union

import attrs
import pandas as pd

from resokit.utils import (
    MAPPINGS,
    RESO_OB_TYPES,
    RESO_PL_TYPES,
    RESO_SR_TYPES,
    float_to_fraction,
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
    >>> metadata = MetaData({"a": 1, "b": 2})
    >>> metadata.a
    1

    >>> metadata["a"]
    1
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
    """
    Initialize a ResoKit DataFrame class.

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
        converter=lambda df: df.squeeze(),
    )
    source: str = attrs.field(
        validator=attrs.validators.in_({"eu", "nasa", "user"}),
        converter=str.lower,
    )
    metadata: dict = attrs.field(factory=MetaData, converter=MetaData)

    columns_: list = attrs.field(init=False)
    n_columns_: int = attrs.field(init=False)
    n_objects_: int = attrs.field(init=False)

    @columns_.default
    def _columns__default(self):
        """Default value for columns_."""
        cols = (
            self.data_df.index
            if isinstance(self.data_df, pd.Series)
            else self.data_df.columns
        )
        return cols.to_list()

    @n_columns_.default
    def _n_columns__default(self):
        """Default value for n_objects_."""
        return len(self.columns_)

    @n_objects_.default
    def _n_objects__default(self):
        """Default value for n_objects_."""
        return (
            self.data_df.shape[0]
            if isinstance(self.data_df, pd.DataFrame)
            else 1
        )

    def __attrs_post_init__(self):
        """Post-initialization hook."""
        if self.data_df.empty:
            warnings.warn("Empty DataFrame.")
        if "name" not in self.columns_:
            warnings.warn("Missing 'name' column in the DataFrame.")
        if (self.n_objects_ == 1) and not isinstance(self.data_df, pd.Series):
            raise TypeError(
                "With only one object, data_df must be a Series "
                + "(not a DataFrame)"
            )
        return

    def __len__(self):
        """len(x) <=> x.__len__()."""
        return self.n_objects_

    def __getitem__(self, slice):
        """x[y] <==> x.__getitem__(y)."""
        if self.n_objects_ == 1:
            if isinstance(slice, int):
                return self.data_df.iloc[slice]
            if isinstance(slice, list):
                if all(isinstance(i, int) for i in slice):
                    return self.data_df.iloc[slice]
            return self.data_df[slice]
        return self.data_df.__getitem__(slice)

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
        """
        Return the data_df as a new DataFrame.

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

    # def iloc(self, *args, **kwargs):
    #     """Return the i-th row of the data, as a ResokitDataFrame."""
    #     return ResokitDataFrame(
    #         data_df=self.data_df.iloc(*args, **kwargs),
    #         source=self.source,
    #         metadata=dict(self.metadata),
    #     )


def df_to_resokit(
    df: pd.DataFrame,
    source: str,
    drop: bool = True,
    copy: bool = False,
    metadata: dict = {},
) -> ResokitDataFrame:
    """
    Convert ExoplanetEU or NASA dataset to ResoKit format.

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

    # Order by P[eriod] column
    if "P" in df.columns:
        df = df.sort_values(by="P", ascending=True)

    return ResokitDataFrame(data_df=df, source=source, metadata=metadata)


# @attrs.define(repr=False, slots=True)
# class Planet:
#     """
#     Planet class representing a planet with various attributes.

#     Attributes
#     ----------
#     name : str
#         Name of the planet.
#     mass : float
#         Mass of the planet in Jupiter masses.
#     radius : float
#         Radius of the planet in Jupiter radii.
#     semi_major_axis : float
#         Semi-major axis of the planet's orbit in AU.
#     eccentricity : float
#         Eccentricity of the planet's orbit.
#     inclination : float
#         Inclination of the planet's orbit in degrees.
#     mean_anomaly : float
#         Mean anomaly of the planet in degrees.
#     argument_of_pericenter : float
#         Argument of pericenter in degrees.
#     longitude_of_ascending_node : float
#         Longitude of the ascending node in degrees.
#     star_name : str
#         Name of the star the planet orbits.
#     metadata : dict
#         Additional metadata about the planet.
#     """

#     name: str = attrs.field(
#         validator=attrs.validators.instance_of((str, type(None))),
#         default=None
#     )

#     mass: float = attrs.field(**DEFAULT_FLOAT_ATTRS)
#     radius: float = attrs.field(**DEFAULT_FLOAT_ATTRS)
#     semi_major_axis: float = attrs.field(**DEFAULT_FLOAT_ATTRS)
#     eccentricity: float = attrs.field(**DEFAULT_FLOAT_ATTRS)
#     inclination: float = attrs.field(**DEFAULT_FLOAT_ATTRS)
#     mean_anomaly: float = attrs.field(**DEFAULT_FLOAT_ATTRS)
#     argument_of_pericenter: float = attrs.field(**DEFAULT_FLOAT_ATTRS)
#     longitude_of_ascending_node: float = attrs.field(**DEFAULT_FLOAT_ATTRS)

#     # Calculated properties
#     _longitude_of_pericenter: float = attrs.field(init=False, default=None)
#     _mean_longitude: float = attrs.field(init=False, default=None)

#     star_name: str = attrs.field(
#         validator=attrs.validators.instance_of((str, type(None))),
#         default=None
#     )

#     metadata: dict = attrs.field(
#         validator=attrs.validators.instance_of(dict), default={}
#     )

#     def __attrs_post_init__(self):
#         """Post-initialization hook."""
#         return

#     @property
#     def longitude_of_pericenter_(self):
#         """Calculate and return the longitude of pericenter."""
#         if (
#             self.argument_of_pericenter is None
#             or self.longitude_of_ascending_node is None
#         ):
#             raise TypeError(
#                 "Longitude of pericenter calculation requires "
#                 + "argument_of_pericenter and longitude_of_ascending_node."
#             )
#         if self._longitude_of_pericenter is None:
#             self._longitude_of_pericenter = (
#                 self.argument_of_pericenter +
#                 self.longitude_of_ascending_node
#             ) % 360
#         return self._longitude_of_pericenter

#     @property
#     def mean_longitude_(self):
#         """Calculate and return the mean longitude."""
#         if (
#             self.mean_anomaly is None
#             or self.argument_of_pericenter is None
#             or self.longitude_of_ascending_node is None
#         ):
#             raise TypeError(
#                 "Mean longitude calculation requires mean_anomaly, "
#                 + "argument_of_pericenter, and longitude_of_ascending_node."
#             )
#         if self._mean_longitude is None:
#             self._mean_longitude = (
#                 self.mean_anomaly
#                 + self.argument_of_pericenter
#                 + self.longitude_of_ascending_node
#             ) % 360
#         return self._mean_longitude

# NEW CODE


@attrs.define(repr=False, frozen=True)
class StaticPlanet(ResokitDataFrame):
    """
    StaticPlanet class representing a static planet.

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
        """Default value for name."""
        return self.data_df["name"]  # ["name"] because .name is a df method

    @user_defined_.default
    def _user_defined__default(self):
        """Default value for user_defined_."""
        return self.source not in ["eu", "nasa"]

    @suffix_.default
    def _suffix__default(self):
        """Default value for suffix_."""
        return self.data_df["name"].split(" ")[-1]

    def __attrs_post_init__(self):
        """Post-initialization hook."""
        # Assert data_series is a Series
        if not isinstance(self.data_df, pd.Series):
            raise TypeError(
                "StaticPlanet must have a pd.Series. "
                + f"Got: {type(self.data_df)} instead."
            )
        if not self.user_defined_:
            for col in self.data_df.index:
                if col not in RESO_PL_TYPES.keys() | RESO_OB_TYPES.keys() | {
                    "star_name"
                }:
                    warnings.warn(
                        "Found columns not in the default planet mapping."
                    )

    def __getitem__(self, slice):
        """x[y] <==> x.__getitem__(y)."""
        if isinstance(slice, int) or (  # integer key indexing
            isinstance(slice, (list, tuple))  # list or tuple
            and all(isinstance(i, int) for i in slice)  # of integers
        ):
            raise IndexError(
                "StaticPlanet does not support integer indexing. "
                + "Use the 'name' column instead."
            )
        if isinstance(slice, tuple) and all(isinstance(i, str) for i in slice):
            warnings.warn(
                "StaticPlanet does not support multi-column indexing. "
                + "Use [[name1, name2, ...]] instead."
            )
        return super().__getitem__(slice)

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""
        return (
            f"StaticPlanet [{self.data_df['name']}]"
            + f" from {self.source} data source."
            if not self.user_defined_
            else "user defined."
        )


@attrs.define(repr=False, frozen=True)
class StaticStar(ResokitDataFrame):
    """
    StaticStar class representing a static star.

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
        """Default value for name."""
        if "star_name" in self.data_df.index:
            return self.data_df["star_name"]
        return self.data_df["name"]  # ["name"] because .name is a df method

    @user_defined_.default
    def _user_defined__default(self):
        """Default value for user_defined_."""
        return self.source not in ["eu", "nasa"]

    def __attrs_post_init__(self):
        """Post-initialization hook."""
        # Assert data_series is a Series
        if not isinstance(self.data_df, pd.Series):
            raise TypeError(
                "StaticStar must have a pd.Series. "
                + f"Got: {type(self.data_df)} instead."
            )
        if not self.user_defined_:
            AUX_COLS = {
                col.replace("star_", "") for col in RESO_SR_TYPES.keys()
            }
            for col in self.data_df.index:
                if col not in AUX_COLS | RESO_OB_TYPES.keys():
                    warnings.warn(
                        "Found columns not in the default star mapping."
                    )
                    print(col)

    def __getitem__(self, slice):
        """x[y] <==> x.__getitem__(y)."""
        if isinstance(slice, int) or (  # integer key indexing
            isinstance(slice, (list, tuple))  # list or tuple
            and all(isinstance(i, int) for i in slice)  # of integers
        ):
            raise IndexError(
                "StaticStar does not support integer indexing. "
                + "Use the 'name' column instead."
            )
        return super().__getitem__(slice)

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""
        return (
            f"StaticStar [{self.name}]" + f" from {self.source} data source."
            if not self.user_defined_
            else "user defined."
        )


@attrs.define(repr=False, frozen=True)
class StaticSystem:
    """
    StaticSystem class representing a static system.

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

    @n_planets_.default
    def _n_planets__default(self):
        """Default value for n_planets_."""
        return len(self.planets)

    @source_.default
    def _source__default(self):
        """Default value for source_."""
        main_source = getattr(self.star, "source", "unknown")
        return (
            main_source
            if all(
                [
                    getattr(planet, "source", "unknown") == main_source
                    for planet in self.planets
                ]
            )
            else "user"
        )

    @user_defined_.default
    def _user_defined__default(self):
        """Default value for user_defined_."""
        return self.source_ not in ["eu", "nasa"]

    @planet_names_.default
    def _planet_names__default(self):
        """Default value for planet_names_."""
        return [getattr(planet, "name") for planet in self.planets]

    def __attrs_post_init__(self):
        """Post-initialization hook."""
        star_name = self.star.name
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
                    + f" is different from Star({star_name})."
                )
        if self.n_planets_ != len(set(self.planet_names_)):
            warnings.warn("Planets must have unique names.")
        return

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""
        star_msg = "\n" + f" Star: {self.star.name}"
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
            + f" from {self.source_} data source."
            if not self.user_defined_
            else ""
        )

    def __getitem__(self, slice):
        """x[y] <==> x.__getitem__(y)."""
        if isinstance(slice, int) or (
            isinstance(slice, (list, tuple))
            and all(isinstance(i, int) for i in slice)
        ):
            sliced = self.planets[slice]
            # Return a new StaticPlanet ???
            # return StaticPlanet(
            #     data_df=sliced.data_df,
            #     source=sliced.source,
            #     metadata=dict(sliced.metadata),
            # )
            return sliced
        return self.get_system_items(slice)

    def __len__(self):
        """len(x) <=> x.__len__()."""
        return self.n_planets_ + 1

    def planet(self, idx: int = 0) -> StaticPlanet:
        """Return the specified planet"""
        return self.planets[idx]

    def _get_star_items(self, items: Union[list[str], str]):
        """Return the specified value items of the star."""
        if isinstance(items, str):
            items = [items]
        items = [item.replace("star_", "") for item in items]
        return [self.star[item] for item in items]

    def _get_planets_items(
        self, items: Union[list[str], str], values: bool = True
    ):
        """Return the specified value items of the planets."""
        if isinstance(items, str):
            items = [items]
        lista = [planet[items] for planet in self.planets]
        if values:
            return [item.values[0] for item in lista]
        return lista

    def get_system_items(self, items: Union[list[str], str]):
        """Return the specified items of the system."""
        if isinstance(items, str):
            items = [items]
        # If only one item, return the value
        if len(items) == 1:
            item = items[0]
            if item.startswith("star_"):
                return self.star[item.replace("star_", "")]
            if self.n_planets_ > 1:
                return pd.Series(
                    self._get_planets_items(item),
                    index=self.planet_names_,
                    name=item,
                )
            return self.planets[0][item]
        # If only 1 planet, return a Series
        if self.n_planets_ == 1:
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
        # If multiple planets, return a DataFrame
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
        for col in df.columns:
            if col.startswith("star_"):
                df[col] = self.star[col.replace("star_", "")]
        return df

    def remove_planet(self, index: int | str, verbose: bool = True):
        """
        Remove a planet from the system.

        Parameters
        ----------
        index : int, str
            Index or suffix (1 char) or name of the planet to remove.
        """
        if isinstance(index, str):
            if len(index) == 1:
                indexes = [planet.suffix_ for planet in self.planets]
                if len(indexes) != len(set(indexes)):
                    raise ValueError("Suffixes must be unique to remove.")
                if index not in indexes:
                    raise ValueError(f"Suffix '{index}' not found in planets.")
                index = indexes.index(index)
            else:
                if self.n_planets_ != len(set(self.planet_names_)):
                    raise ValueError(
                        "Planets must have unique names to remove."
                    )
                if index not in self.planet_names_:
                    raise ValueError(f"Name '{index}' not found in planets.")
                index = self.planet_names_.index(index)
        if index < 0 or index >= self.n_planets_:
            raise IndexError("Index out of range.")
        new_planets = self.planets[:index] + self.planets[index + 1:]
        new_meta = dict(self.metadata)
        if "removed_planet" not in new_meta:
            new_meta["removed_planet"] = self.planets[index].name
        else:
            new_meta["removed_planet"] += f", {self.planets[index].name}"
        ss = StaticSystem(
            star=self.star,
            planets=new_planets,
            name=self.name,
            metadata=new_meta,
        )
        if verbose:
            print(f"Planet {self.planets[index].name} [{index}] removed.")
        return ss

    def add_planet(
        self, planet: StaticPlanet, sort: bool = True, verbose: bool = True
    ):
        """
        Add a planet to the system.

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
        new_planets = self.planets + [planet]
        if sort:
            new_planets = sorted(new_planets, key=lambda x: x["P"])
        new_meta = dict(self.metadata)
        if "added_planet" not in new_meta:
            new_meta["added_planet"] = planet.name
        else:
            new_meta["added_planet"] += f", {planet.name}"
        ss = StaticSystem(
            star=self.star,
            planets=new_planets,
            name=self.name,
            metadata=new_meta,
        )
        if verbose:
            print(f"Planet {planet.name} added.")
        return ss

    def get_period_ratio(
        self, pair: Union[list, tuple, str] = "all", fraction_error: float = 0
    ) -> Union[float, pd.DataFrame]:
        """
        Return the period ratio of the planets.

        Parameters
        ----------
        pair : list, tuple, str, optional. Default: 'all'.
            Which pair of planets to consider.
            Either 'all' or a list/tuple of planet names/indexes.
        fraction_error : float, optional. Default: 0.
            If > 0, return the fraction with error lower than this value.

        Returns
        -------
        float, pd.DataFrame
            Period ratio of the planets
        """
        if self.n_planets_ < 2:
            raise ValueError("There must be at least 2 planets to compare.")
        if isinstance(pair, str):
            if not pair == "all":
                raise ValueError("Invalid pair value.")
            if self.n_planets_ == 2:
                return self.planets[0].P / self.planets[1].P
            # Create a DataFrame with all the period ratios
            periods = self.get_system_items("P")
            df = pd.DataFrame(
                [[p1 / p2 for p2 in periods] for p1 in periods],
                index=periods.index,
                columns=periods.index,
            )
            if abs(fraction_error):
                return df.map(
                    lambda x: float_to_fraction(x, max_error=fraction_error)
                )
            return df
        if len(pair) != 2:
            raise ValueError("Pair must have 2 elements.")
        if all(isinstance(i, int) for i in pair):
            ratio = self.planets[pair[0]].P / self.planets[pair[1]].P
        elif all(isinstance(i, str) for i in pair):
            ratio = self.get_system_items("P").loc[pair[0], pair[1]]
        else:
            raise ValueError("Invalid pair value.")
        if abs(fraction_error):
            return float_to_fraction(ratio, max_error=fraction_error)
        return ratio

    def to_dataframe(self, columns=None, copy=False) -> pd.DataFrame:
        """
        Return the data_df as a new DataFrame.

        Parameters
        ----------
        columns : list, optional. Default: None.
            Columns to return.
        copy : bool, optional. Default: False.
            Whether to return a copy of the DataFrame.
        """
        df = pd.DataFrame()
        for planet in self.planets:
            df = df.append(planet.data_df)
        df = pd.concat([self.star.data_df, df], axis=0)
        if columns is not None:
            used_cols = [col for col in columns if col in df.columns]
            df = df[used_cols]
        return df


# =============================================================================
# NEW FUNCTIONS
# =============================================================================


def _create_static_system(
    star,
    planets,
    name,
    metadata={},
) -> StaticSystem:
    """
    Create a StaticSystem instance.

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
    return StaticSystem(
        star=star,
        planets=planets,
        name=name,
        metadata=metadata,
    )


def _create_static_star(
    star_data,
    source="user",
    metadata={},
) -> StaticStar:
    """
    Create a StaticStar instance.

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
    return StaticStar(data_df=star_data, source=source, metadata=metadata)


def _create_static_planet(
    planet_data,
    source="user",
    metadata={},
) -> StaticPlanet:
    """
    Create a StaticPlanet instance.

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
    return StaticPlanet(data_df=planet_data, source=source, metadata=metadata)


# def resokit_to_planet(
#     resokit_data: ResokitDataFrame,
#     row: int = 0,
# ) -> StaticPlanet:
#     """
#     Convert a ResokitDataFrame to a StaticPlanet instance.
#     Not usable at the moment.

#     Parameters
#     ----------
#     resokit_data : ResokitDataFrame
#         ResokitDataFrame instance.
#     row : int, optional. Default: 0.
#         Row index to convert.
#     Returns
#     -------
#     StaticPlanet
#         StaticPlanet instance.
#     """
#     if not isinstance(resokit_data, ResokitDataFrame):
#         raise TypeError(
#             "resokit_data must be a ResokitDataFrame instance."
#             + f" Got: {type(resokit_data)} instead."
#         )

#     # Get planet columns df from resokit
#     columns = RESO_PL_TYPES.keys() | RESO_OB_TYPES.keys() | {"star_name"}
#     planet_df = resokit_data.to_dataframe(columns=columns)
#     source = resokit_data.source
#     meta = resokit_data.to_dict()

#     # Get the row if multiple lines
#     if len(resokit_data) > 1:
#         planet_df = planet_df.iloc[row]

#     return _create_static_planet(
#         planet_data=planet_df, source=source, metadata=meta
#     )


# def resokit_to_star(
#     resokit_data: ResokitDataFrame,
#     row: int = 0,
# ) -> StaticStar:
#     """
#     Convert a ResokitDataFrame to a StaticStar instance.
#     Not usable at the moment.

#     Parameters
#     ----------
#     resokit_data : ResokitDataFrame
#         ResokitDataFrame instance.
#     row : int, optional. Default: 0.
#         Row index to convert.
#         None to get the row with the most recent rowupdate date.

#     Returns
#     -------
#     StaticStar
#         StaticStar instance.
#     """
#     # Get planet columns df from resokit
#     cols = RESO_SR_TYPES.keys() | RESO_OB_TYPES.keys()
#     columns = {col.replace("star_", "") for col in cols}
#     star_df = resokit_data.to_dataframe(columns=columns)
#     source = resokit_data.source
#     meta = resokit_data.to_dict()

#     # Get the row if multiple lines.
#     # Use the most recent rowupdate date, if row not specified
#     if len(resokit_data) > 1:
#         if row < 0 or row is None:
#             star_df["rowupdate"] = pd.to_datetime(star_df["rowupdate"])
#             row = star_df["rowupdate"].idxmax()
#         star_df = star_df.iloc[row]

#     return _create_static_star(
#       star_data=star_df, source=source, metadata=meta
#     )


def resokit_to_system(
    resokit_data: ResokitDataFrame,
) -> StaticSystem:
    """
    Convert a ResokitDataFrame to a StaticSystem instance.

    Parameters
    ----------
    resokit_data : ResokitDataFrame
        ResokitDataFrame instance.

    Returns
    -------
    StaticSystem
        StaticSystem instance.
    """
    columns = resokit_data.columns_
    resokit_df = resokit_data.to_dataframe()

    # Stars
    aux_star_cols = RESO_SR_TYPES.keys() | RESO_OB_TYPES.keys()
    star_cols = list(set(aux_star_cols).intersection(columns))
    star_df = resokit_df[star_cols]

    # Planets
    aux_planet_cols = (
        RESO_PL_TYPES.keys() | RESO_OB_TYPES.keys() | {"star_name"}
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

        # If multiple lines (i.e. multiple planets), then create a star
        # from the star_df line with less null or NaN values
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
    if resokit_data.n_objects_ > 1:
        planets = [
            _create_static_planet(
                planet_data=planet,
                source=resokit_data.source,
                metadata=resokit_data.metadata,
            )
            for _, planet in planet_df.iterrows()
        ]
    else:
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
