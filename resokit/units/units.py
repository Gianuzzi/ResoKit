#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   ResoKit Project (https://github.com/Gianuzzi/resokit).
# Copyright (c) 2025, Emmanuel Gianuzzi
# License: MIT
#   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Module with utilities for unit conversion and manipulation."""

# =============================================================================
# IMPORTS
# =============================================================================

from types import MappingProxyType
from typing import List, Tuple, Union

from numpy import pi

# =============================================================================
# CONSTANTS AND NORMALIZED UNIT DICTIONARIES
# =============================================================================

# Gravitational constant in SI units
G = 6.67430e-11  # m^3 kg^-1 s^-2

# UNITS dictionary
_units = {
    "mass": {
        "g": 1e-3,  # alias
        "gr": 1e-3,  # grams → kg
        "kg": 1.0,
        "ton": 1e3,  # metric ton → kg
        "me": 5.9722e24,  # Earth mass → kg
        "mj": 1.89813e27,  # Jupiter mass → kg
        "ms": 1.989e30,  # Solar mass → kg
    },
    "distance": {
        "cm": 1e-2,  # cm → m
        "m": 1.0,
        "km": 1e3,  # km → m
        "re": 6.371e6,  # Earth eq radius → m
        "rj": 6.9911e7,  # Jupiter radius → m
        "rs": 6.957e8,  # Solar radius → m
        "au": 1.495978e11,  # Astronomical unit → m
        "pc": 3.0857e16,  # Parsec → m
    },
    "time": {
        "sec": 1.0,
        "s": 1.0,  # alias
        "min": 60.0,  # min → s
        "hour": 3600.0,  # hour → s
        "day": 86400.0,  # day → s
        "year": 31557600.0,  # year → s
        "yr": 31557600,  # alias 2
    },
    "angle": {
        "rad": 1.0,
        "deg": pi / 180.0,
    },
}
# Add density units
_units["density"] = {
    "rhow": _units["mass"]["g"] / _units["distance"]["cm"] ** 3,
    "rhos": _units["mass"]["ms"]
    / (4.0 / 3.0 * pi * _units["distance"]["rs"] ** 3),
    "rhoj": _units["mass"]["mj"]
    / (4.0 / 3.0 * pi * _units["distance"]["rj"] ** 3),
    "rhoe": _units["mass"]["me"]
    / (4.0 / 3.0 * pi * _units["distance"]["re"] ** 3),
}

# Create immutable dict
UNITS = MappingProxyType(_units)

# Custom with MKS
_mks = {u: v for uv in UNITS.values() for u, v in uv.items()}
_mks["G"] = G

# Create dict to set MKS
MKS = MappingProxyType(_mks)


def _convert_units(from_unit, to_unit, power=1):
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    for unit_dict in UNITS.values():
        if from_unit in unit_dict and to_unit in unit_dict:
            origin = unit_dict[from_unit]
            dest = unit_dict[to_unit]
            return (origin / dest) ** power

    raise ValueError(
        f"Cannot convert '{from_unit}' to '{to_unit}': incompatible or unknown."
    )


def __normalize_units(u):
    if isinstance(u, str):
        return [u]
    return u


def __normalize_powers(p, n):
    if isinstance(p, int):
        return [p] * n
    if len(p) != n:
        raise ValueError(
            f"Power length {len(p)} does not match units length {n}."
        )
    return p


def convert(
    *values,
    from_units: Union[None, str, Tuple[str, ...], List[str]] = None,
    to_units: Union[None, str, Tuple[str, ...], List[str]] = None,
    power: Union[int, Tuple[int, ...], List[int]] = 1,
) -> Union[float, List[float]]:
    """
    Convert between compound units (e.g. km^2/s → m^2/s).

    Parameters
    ----------
    from_units : str or list of str
        Units to convert from.
    to_units : str or list of str
        Units to convert to.
    power : int or list of int, optional
        Power(s) for each unit (default = 1).
        Must match length if list.

    Examples
    --------
    To convert 5 km s^{-2} to au yr^{-2}:

    >>> resokit.units.convert(
    ...     5,
    ...     from_units=("km", "s"),
    ...     to_units=("au", "yr"),
    ...     power=(1, -2)
    ... )
    33247713.903743323

    Returns
    -------
    float or list of float
        Converted value, or list of converted values.
    """
    if from_units is None:
        raise ValueError("Argument 'from_units' must be set.")
    if to_units is None:
        raise ValueError("Argument 'to_units' must be set.")

    from_units = __normalize_units(from_units)
    to_units = __normalize_units(to_units)
    powers = __normalize_powers(power, len(from_units))

    if len(from_units) != len(to_units):
        raise ValueError("Mismatch between number of from_units and to_units.")

    factor = 1.0
    for fu, tu, pw in zip(from_units, to_units, powers):
        factor *= _convert_units(fu, tu, power=pw)

    if not values:
        return factor
    if len(values) == 1:
        return factor * values[0]
    return [value * factor for value in values]
