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

import attrs
import warnings

# =============================================================================
# CONSTANTS
# =============================================================================

# Default attributes for float fields
DEFAULT_FLOAT_ATTRS = {
    "validator": attrs.validators.instance_of((int, float, type(None))),
    "default": None,
}

# ============================================================================
# CLASSES
# ============================================================================


@attrs.define(repr=False)
class Planet:
    """
    Planet class representing a planet with various attributes.

    Attributes
    ----------
    name : str
        Name of the planet.
    mass : float
        Mass of the planet in Jupiter masses.
    radius : float
        Radius of the planet in Jupiter radii.
    semi_major_axis : float
        Semi-major axis of the planet's orbit in AU.
    eccentricity : float
        Eccentricity of the planet's orbit.
    inclination : float
        Inclination of the planet's orbit in degrees.
    mean_anomaly : float
        Mean anomaly of the planet in degrees.
    argument_of_pericenter : float
        Argument of pericenter in degrees.
    longitude_of_ascending_node : float
        Longitude of the ascending node in degrees.
    star_name : str
        Name of the star the planet orbits.
    metadata : dict
        Additional metadata about the planet.
    """

    name: str = attrs.field(
        validator=attrs.validators.instance_of((str, type(None))), default=None
    )

    mass: float = attrs.field(**DEFAULT_FLOAT_ATTRS)
    radius: float = attrs.field(**DEFAULT_FLOAT_ATTRS)
    semi_major_axis: float = attrs.field(**DEFAULT_FLOAT_ATTRS)
    eccentricity: float = attrs.field(**DEFAULT_FLOAT_ATTRS)
    inclination: float = attrs.field(**DEFAULT_FLOAT_ATTRS)
    mean_anomaly: float = attrs.field(**DEFAULT_FLOAT_ATTRS)
    argument_of_pericenter: float = attrs.field(**DEFAULT_FLOAT_ATTRS)
    longitude_of_ascending_node: float = attrs.field(**DEFAULT_FLOAT_ATTRS)

    # Calculated properties
    longitude_of_pericenter_: float = attrs.field(init=False, default=None)
    mean_longitude_: float = attrs.field(init=False, default=None)

    star_name: str = attrs.field(
        validator=attrs.validators.instance_of((str, type(None))), default=None
    )

    metadata: dict = attrs.field(
        validator=attrs.validators.instance_of(dict), default={}
    )

    def __attrs_post_init__(self):
        """Post-initialization hook."""
        return

    @property
    def longitude_of_pericenter(self):
        """Calculate and return the longitude of pericenter."""
        if (
            self.argument_of_pericenter is None
            or self.longitude_of_ascending_node is None
        ):
            raise TypeError(
                "Longitude of pericenter calculation requires "
                + "argument_of_pericenter and longitude_of_ascending_node."
            )
        if self.longitude_of_pericenter_ is None:
            self.longitude_of_pericenter_ = (
                self.argument_of_pericenter + self.longitude_of_ascending_node
            ) % 360
        return self.longitude_of_pericenter_

    @property
    def mean_longitude(self):
        """Calculate and return the mean longitude."""
        if self.mean_longitude_ is not None:
            return self.mean_longitude_
        if (
            self.mean_anomaly is None
            or self.argument_of_pericenter is None
            or self.longitude_of_ascending_node is None
        ):
            raise TypeError(
                "Mean longitude calculation requires mean_anomaly, "
                + "argument_of_pericenter, and longitude_of_ascending_node."
            )
        self.mean_longitude_ = (
            self.mean_anomaly
            + self.argument_of_pericenter
            + self.longitude_of_ascending_node
        ) % 360
        return self.mean_longitude_

    @property
    def has_star_name(self):
        """Check if the planet has a star name."""
        return self.star_name is not None

    def __repr__(self):
        """String representation of the Planet instance."""
        if self.has_star_name:
            return f"Planet[{self.name}] orbiting Star[{self.star_name}]"
        return f"Planet[{self.name}]"

    def __len__(self):
        """Length of the Planet instance, always 1."""
        return 1


@attrs.define(repr=False)
class Star:
    """
    Star class representing a star with various attributes.

    Attributes
    ----------
    name : str
        Name of the star.
    mass : float
        Mass of the star in solar masses.
    radius : float
        Radius of the star in solar radii.
    effective_temperature : float
        Effective temperature of the star in Kelvin.
    luminosity : float
        Luminosity of the star in solar luminosities.
    age : float
        Age of the star in Gyr.
    system_name : str
        Name of the system the star belongs to.
    metadata : dict
        Additional metadata about the star.
    """

    name: str = attrs.field(
        validator=attrs.validators.instance_of((str, type(None))), default=None
    )

    mass: float = attrs.field(**DEFAULT_FLOAT_ATTRS)
    radius: float = attrs.field(**DEFAULT_FLOAT_ATTRS)
    effective_temperature: float = attrs.field(**DEFAULT_FLOAT_ATTRS)
    luminosity: float = attrs.field(**DEFAULT_FLOAT_ATTRS)
    age: float = attrs.field(**DEFAULT_FLOAT_ATTRS)

    system_name: str = attrs.field(
        validator=attrs.validators.instance_of((str, type(None))), default=None
    )

    metadata: dict = attrs.field(
        validator=attrs.validators.instance_of(dict), default={}
    )

    def __repr__(self):
        """String representation of the Star instance."""
        return f"Star[{self.name}]"

    def __len__(self):
        """Length of the Star instance, always 1."""
        return 1


@attrs.define(repr=False)
class System:
    """
    System class representing a planetary system with a central star
    and bodies.

    Attributes
    ----------
    name : str
        Name of the system.
    star : Star
        Central star of the system.
    bodies : list
        List of planets and other bodies in the system.
    """

    name: str = attrs.field()
    star: Star = attrs.field(validator=attrs.validators.instance_of(Star))
    bodies: list = attrs.field(
        validator=attrs.validators.instance_of((list, tuple, Star, Planet))
    )

    def __attrs_post_init__(self):
        """Post-initialization hook."""
        if isinstance(self.bodies, (list, tuple)):
            for body in self.bodies:
                if not isinstance(body, (Star, Planet)):
                    raise TypeError(
                        "bodies must be a list of Star or Planet instances. "
                        + f"Got: {type(body)} instead"
                    )
        if isinstance(self.bodies, (Star, Planet)):
            self.bodies = [self.bodies]
        if len(self.bodies) == 0:
            raise ValueError(
                "System must have at least one body "
                + "(apart from the central star)."
            )
        if (
            self.star.system_name is not None
            and self.star.system_name != self.name
        ):
            warnings.warn(
                f"Star({self.star.name}) system name is different from "
                + f"System({self.name})."
            )
        return

    def __repr__(self):
        """String representation of the System instance."""
        return f"System[{self.name}]"

    def __len__(self):
        """Length of the System instance, number of bodies plus the star."""
        return len(self.bodies) + 1


# =============================================================================
# FUNCTIONS
# =============================================================================


def create_planet(
    name,
    mass=None,
    radius=None,
    period=None,
    semi_major_axis=None,
    eccentricity=None,
    inclination=None,
    mean_anomaly=None,
    argument_of_pericenter=None,
    longitude_of_ascending_node=None,
    star_name=None,
    metadata=None,
):
    """
    Create a Planet instance.

    Parameters
    ----------
    name : str
        Name of the planet.
    mass : float
        Mass of the planet in Jupiter masses.
    radius : float
        Radius of the planet in Jupiter radii.
    period : float
        Orbital period of the planet in days.
    semi_major_axis : float
        Semi-major axis of the planet's orbit in AU.
    eccentricity : float
        Eccentricity of the planet's orbit.
    inclination : float
        Inclination of the planet's orbit in degrees.
    mean_anomaly : float
        Mean anomaly of the planet in degrees.
    argument_of_pericenter : float
        Argument of pericenter in degrees.
    longitude_of_ascending_node : float
        Longitude of the ascending node in degrees.
    star_name : str
        Name of the star the planet orbits.
    metadata : dict
        Additional metadata about the planet.

    Returns
    -------
    Planet
        A new Planet instance.
    """
    return Planet(
        name=name,
        mass=mass,
        radius=radius,
        semi_major_axis=semi_major_axis,
        eccentricity=eccentricity,
        inclination=inclination,
        mean_anomaly=mean_anomaly,
        argument_of_pericenter=argument_of_pericenter,
        longitude_of_ascending_node=longitude_of_ascending_node,
        star_name=star_name,
        metadata=metadata,
    )
