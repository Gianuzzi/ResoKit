import numpy as np
import pandas as pd
import attrs

# ####--------------------- FORMATOS DE VARIABLES
_SEQUENCE_TYPES = (list, tuple, np.ndarray)
_TIME_SERIES_KWARGS = {
    # "validator": attrs.validators.instance_of(_SEQUENCE_TYPES),
    # "converter": np.asarray,
    "default": None,
}

_TABLE_KWARGS = {
    "validator": attrs.validators.instance_of(pd.DataFrame),
    "factory": pd.DataFrame,
}

_PLNAME_KWARGS = {
    "validator": attrs.validators.instance_of(str),
    "default": "",
}

_CONSTANT_KWARGS = {
    "validator": attrs.validators.instance_of((int,float,type(None))),
    "default": np.nan,
}

_BOOL_KWARGS = {
    "validator": attrs.validators.instance_of(bool),
    "default": False,
}

_INTEGER_KWARGS = {
    "validator": attrs.validators.instance_of(int),
    "default": 0,
}


#####--------------------- DynamicPlanet
@attrs.define(frozen=False, slots=True, repr=False)
class DynamicPlanet:
    """Time series for the evolution of a planet"""
    
    # input parameters
    
    time_series_table: pd.DataFrame = attrs.field(**_TABLE_KWARGS)

    times: list = attrs.field(**_TIME_SERIES_KWARGS)  # days
    a: list = attrs.field(**_TIME_SERIES_KWARGS)  # AU
    e: list = attrs.field(**_TIME_SERIES_KWARGS)  #
    inc: list = attrs.field(**_TIME_SERIES_KWARGS)  # deg
    m_an: list = attrs.field(**_TIME_SERIES_KWARGS)  # M , deg
    w: list = attrs.field(**_TIME_SERIES_KWARGS)  # deg
    Omega: list = attrs.field(**_TIME_SERIES_KWARGS)  # deg
    
    mass: float = attrs.field(**_CONSTANT_KWARGS)  # earth masses
    radius: float = attrs.field(**_CONSTANT_KWARGS)  # earth radii

    is_star: bool = attrs.field(**_BOOL_KWARGS) # is_star flag

    name: str = attrs.field(**_PLNAME_KWARGS) # planets name
    
    M : list = attrs.field(**_TIME_SERIES_KWARGS)  # mean anomaly, only for usr
    lam : list = attrs.field(**_TIME_SERIES_KWARGS) # mean longitude
    
    def __attrs_post_init__(self):
        if self.m_an is not None: self.M = self.m_an
        if self.M is not None and \
           self.w is not None and \
           self.Omega is not None:
            lam = (self.M+self.w+self.Omega)%360
            self.lam = lam

    def planet_method(self): ...


#####--------------------- STAR
@attrs.define(repr=False)
class Star:
    """
    Star of the system

    """

    mass: float = attrs.field(init=False,**_CONSTANT_KWARGS)  # solar masses
    radius: float = attrs.field(init=False,**_CONSTANT_KWARGS)  # solar radii
    teff: float = attrs.field(init=False,**_CONSTANT_KWARGS)  # kelvin
    met: float = attrs.field(init=False,**_CONSTANT_KWARGS)

    def star_method(self): ...


#####--------------------- DynamicSystem
#####--------------------- FORMATO DE VARIABLES
_STAR_OBJECT_KWARGS = {
    "validator": attrs.validators.instance_of(Star),
    "default": Star(),  # empty star
}

_PLANET_OBJECT_KWARGS = {
    "validator": attrs.validators.instance_of((list, tuple, np.ndarray)),
    "default": [np.nan],
}


@attrs.define(repr=False)
class DynamicSystem:
    """Global properties of the system"""

    nstars: int = attrs.field(**_INTEGER_KWARGS)
    star: Star = attrs.field(**_STAR_OBJECT_KWARGS)
    planets: list = attrs.field(**_PLANET_OBJECT_KWARGS)
    npl: int = attrs.field(init=False, **_INTEGER_KWARGS)

    def __attrs_post_init__(self):
        """Validate types of star and planets,
        AND add calculated attributes.

        """
        # Validation
        if not isinstance(self.star, Star):
            raise TypeError("star must be of type Star")

        for ith, planet in enumerate(self.planets):
            if not isinstance(planet, DynamicPlanet):
                raise TypeError(
                    f"planet {ith+1} must be of type DynamicPlanet"
                )

        # Additional calculated attributes
        self.npl = len(self.planets)
        
    def system_method(self): ...
