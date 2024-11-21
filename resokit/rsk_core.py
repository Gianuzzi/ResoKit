# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np
import attrs
import warnings


# ============================================================================
# UTILTY FUNCTIONS AND CLASSES
# ============================================================================


# list of angles
class Angles:
    def __new__(cls,angls):
        if isinstance(angls, Angles):
            return angls  # Return existing instance, skips new object creation
        return super().__new__(Angles)  # Create a new Angles instance

    def __init__(self, angls):
        # Avoid re-initializing if angls is already an Angles instance
        if isinstance(angls, Angles): return
        angls = np.asarray(angls)
        self.angls = angls % 360

    def __add__(self, other_angls):
        if isinstance(other_angls, Angles):
            return (self.angls + other_angls.angls) % 360
        warnings.warn('Added Angle type to non-Angle type')
        return self.angls + other_angls

    def __repr__(self):
        angls = np.asarray(self.angls)
        return str(angls)
    
    
# Custom validators
_ALLOWED_SEQS = (list,tuple,np.ndarray,Angles)
def _validate_sequence(instance, attribute, value):
    if value is not None and not isinstance(value, _ALLOWED_SEQS):
        raise TypeError(
            f"{attribute.name} must be a list, tuple, np.ndarray, \
                        or None."
        )


def _validate_constant(instance, attribute, value):
    if value is not None and not isinstance(value, (int, float)):
        raise TypeError(f"{attribute.name} must be an int, float, or None.")


# Converter to ensure sequences are converted to numpy arrays if not None
def _to_numpy_array(value):
    return np.array(value) if value is not None else None

def _to_Angles(value):
    return Angles(value) if value is not None else None


# Join validator and converter
_TIME_SEQ_VALID_AND_CONV = {
    "validator": _validate_sequence,
    "converter": _to_numpy_array,
}

_ANGL_TIME_SEQ_VALID_AND_CONV = {
    "validator": _validate_sequence,
    "converter": _to_Angles,
}


# Utility functions
def _exists(var):
    return var is not None


def _everything_exists(*variables):
    return all([_exists(var) for var in variables])


# ============================================================================
# MAIN CLASSES
# ============================================================================


@attrs.define(frozen=False, slots=True, repr=False)
class DynamicPlanet:
    """Time series for the evolution of a planet"""

    # Input time arrays
    times: list = attrs.field(default=None, **_TIME_SEQ_VALID_AND_CONV)  # yr
    a: list = attrs.field(default=None, **_TIME_SEQ_VALID_AND_CONV)  # AU
    e: list = attrs.field(default=None, **_TIME_SEQ_VALID_AND_CONV)  #
    inc: list = attrs.field(default=None, **_ANGL_TIME_SEQ_VALID_AND_CONV)  # deg
    M: list = attrs.field(default=None, **_ANGL_TIME_SEQ_VALID_AND_CONV)  # deg
    w: list = attrs.field(default=None, **_ANGL_TIME_SEQ_VALID_AND_CONV)  # deg
    Omega: list = attrs.field(default=None, **_ANGL_TIME_SEQ_VALID_AND_CONV)  # deg

    # Input constants
    mass: float = attrs.field(
        default=None, validator=_validate_constant
    )  # earth masses
    radius: float = attrs.field(
        default=None, validator=_validate_constant
    )  # earth radii

    # Flags
    is_star: bool = attrs.field(default=False)  # is_star flag

    # Additional info
    name: str = attrs.field(
        default="", validator=attrs.validators.instance_of(str)  # planets name
    )

    # Calculated arrays
    _varpi: list = attrs.field(
        init=False, default=None, validator=_validate_sequence
    )  # mean longitude
    _lam: list = attrs.field(
        init=False, default=None, validator=_validate_sequence
    )  # mean longitude

    @property
    def varpi(self):
        if not _everything_exists(self.w, self.Omega):
            raise TypeError(
                "'w' and 'Omega' are \
                            required to calculate 'varpi'"
            )
        elif self._varpi is None:
            self._varpi = (self.w + self.Omega) % 360
        return self._varpi

    @property
    def lam(self):
        if not _everything_exists(self.M, self.w, self.Omega):
            raise TypeError(
                "'M', 'w' and 'Omega' are\
                            required to calculate 'varpi'"
            )
        elif self._lam is None:
            self._lam = (self.M + self.w + self.Omega) % 360
        return self._lam

    def planet_method(self): ...


# ------------------------- STAR
@attrs.define(repr=False)
class Star:
    """
    Star of the system

    """

    mass: float = attrs.field(
        default=None, validator=_validate_constant
    )  # solar masses
    radius: float = attrs.field(
        default=None, validator=_validate_constant
    )  # solar radii

    def star_method(self): ...


_LIST_OF_PLANETS_VALID = attrs.validators.deep_iterable(
    member_validator=attrs.validators.instance_of(DynamicPlanet)
)


# ------------------------- DynamicSystem
@attrs.define(repr=False)
class DynamicSystem:
    """Global properties of the system"""

    # planets
    planets: list = attrs.field(default=[], validator=_LIST_OF_PLANETS_VALID)
    # star
    star: Star = attrs.field(
        default=Star(), validator=attrs.validators.instance_of(Star)
    )
    # parameters
    npl: int = attrs.field(init=False)
    nstars: int = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    # time array
    times: list = attrs.field(init=False, default=None, **_TIME_SEQ_VALID_AND_CONV)

    def __attrs_post_init__(self):
        self.times = self.planets[0].times
        self.npl = len(self.planets)

    def Prat(self,which=False):
        
        masses = np.asarray([pli.mass for pli in self.planets])
        if not all(masses):
            mu_l = 1
            Warning("Assuming mass=0 for planets in calculating prat")
        else:
            mu_l = self.star.mass + masses  # neglect factor G because of ratio
            
        if not which: planets = self.planets
        elif not np.shape(which): planets = self.planets[which:which+2]
        else: planets = [self.planets[which[0]],self.planets[which[1]]]
        nl = np.asarray([(mu_l / pli.a**3) ** 0.5 for pli in planets])
        prat = nl[:-1] / nl[1:]
        return prat






