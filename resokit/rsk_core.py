# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt
import attrs


# =============================================================================
# UTILTY FUNCTIONS AND CLASSES
# =============================================================================


# list of angles
class Angles(np.ndarray):
    # First two methods are required when subclassing np.ndarray
    def __new__(cls, angls,rad=False):
        units = 180/np.pi if rad else 1 
        ang_arr = np.asarray(angls)*units
        ang_arr = ang_arr%360
        return ang_arr.view(cls)

    def __add__(self, other_angls):
        return(np.ndarray.__add__(self,other_angls) % 360)
    
    def __sub__(self, other_angls):
        return(np.ndarray.__sub__(self,other_angls) % 360)
    
    @property
    def rad(self):
        return(self*np.pi/180)
    
    @property
    def arr(self):
        return(np.asarray(self))
    
    def __repr__(self):
        # Truncate representation if array is large
        max_items = 10  # Maximum items to display before truncation
        if self.size > max_items:
            # Show the first and last few elements with ellipsis in the middle
            data = np.concatenate((self[:5],['...'], self[-5:]))
            return f"Angles([{', '.join(map(str, data))}])"
        else:
            return np.ndarray.__repr__(self)
    
    
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
    times: list  # yr
    a:     list  # AU
    e:     list  #
    inc:   list  # deg
    M:     list  # deg
    w:     list  # deg
    Omega: list  # deg

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
        init=False, default=None)  # mean longitude

    @property
    def varpi(self):
        if not _everything_exists(self.w, self.Omega):
            raise TypeError(
                "'w' and 'Omega' are \
                            required to calculate 'varpi'"
            )
        elif self._varpi is None:
            self._varpi = self.w + self.Omega
        return self._varpi

    @property
    def lam(self):
        if not _everything_exists(self.M, self.w, self.Omega):
            raise TypeError(
                "'M', 'w' and 'Omega' are\
                            required to calculate 'varpi'"
                            )
        elif self._lam is None:
            self._lam = self.M + self.w + self.Omega
        return self._lam
    
    def plot(self,var,ax=None,**plot_kw):
      if ax is None: ax=plt.gca()
      ax.plot(self.times,getattr(self,var),**plot_kw)
      return ax
  
    def scatter(self,var,ax=None,**scatter_kw):
      if ax is None: ax=plt.gca()
      ax.scatter(self.times,getattr(self,var),**scatter_kw)
      return ax

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

_PICK_RIGHT_VP_COEFS = 'sum(vp_coefs) has to equal order of MMR.'
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
    
    resangs: list = attrs.field(default=None, validator=_validate_sequence)
    
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
    
    
    
    def calc_2pmmr(self,num,den,ij=[0,1],vp_coefs=[0,0]):
        assert (num-den - sum(vp_coefs) == 0), _PICK_RIGHT_VP_COEFS
        
        i1,i2 = ij
        pl1 = self.planets[i1]
        pl2 = self.planets[i2]
        
        p = den
        q = num-den
        
        _2pmmr_ang = pl1.lam*p - pl2.lam*(p+q) \
                   + np.dot(vp_coefs,[pl1.varpi,pl2.varpi])
        
        return(_2pmmr_ang)
    
    def calc_3pmmr(self,k1,k2,k3,ijk=[0,1,2],vp_coefs=[0,0,0]):
        assert (self.npl>=3), "Not enough planets for a 3P-MMR."
        assert (k1+k2+k3 - sum(vp_coefs) == 0), _PICK_RIGHT_VP_COEFS
        
        i1,i2,i3 = ijk
        pl1 = self.planets[i1]
        pl2 = self.planets[i2]
        pl3 = self.planets[i3]

        _3pmmr_ang = pl1.lam*k1 + pl2.lam*k2 + pl3.lam*k3 \
                   + np.dot(vp_coefs,[pl1.varpi,pl2.varpi,pl3.varpi])
        return(_3pmmr_ang)
    
    def _plot_or_scatter_resangs(self,ax,method,which_resang,**any_kw):
        which = np.asarray(which_resang)
        which = [which] if which.ndim==0 else which
        for i in which:
            resangi = self.resangs[i]
            plotting_func = getattr(ax, method)
            plotting_func(self.times,resangi,**any_kw)
        return ax
    
    def plot(self,var,ax=None,which_resang=None,**plot_kw):
      if ax is None: ax=plt.gca()
      if var=='resangs':
          self._plot_or_scatter_resangs(ax,'plot',which_resang,**plot_kw)
          return ax
      ax.plot(self.times,getattr(self,var),**plot_kw)
      return ax
  
    def scatter(self,var,ax=None,which_resang=None,**scatter_kw):
      if ax is None: ax=plt.gca()
      if var=='resangs':
          self._plot_or_scatter_resangs(ax,'scatter',which_resang,**scatter_kw)
          return ax
      ax.scatter(self.times,getattr(self,var),**scatter_kw)
      return ax
