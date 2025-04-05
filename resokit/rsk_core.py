# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt
import attrs
from .tools import rmm


# =============================================================================
# UTILTY FUNCTIONS AND CLASSES
# =============================================================================


# constants
pi = np.pi
G = 4*pi**2 # in Msol, au, yr
E2S = 1/333000. # Mearth to Msol

# list of angles
class Angles(np.ndarray):
    # First two methods are required when subclassing np.ndarray
    def __new__(cls, angls,rad=False):
        units = 180/pi if rad else 1 
        ang_arr = np.asarray(angls)*units
        ang_arr = ang_arr%360
        return ang_arr.view(cls)

    def __add__(self, other_angls):
        return(np.ndarray.__add__(self,other_angls) % 360)
    
    def __sub__(self, other_angls):
        return(np.ndarray.__sub__(self,other_angls) % 360)
    
    @property
    def rad(self):
        return(self*pi/180)
    
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
    mass: float = attrs.field(default=None, validator=_validate_constant)  # earth masses
    radius: float = attrs.field(default=None, validator=_validate_constant)  # earth radii

    # Flags
    is_star: bool = attrs.field(default=False)  # is_star flag

    # Star reference
    star: "Star" = attrs.field(default=None)  # Reference to the system's star
    
    # System reference
    system: "DynamicSystem" = attrs.field(default=None)  # Reference to the system

    # Additional info
    name: str = attrs.field(default="", validator=attrs.validators.instance_of(str))  # planet's name

    # Calculated arrays
    _varpi: list = attrs.field(init=False, default=None, validator=_validate_sequence)  # pericenter longitude
    _lam: list = attrs.field(init=False, default=None)  # mean longitude
    _n: list = attrs.field(init=False,default=None) # mean motion
    
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
        
    @property
    def n(self):
    	if not _everything_exists(self.star.mass, self.a):
    	    raise TypeError(
                "'w' and 'Omega' are \
                            required to calculate 'varpi'"
            )
    	elif self.mass is None:
    	    mu = G*self.star.mass
    	    raise Warning('Assuming m_planet = 0 for mean-motion calculation')
    	elif self.mass is not None:
    	    mu = G*(self.star.mass + self.mass*E2S)
    	return np.sqrt(mu / (self.a**3))
  
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
    nsteps: int = attrs.field(init=False)
    
    # time array
    times: list = attrs.field(init=False, default=None, **_TIME_SEQ_VALID_AND_CONV)
    
    resangs: list = attrs.field(default=None, validator=_validate_sequence)
    
    def __attrs_post_init__(self):
        self.times = self.planets[0].times
        self.npl = len(self.planets)
        self.nsteps = len(self.times)

    def Prat(self,which=None):
        """
        Calculate period ratios between pairs of planets.
        
        If which is None: Return period ratios from all pairs of adjacent 
                          planets.
        If which is a number i: Return the period ratio from only the i-th
                            pair of adjacent planets.
        If which is a list of two numbers [i,j]: Return the period ratio 
                                                 between the ith- and 
                                                 jth-planet.
        

        Parameters
        ----------
        which : None, int or list of two numbers, optional
            Which planets to use. The default is None.

        Returns
        -------
        prat : List of floats, or list of lists of floats.
            Period ratios.

        """
        # P2/P1 = sqrt((M+m1)/(M+m2) * (a2/a1)**3)
        
        # Masses
        masses = np.asarray([pli.mass for pli in self.planets])
        if not all(masses):
            # if pl_masses = 0, then (M+m1)/(M+m2) = M/M = 1
            # so we can take M+m1 = M+m2 = 1
            m_fac = 1
            Warning("Assuming mass=0 for planets in calculating prat")
        else:
            m_fac = self.star.mass + masses*E2S

        # Pick planets involved
        if which is None:  # all pairs of adjacent planets
            planets = self.planets
        elif len(np.shape(which))==0: # only the which-th pair
            planets = self.planets[which:which+2]
            m_fac = m_fac[which:which+2]
        else: # only between the ith- and jth-planet
            planets = [self.planets[which[0]],self.planets[which[1]]]
            m_fac = [m_fac[which[0]],m_fac[which[1]]]
        npl_ = len(planets)
        
        # sma
        a_l = [pli.a for pli in planets]
        
        # calc prat
        prat = np.zeros((npl_-1,self.nsteps))
        for i in range(npl_-1):
            prat[i] = (m_fac[i]/m_fac[i+1] * (a_l[i+1]/a_l[i])**3)**0.5
        
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

    def sepspace(self,bounds=None,which=None,ax=None,r3p_labels=True,**kwargs):
        """
        Plot periods ratios for planetary triplets.
        
        There are many ways to select the planets involved in the plotted
        triplets, using the 'which' parameter.
        
        Let's see examples for which:
            - which = None : Will use all triplets with adjacent planets.
                             This would correspond to Npl-2 triplets.
            - which = 2 : Will use only the third triplet of adjacent planets.
            - which = [0,2] : Will use the first and third triplet of
                              adjacent planets.
            - which = [[0,2,4],[0,1,2]] : Will use two triplets. The first one
                                          being composed of the first, third
                                          and fifth planet; and the second one
                                          would involve the first, second and
                                          third planet.

        Parameters
        ----------
        bounds : list of four floats, optional
            Set plot bounds. The default is None and will assign bounds barely
            farther than the plotted tracks.
        which : None, int, list of ints, or list of lists of ints, optional
            Select which planets are involved in the plotted triplets. 
            The default is None.
        ax : matplotlib.pyplot.Axes() object, optional
            Axis on which to plot. The default is None and will look for an
            existing ax or create one.
        r3p_labels : bool, optional
            If True, sets labels to indicate 3P-MMRs. The default is True.
        **kwargs : dict
            Kwargs for either the plot of points, or the MMR searching 
            function.

        Returns
        -------
        ´.

        """        
        
        # Organize kwargs
        rmm_kw_keys = ['r3p_order','r3p_maxint','r2p_order','r2p_maxint']
        rmm_kw = {k:kwargs[k] for k in rmm_kw_keys if k in kwargs}
        plot_kw = {k:v for k,v in kwargs.items() if k not in rmm_kw_keys}
        
        # cases por triplet selection
        case = len(np.shape(which))
        if which is None: # all triplets
            all_Prat = self.Prat()
            nnx = all_Prat[:-1]
            nny = all_Prat[1:]
            order=np.arange(self.npl - 2)+1
        elif case == 0:  # only the which-th triplet
            nnx = self.Prat(which=which)
            nny = self.Prat(which=which+1)
            order = [which+1]
        elif case == 1:  # many -th triplets
            nnx = []
            nny = []
            for pairi in which:
                nnx.append(self.Prat(which=pairi)[0])
                nny.append(self.Prat(which=pairi+1)[0])
            order = [i+1 for i in which]
        elif case == 2:  # triplets with individually selected planets
            nnx = []
            nny = []
            for tripi in which:
                nnx.append(self.Prat(which=[tripi[0],tripi[1]])[0])
                nny.append(self.Prat(which=[tripi[1],tripi[2]])[0])
            order = np.arange(len(which))+1  # here corresponds to the user-
                                             # given order
            
        if ax is None: ax = plt.gca()
        
        # Get or assign boundaries
        if bounds is None:
            l1x = np.min(nnx)*0.95
            l2x = np.max(nnx)*1.05
            l1y = np.min(nny)*0.95
            l2y = np.max(nny)*1.05
        else:
            l1x,l2x,l1y,l2y = bounds
        
        # Get map of MMRs inside bounds
        r3p,r2x,r2y = rmm.rmm_in_area(lims=[l1x,l2x,l1y,l2y],**rmm_kw)
        
        # Plot period ratios
        ntrips = len(nnx)
        for tripi in range(ntrips):
            label=f'{order[tripi]}° triplet'
            ax.scatter(nnx[tripi],nny[tripi],label=label,**plot_kw)
            # Plot starting points
            ax.scatter(nnx[tripi][0],nny[tripi][0],s=30,ec='k',c='none')
        # Put just one label for starting points
        ax.scatter(None,None,s=30,ec='k',c='none',label='Starting points')
        
        # Plot 2P-MMRs
        for r2xi in r2x:
            ax.axvline(r2xi[0]/r2xi[1],lw=.75,c='k',linestyle='dashed')
        for r2yi in r2y:
            ax.axhline(r2yi[0]/r2yi[1],lw=.75,c='k',linestyle='dashed')
        
        # Plot 3P-MMRs
        dom = np.linspace(l1x,l2x,1000)
        for r3i in r3p:
            ax.plot(dom,rmm.r3p(dom,r3i),lw=0.75,c='k')
            if r3p_labels:
                rmm.r3p_label(r3i, ax, [l1x,l2x,l1y,l2y])
        
        # Set lims
        plt.xlim(l1x,l2x)
        plt.ylim(l1y,l2y)
        
        plt.show()
