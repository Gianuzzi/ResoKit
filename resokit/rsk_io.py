# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import pandas as pd
from .rsk_core import DynamicPlanet, Star, DynamicSystem, Angles

# =============================================================================
# CONSTANTS
# =============================================================================

# allowed inputs for columns of integration file
ELEM_SPACE = {
    "times": None,
    "ibody": None,
    "a": None,
    "e": None,
    "inc": None,
    "M": None,
    "w": None,
    "Omega": None,
    "_": None,
    "mass": None,
    "resangs": None,
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _try_getting(df, key):
    """Retrieve a column's values or return None."""
    return df[key].values if key in df.columns else None

def _to_Angles(angls):
    """Convert list to Angles or return None."""
    return None if angls is None else Angles(angls)

def _separate_resangs(resangs,npl):
    """Separate resonance angles by planet."""
    return None if resangs is None else [resangs[i::npl] for i in range(npl)]

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def load_integration(
    file,
    npl,
    names=["times", "ibody", "a", "e", "inc", "M", "w", "Omega"],
    sep_files=False,
    mass=None,
    radius=None,
    usecols=None,
    plnames=None,
    is_star=None,
    st_m=None,
    st_r=None,
):
    """
    Load integration from file.

    Parameters
    ----------
    file : str
        Path to file. If sep_files is True, this parameter should be the names
        of the individual files, with an asterisk replacing the body id,
        eg. " file='planet*.dat' ".
    npl : int
        Number of planets.
    names : list of str, optional
        Contents of the columns. Can only include the terms ["times", "ibody",
        "a", "e", "inc", "M", "w", "Omega", "mass", "_"]. Use "_" for
        throwaways. The default is ["times", "ibody", "a", "e", "inc", "M",
        "w", "Omega"].
    sep_files: bool, optional
        Should be true if data is scattered in a one-file-per-body manner.
    mass : list of floats, optional
        Planet masses in Earth masses.
        The default is None.

    Returns
    -------
    planets : list of dicts
        Data separated per planet per element.

    """

    # =============== VALIDATE PARAMETERS =============== #
    for namei in names:
        assert (
            namei in ELEM_SPACE
        ), f"{namei} is not an \
    allowed input"

    if (usecols is not None) and (len(names) != len(usecols)):
        raise Exception("usecols doesn't match names length")

    # correct use of mass
    if ("mass" in names) and (mass is not None):
        raise Exception("can't input mass twice")

    # is_star flag
    if is_star != None:
        if isinstance(is_star, (int, float)) and is_star > 0:
            is_star_list = [False] * npl
            is_star_list[is_star - 1] = True
            is_star = is_star_list
        elif isinstance(is_star, (list, tuple, np.ndarray)) and (0 not in is_star):
            is_star_list = np.array([False] * (npl + 1))
            is_star_list = [True if i in is_star else False for i in range(npl + 1)]
            is_star = is_star_list[1:]
        else:
            raise Exception("bad use of is_star")

    if plnames and len(plnames) != npl:
        raise ValueError("Shape of plnames mismatch")
    
    # if sep_files, there shouldnt be an ibody column
    if sep_files and ("ibody" in names):
        raise Exception("separated files don't have 'ibody' column")
        
    # if not sep_files, there should be an ibody column
    if not sep_files and ("ibody" not in names):
        raise Exception("'ibody' column missing in 'names'")

    # =============== READ DATA =============== #
    # select parameters with usecols
    N_names = len(names)
    usecols = [i for i in range(N_names) if names[i] != "_"]
    names = [names[i] for i in usecols]

    # read data
    if not sep_files:
        data = pd.read_table(
            file,
            delimiter=r"\s+",
            names=names,
            header=None,
            usecols=usecols,
            )
    elif sep_files:
        data = pd.DataFrame()
        for ipl in np.arange(1,npl+1):
            astk_ind = file.index('*')
            file_ith = file[:astk_ind]+str(ipl)+file[astk_ind+1:]
            data_ith = pd.read_table(
                    file_ith,
                    delimiter=r"\s+",
                    names=names,
                    header=None,
                    usecols=usecols,
                    )
            data_ith['ibody'] = ipl
            data = pd.concat([data,data_ith])
        data = data.sort_values(['times','ibody'])
    nrows = len(data.index)

    # =============== PREPARE DATAFRAME =============== #
    # validate consistent number of planets
    if "ibody" in names:
        _npl_from_table = np.max(data["ibody"].values)
        if npl != _npl_from_table:
            raise Exception("number of planets mismatch")

    # add ibody column if not there
    if "ibody" not in names:
        pl_ibodies = np.arange(npl) + 1
        pl_ibodies = np.tile(pl_ibodies, nrows // npl)
        if len(pl_ibodies) != nrows:
            raise Exception(
                f"{file} missing lines. If collisions, consider \
                            using an 'ibody' column or one file per planet"
            )
        data["ibody"] = pl_ibodies
        
    
    # =============== ORGANIZE DATA =============== #
    # set default values of mass and radius to list of nones
    if mass is None:
        mass = [None] * npl
    if radius is None:
        radius = [None] * npl
    if plnames is None:
        plnames = [""] * npl
    if is_star is None:
        is_star = [False] * npl

    # mass list of masses
    if "mass" in names:
        mass = data["mass"].values[:npl]


    # =============== CREATE DYNAMIC PLANETS =============== #
    planets = []
    for ipl in range(npl):
        ith_pl = data[data["ibody"] == ipl + 1]
        
        # get everything
        times_i   = _try_getting(ith_pl, "times")
        a_i       = _try_getting(ith_pl, "a")
        e_i       = _try_getting(ith_pl, "e")
        inc_i     = _try_getting(ith_pl, "inc")
        M_i       = _try_getting(ith_pl, "M")
        w_i       = _try_getting(ith_pl, "w")
        Omega_i   = _try_getting(ith_pl, "Omega")
        mass_i    = mass[ipl]
        radius_i  = radius[ipl]
        name_i    = plnames[ipl]
        is_star_i = is_star[ipl]
        
        # convert angles
        inc_i   = _to_Angles(inc_i)
        M_i     = _to_Angles(M_i)
        w_i     = _to_Angles(w_i)
        Omega_i = _to_Angles(Omega_i)
        
        # create the object
        ith_pl_obj = DynamicPlanet(
            times   = times_i,
            a       = a_i,
            e       = e_i,
            inc     = inc_i,
            M       = M_i,
            w       = w_i,
            Omega   = Omega_i,
            mass    = mass_i,
            radius  = radius_i,
            name    = name_i,
            is_star = is_star_i,
        )
        planets.append(ith_pl_obj)

    # =============== CREATE STAR =============== #
    star = Star(mass=st_m, radius=st_r)
    
    # ======== DYNAMIC SYSTEM PROPERTIES ======== #
    resangs = _try_getting(data, "resangs")
    resangs = _to_Angles(resangs)
    resangs_l = _separate_resangs(resangs,npl)
    
    return DynamicSystem(star=star, planets=planets, resangs=resangs_l)
