import numpy as np
import pandas as pd
import attrs
from core import DynamicPlanet, Star, DynamicSystem

# allowed inputs
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
}


def _my_get_key(df, key):
    if key in df.columns:
        return df[key].values
    else:
        return None


#####--------------------- LOAD INTEGRATION
def load_integration(
    file,
    npl,
    names=["time", "ibody", "a", "e", "inc", "M", "w", "Omega"],
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
        Path to file. Has to be separated by spaces.
    npl : int
        Number of planets.
    names : list of str, optional
        Contents of the columns. Can only include the terms ["time", "ibody",
        "a", "e", "inc", "M", "w", "Omega", "mass", "_"]. Use "_" for
        throwaways. The default is ["time", "ibody", "a", "e", "inc", "M",
        "w", "Omega"].
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

    # =============== READ DATA =============== #
    # select parameters with usecols
    N_names = len(names)
    usecols = [i for i in range(N_names) if names[i] != "_"]
    names = [names[i] for i in usecols]

    # read data
    data = pd.read_table(
        file,
        delimiter=r"\s+",
        names=names,
        header=None,
        usecols=usecols,
    )
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

    # set default values of mass and radius to list of nones
    if mass is None:
        mass = [None] * npl
    if radius is None:
        radius = [None] * npl
    if plnames is None:
        plnames = [""] * npl
    if is_star is None:
        is_star = [False] * npl

    # always use a list of masses
    if "mass" in names:
        mass = data["mass"].values[:npl]

    # construct DynamicPlanets objects
    planets = []
    for ipl in range(npl):
        ith_pl = data[data["ibody"] == ipl + 1]
        ith_pl_obj = DynamicPlanet(
            times=_my_get_key(ith_pl, "times"),
            a=_my_get_key(ith_pl, "a"),
            e=_my_get_key(ith_pl, "e"),
            inc=_my_get_key(ith_pl, "inc"),
            M=_my_get_key(ith_pl, "M"),
            w=_my_get_key(ith_pl, "w"),
            Omega=_my_get_key(ith_pl, "Omega"),
            mass=mass[ipl],
            radius=radius[ipl],
            name=plnames[ipl],
            is_star=is_star[ipl],
        )
        planets.append(ith_pl_obj)

    # =============== CREATE STAR =============== #
    star = Star(mass=st_m, radius=st_r)
    return DynamicSystem(star=star, planets=planets)


sys1 = load_integration(
    "datasets/2planet_example.dat",
    npl=2,
    names=["times", "ibody", "a", "e", "_", "_", "w", "Omega", "_", "_"],
)

pl1 = sys1.planets[0]
pl2 = sys1.planets[1]
