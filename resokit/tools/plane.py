import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from . import rmm

# REQUIERE EL ARCHIVO DE DATOS DE LA BASE DEL EXOPLANET.EU
#  QUE DEVUELVE EL read_eu.f -> "geneva.txt"

cols = [
    "isys",
    "npl",
    "ipl",
    "plname",
    "m",
    "msini",
    "R",
    "P",
    "a",
    "e",
    "inc",
    "w",
    "lambda",
    "detection",
    "st_met",
    "st_m",
    "st_R",
    "age",
    "discovery",
]
widths = [
    6,
    6,
    11,
    28,
    13,
    10,
    10,
    17,
    13,
    12,
    10,
    10,
    10,
    4,
    8,
    8,
    8,
    8,
    8,
]  # ancho fijo de cada columna, como se los dio en el read_eu
dtype = [
    "int",
    "int",
    "int",
    "str",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "int",
    "float",
    "float",
    "float",
    "float",
    "int",
    "int",
]
dtype = dict(zip(cols, dtype))
data = pd.read_fwf("geneva.txt", widths=widths, header=None, dtype=dtype, names=cols)


# SISTEMA EN CUESTION -- ELEGIR SOLO 1 METODO 1) O 2) Y COMENTAR EL OTRO!!
# 1) buscar por indice de sistema
# sysi = 2337

# 2) buscar por nombre del sistema
name = "V1298 Tau"
plnames = data["plname"].values
if "sysi" in locals():
    raise Exception("Ambos métodos de selección de sistemas activados. COMENTAREAR UNO")
for i in range(len(plnames)):
    if plnames[i][: len(name)] != name:
        continue
    sysi = data.loc[i]["isys"]
    break


# datos del sistema
sys_data = data[data["isys"] == sysi]
npl = sys_data["npl"].values[0]
Pl = sys_data["P"].values
nnl = []
for i in range(npl - 1):
    nnl.append(Pl[i + 1] / Pl[i])
if npl < 3:
    raise Exception(r"El sistema sólo tiene {} planeta/s".format(npl))
sys_name = data[data["isys"] == sysi]["plname"].values[0][:-1]


# PLOT
l1x, l2x = 1.1, 2.5
# l1x,l2x = 1.49,1.57
# l1y,l2y = 1.39,1.48
l1y, l2y = l1x, l2x
lims = [l1x, l2x, l1y, l2y]

r3p, r2x, r2y = rmm.rmm_in_area(lims)

plt.figure(figsize=(5, 5), dpi=120)

l2k = dict(lw=0.75, alpha=0.75, linestyle="dashed", c="grey", zorder=2)
l3k = dict(lw=0.75, alpha=0.75, c="k", zorder=3)
l3k_i = dict(lw=1.3, alpha=0.75, c="red", zorder=3)
pk = dict(s=30, lw=0.5, edgecolor="k", zorder=130)

# ---------
dom = np.linspace(l1x, l2x, 2000)
for r3pi in r3p:
    k1, k2, k3 = r3pi
    plt.plot(dom, rmm.r3p(dom, [k1, k2, k3]), **l3k)
    rmm.r3p_label(r3pi, plt.gca(), lims=lims)
for r2xi in r2x:
    plt.gca().axvline(r2xi[0] / r2xi[1], **l2k)
for r2yi in r2y:
    plt.gca().axhline(r2yi[0] / r2yi[1], **l2k)

# ---------
for i in range(npl - 2):
    plt.scatter(nnl[i], nnl[i + 1], label=str(i + 1) + str(i + 2) + str(i + 3), **pk)


plt.xlabel("$n_i / n_{i+1}$      -     " + sys_name)
plt.ylabel("$n_{i+1} / n_{i+2}$")

plt.xlim(l1x, l2x)
plt.ylim(l1y, l2y)

plt.legend(edgecolor="k", facecolor="w")

plt.show()
