from rsk_io import load_integration
import matplotlib.pyplot as plt
import forgacs as fg

sys1 = load_integration(
    "datasets/planet*.dat",
    npl=3,
    names=["times", "a", "e", "inc", "M", "w", "Omega","resangs"],
    sep_files=True
)

# sys1 = load_integration(
#     "datasets/2planet_example.dat",
#     npl=2,
#     names=["times", "ibody", "a", "e", "inc", "M", "w", "Omega", "_","resangs"],
# )

# sys1 = load_integration(
#     "datasets/nonrescase.dat",
#     npl=2,
#     names=["times", "ibody", "a", "e", "inc", "M", "w", "Omega","resangs"],
# )

# sys1 = load_integration(
#     "datasets/21mmr.dat",
#     npl=2,
#     names=["times", "ibody", "a", "e", "inc", "M", "w", "Omega","resangs"],
# )

pl1 = sys1.planets[0]
pl2 = sys1.planets[1]

lam1 = pl1.lam
times = sys1.times

# plt.figure()
# sys1.plot('resangs',which_resang=2)
# sys1.scatter('resangs',which_resang=0,zorder=100,c='k')
# plt.show()

M1 = pl1.M
M2 = pl2.M
lam1 = pl1.lam
lam2 = pl2.lam
# print(fg.detect_2pmmr(M1,M2,lam1,lam2))
# fg.plot_forgacs(M1, M2, lam1, lam2,alpha=.1,s=1)