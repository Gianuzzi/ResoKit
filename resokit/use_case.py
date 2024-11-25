from rsk_io import load_integration
import matplotlib.pyplot as plt

# sys1 = load_integration(
#     "datasets/joint_indiv_files.dat",
#     npl=3,
#     names=["times", "ibody", "a", "e", "_", "M", "w", "Omega", "resangs"],
# )

sys1 = load_integration(
    "datasets/planet*.dat",
    npl=3,
    sep_files=True,
    names=["times", "a", "e", "_", "M", "w", "Omega", "resangs"],
)

pl1 = sys1.planets[0]
pl2 = sys1.planets[1]

lam1 = pl1.lam
times = sys1.times

plt.figure()
sys1.plot('resangs',which_3p_ang=2)
sys1.scatter('resangs',which_3p_ang=0,zorder=100,c='k')
plt.show()