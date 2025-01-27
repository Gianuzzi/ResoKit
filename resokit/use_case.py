from rsk_io import load_integration
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# sys1 = load_integration(
#     "datasets/joint_indiv_files.dat",
#     npl=3,
#     names=["times", "ibody", "a", "e", "_", "M", "w", "Omega", "resangs"],
# )

# sys1 = load_integration(
#     "datasets/planet*.dat",
#     npl=3,
#     sep_files=True,
#     names=["times", "a", "e", "_", "M", "w", "Omega", "resangs"],
# )

sys1 = load_integration(
    "datasets/5planet_migration.dat",
    npl=5,
    names=["times","ibody", "a", "e", "_", "M", "w", "Omega"],
    st_m=1,
    mass=[1,1,1,1,1]
)

pl1 = sys1.planets[0]
pl2 = sys1.planets[1]
pl3 = sys1.planets[2]

n1n2 = sys1.Prat(0)
n2n3 = sys1.Prat(1)

# l1x,l2x = np.min(n1n2)*0.85,np.max(n1n2)*1.15
# l1y,l2y = np.min(n2n3)*0.85,np.max(n2n3)*1.15
# plt.figure()
# plt.scatter(n1n2,n2n3)
# plt.xlim(l1x,l2x)
# plt.ylim(l1y,l2y)
# plt.show()

sys1.sepspace(s=1)




# lam1 = pl1.lam
# times = sys1.times

# plt.figure()
# sys1.plot('resangs',which_resang=2)
# sys1.scatter('resangs',which_resang=0,zorder=100,c='k')
# plt.show()