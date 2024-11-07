from rsk_io import load_integration

sys1 = load_integration(
    "datasets/2planet_example.dat",
    npl=2,
    names=["times", "ibody", "a", "e", "_", "_", "w", "Omega", "_", "_"],
)

pl1 = sys1.planets[0]
pl2 = sys1.planets[1]
