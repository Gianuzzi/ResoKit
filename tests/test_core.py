# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# # This file is part of the
# #   ResoKit Project (https://github.com/Gianuzzi/resokit).
# # Copyright (c) 2025, Emmanuel Gianuzzi
# # License: MIT
# #   Full Text: https://github.com/Gianuzzi/resokit/blob/master/LICENSE

# # ============================================================================
# # IMPORTS
# # ============================================================================

# import pandas as pd

# import pytest

# import resokit.io as rio
# from resokit.core import (
#     StaticBinaryStar,
#     StaticPlanet,
#     StaticStar,
#     StaticSystem,
# )

# # ============================================================================
# # CONSTANTS
# # ============================================================================


# bin_syst = "kepler47"  # Binary system with 3 planets
# simple_syst = "kepler11"  # Simple system with 6 planets
# load_function = {
#     "eu": rio.load_system_from_eu,
#     "nasa": rio.load_system_from_nasa,
# }


# class K11:
#     npl = 6
#     pl_names = [
#         "Kepler-11 b",
#         "Kepler-11 c",
#         "Kepler-11 d",
#         "Kepler-11 e",
#         "Kepler-11 f",
#         "Kepler-11 g",
#     ]
#     data = pd.DataFrame(
#         data=[
#             [  # Kepler-11 b
#                 6.000000e-03,
#                 1.610000e-01,
#                 1.030390e01,
#                 1.000000e-03,
#                 6.000000e-04,
#             ],
#             [  # Kepler-11 c
#                 9.000000e-03,
#                 2.560000e-01,
#                 1.302410e01,
#                 8.000000e-04,
#                 1.300000e-03,
#             ],
#             [  # Kepler-11 d
#                 2.300000e-02,
#                 2.780000e-01,
#                 2.268450e01,
#                 9.000000e-04,
#                 9.000000e-04,
#             ],
#             [  # Kepler-11 e
#                 2.500000e-02,
#                 3.740000e-01,
#                 3.199960e01,
#                 1.200000e-03,
#                 8.000000e-04,
#             ],
#             [  # Kepler-11 f
#                 6.000000e-03,
#                 2.220000e-01,
#                 4.668880e01,
#                 3.200000e-03,
#                 2.700000e-03,
#             ],
#             [  # Kepler-11 g
#                 7.900000e-02,
#                 2.970000e-01,
#                 1.183807e02,
#                 6.000000e-04,
#                 1.000000e-03,
#             ],
#         ],
#         columns=["mass", "radius", "P", "P_err_min", "P_err_max"],
#         index=pl_names,
#     )
#     nasa_p_err_min = [0.0006, 0.0013, 0.0009, 0.0008, 0.0027, 0.001]
#     nasa_p_err_max = [0.001, 0.0008, 0.0009, 0.0012, 0.0032, 0.0006]

#     perat = pd.DataFrame(
#         data=[
#             [1.0, 1.26399713, 2.20154505, 3.10558138, 4.53117752, 11.48892167],
#             [0.79114104, 1.0, 1.74173263, 2.45695288, 3.58480049, 9.08935742],
#             [0.45422645, 0.57414093, 1.0, 1.41063722, 2.0581807, 5.21857215],
#             [0.3220009, 0.40700821, 0.70889949, 1.0, 1.45904324, 3.69944312],
#             [0.22069319, 0.27895555, 0.48586599, 0.68538065, 1.0, 2.53552672],
#             [0.08704037, 0.11001878, 0.1916233, 0.27031095, 0.39439537, 1.0],
#         ],
#         columns=pl_names,
#         index=pl_names,
#     )

#     estimated_mass = pd.DataFrame(
#         data=[
#             [  # ck17
#                 0.01187256,
#                 0.02609178,
#                 0.03001185,
#                 0.04966084,
#                 0.02048483,
#                 0.0335768,
#             ],
#             [  # o20
#                 0.0201735,
#                 0.09992169,
#                 0.13279644,
#                 0.0509617,
#                 0.06111526,
#                 0.03540478,
#             ],
#             [  # e23
#                 0.01848317,
#                 0.03655726,
#                 0.04126936,
#                 0.06383795,
#                 0.02964581,
#                 0.04548318,
#             ],
#             [  # m24
#                 0.01748089,
#                 0.03492863,
#                 0.03950223,
#                 0.06150359,
#                 0.02823663,
#                 0.04359883,
#             ],
#         ],
#         columns=pl_names,
#         index=["ck17", "o20", "e23", "m24"],
#     ).T

#     estimated_radii = pd.DataFrame(
#         data=[
#             [  # ck17
#                 0.10998515,
#                 0.13676285,
#                 0.23767145,
#                 0.24963529,
#                 0.10998515,
#                 0.49161109,
#             ],
#             [  # o20
#                 0.11318642,
#                 0.1273095,
#                 0.1671212,
#                 0.17121156,
#                 0.11318642,
#                 0.23902445,
#             ],
#             [  # e23
#                 0.11027453,
#                 0.12353237,
#                 0.18680687,
#                 0.19770476,
#                 0.11027453,
#                 0.43231902,
#             ],
#             [  # m24
#                 0.11064982,
#                 0.12345122,
#                 0.19349331,
#                 0.20461059,
#                 0.11064982,
#                 0.44230155,
#             ],
#         ],
#         columns=pl_names,
#         index=["ck17", "o20", "e23", "m24"],
#     ).T


# # ============================================================================
# # TESTS
# # ============================================================================


# class TestLoadSystem:
#     @pytest.mark.parametrize("source", ["eu", "nasa"])
#     def test_load_binary_system(self, source: str):
#         """Test load_system with a binary system."""
#         syst = load_function[source](
#             name=bin_syst, verbose=False, exact_match=False
#         )

#         # Assert types
#         assert isinstance(syst, StaticSystem)
#         assert isinstance(syst.star, StaticBinaryStar)
#         for star in syst.star:
#             assert isinstance(star, StaticStar)
#         for planet in syst.planets:
#             assert isinstance(planet, StaticPlanet)

#         # Assert source
#         if source == "eu":
#             assert syst.source_ == source
#         else:
#             assert syst.source_ == "eu_and_nasa"

#         # Assert binarity
#         assert syst.is_binary_
#         assert syst.is_circumbinary

#         # Assert number of planets
#         assert len(syst.planets) == 3

#         # Assert names
#         assert syst.star.name == "Kepler-47"
#         assert syst.star.star0.name == "Kepler-47 A"
#         assert syst.star.star1.name == "Kepler-47 B"

#     @pytest.mark.parametrize("source", ["eu", "nasa"])
#     def test_load_binary_without_binary_system(self, source: str):
#         """Test load_system with a binary system."""
#         syst = load_function[source](
#             name=bin_syst, verbose=False, exact_match=False, check_binary=False
#         )

#         # Assert types
#         assert isinstance(syst, StaticSystem)
#         assert isinstance(syst.star, StaticStar)
#         for planet in syst.planets:
#             assert isinstance(planet, StaticPlanet)

#         # Assert source
#         assert syst.source_ == source

#         # Assert not binarity
#         assert not syst.is_binary_

#         # Assert number of planets
#         assert len(syst.planets) == 3

#         # Assert names
#         if source == "eu":
#             assert syst.name == "Kepler-47 A"
#             assert syst.star.name == "Kepler-47 A"
#         else:
#             assert syst.name == "Kepler-47"
#             assert syst.star.name == "Kepler-47"

#     @pytest.mark.parametrize("source", ["eu", "nasa"])
#     def test_load_simple_system(self, source: str):
#         """Test load_system with a simple system."""
#         syst = load_function[source](
#             name=simple_syst, verbose=False, exact_match=False
#         )

#         # Assert types
#         assert isinstance(syst, StaticSystem)
#         assert isinstance(syst.star, StaticStar)
#         for planet in syst.planets:
#             assert isinstance(planet, StaticPlanet)

#         # Assert source
#         assert syst.source_ == source

#         # Assert binarity
#         assert not syst.is_binary_

#         # Assert number of planets
#         assert len(syst.planets) == K11.npl

#         # Assert names
#         assert syst.star.name == "Kepler-11"


# class TestStaticSystem:
#     @pytest.mark.parametrize("name", ["Kepler-11 b", "b", 0])
#     def test_static_system_get_planet(self, name):
#         """Test StaticSystem class planet method."""
#         syst = rio.load_system_from_eu(
#             name=simple_syst, verbose=False, exact_match=False
#         )

#         # Test get planet
#         pl1 = syst.planet("Kepler-11 b")

#         # Assert types
#         assert isinstance(pl1, StaticPlanet)

#         # Assert is the same as the first planet
#         assert pl1 == syst.planets[0]

#     def test_static_system_get_item(self):
#         """Test StaticSystem class get_item method."""
#         syst = rio.load_system_from_eu(
#             name=simple_syst, verbose=False, exact_match=False
#         )

#         # Test get item
#         per = syst.get_item("P")

#         # Assert types
#         assert isinstance(per, pd.Series)

#         # Assert equal to base example
#         pd.testing.assert_series_equal(
#             per,
#             K11.data["P"],
#         )

#     @pytest.mark.parametrize("source", ["eu", "nasa"])
#     def test_static_system_get_item_errors(self, source: str):
#         """Test StaticSystem class get_item with errors method."""
#         syst = load_function[source](
#             name=simple_syst, verbose=False, exact_match=False
#         )

#         # Test get item
#         per = syst.get_item("P", error=True)

#         # Assert types
#         assert isinstance(per, pd.DataFrame)

#         base = K11.data[["P", "P_err_min", "P_err_max"]]

#         # Assert equal to base example
#         if source == "eu":
#             pd.testing.assert_frame_equal(
#                 per,
#                 base,
#             )
#         else:
#             base["P_err_min"] = K11.nasa_p_err_min
#             base["P_err_max"] = K11.nasa_p_err_max
#             pd.testing.assert_frame_equal(
#                 per,
#                 base,
#             )

#     @pytest.mark.parametrize("source", ["eu", "nasa"])
#     def test_static_system_get_wrong_item(self, source: str):
#         """Test StaticSystem class get_item with errors method."""
#         syst = load_function[source](
#             name=simple_syst, verbose=False, exact_match=False
#         )

#         with pytest.raises(KeyError):
#             syst.get_item("wrong_item", error=True)

#     def test_static_system_period_ratios(self):
#         """Test StaticSystem class period_ratios method."""
#         syst = rio.load_system_from_eu(
#             name=simple_syst, verbose=False, exact_match=False
#         )

#         # Test get item
#         perat = syst.period_ratios_

#         # Assert types
#         assert isinstance(perat, pd.DataFrame)

#         # Assert equal to base example
#         pd.testing.assert_frame_equal(
#             perat,
#             K11.perat,
#         )

#     def test_static_system_pair_ratio(self):
#         """Test StaticSystem class period_ratios method."""
#         syst = rio.load_system_from_eu(
#             name=simple_syst, verbose=False, exact_match=False
#         )

#         # Test get item
#         perat = syst.pair_ratio()

#         # Assert types
#         assert isinstance(perat, pd.DataFrame)

#         # Assert equal to base example
#         pd.testing.assert_frame_equal(
#             perat,
#             K11.perat,
#         )

#     @pytest.mark.parametrize("model", ["ck17", "o20", "e23", "m24", "bad"])
#     @pytest.mark.parametrize("force", [True, False])
#     def test_estimate_mass(self, model, force):
#         """Test StaticSystem class estimate_mass method."""
#         syst = rio.load_system_from_eu(
#             name=simple_syst, verbose=False, exact_match=False
#         )
#         multivariate = (
#             0.9999
#             if model in ["o20", "m24"]
#             else ((0.0001, 0.9999) if model == "ch17" else None)
#         )  # Ensure that the test is always the same
#         if not force:
#             # Assert equal to base example
#             mass_ss = syst.estimate_mass(
#                 model=model, force=force, multivariate=multivariate
#             )
#             assert mass_ss.equals(K11.data["mass"])

#         elif model == "bad":
#             with pytest.raises(ValueError):
#                 syst.estimate_mass(
#                     model=model, force=force, multivariate=multivariate
#                 )
#         else:
#             if model == "o20":
#                 with pytest.warns(UserWarning):
#                     mass_ss = syst.estimate_mass(
#                         model=model, force=force, multivariate=multivariate
#                     )
#             else:
#                 mass_ss = syst.estimate_mass(
#                     model=model, force=force, multivariate=multivariate
#                 )

#             # Assert types
#             assert isinstance(mass_ss, pd.Series)

#             # Assert equal to base example
#             pd.testing.assert_series_equal(
#                 mass_ss,
#                 K11.estimated_mass[model],
#                 check_names=False,  # Names are different
#             )

#     @pytest.mark.parametrize("model", ["ck17", "o20", "e23", "m24", "bad"])
#     @pytest.mark.parametrize("force", [True, False])
#     def test_estimate_radius(self, model, force):
#         """Test StaticSystem class estimate_radius method."""
#         syst = rio.load_system_from_eu(
#             name=simple_syst, verbose=False, exact_match=False
#         )
#         bivariate = 0.99 if model == "o20" else None
#         if not force:
#             # Assert equal to base example
#             radii_ss = syst.estimate_radius(
#                 model=model, force=force, bivariate=bivariate
#             )
#             assert radii_ss.equals(K11.data["radius"])

#         elif model == "bad":
#             with pytest.raises(ValueError):
#                 syst.estimate_radius(
#                     model=model, force=force, bivariate=bivariate
#                 )
#         else:
#             if model == "o20":
#                 with pytest.warns(UserWarning):
#                     radii_ss = syst.estimate_radius(
#                         model=model, force=force, bivariate=bivariate
#                     )
#             else:
#                 radii_ss = syst.estimate_radius(
#                     model=model, force=force, bivariate=bivariate
#                 )

#             # Assert types
#             assert isinstance(radii_ss, pd.Series)

#             # Assert equal to base example
#             pd.testing.assert_series_equal(
#                 radii_ss,
#                 K11.estimated_radii[model],
#                 check_names=False,  # Names are different
#             )


# # ============================================================================
