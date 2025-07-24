# Welcome to **ResoKit**

![logo](https://github.com/gianuzzi/resokit/raw/ema/docs/source/_static/resokit_logo.png)

<!-- BODY -->

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://badge.fury.io/py/uttrs)
[![ResoKit CI](https://github.com/Gianuzzi/ResoKit/actions/workflows/CI.yml/badge.svg)](https://github.com/Gianuzzi/ResoKit/actions/workflows/CI.yml)
[![Documentation Status](https://readthedocs.org/projects/resokit/badge/?version=latest)](https://resokit.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://tldrlegal.com/license/mit-license)
[![https://github.com/leliel12/diseno_sci_sfw](https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00)](https://github.com/leliel12/diseno_sci_sfw)

**ResoKit** is a toolkit package for easy exoplanetary databases access, and the analysis and diagnostics of resonances in n-body planetary systems.

## Motivation
Given the vast number of exoplanetary systems surveyed, and the expectation of many more to come, we believe it is essential to have a tool capable of processing and providing valuable insights from such a large volume of observational data. Two prominent sources of this data are the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) and the [Encyclopaedia of exoplanetary systems](https://exoplanet.eu/home/).

## Features
The two main currently ResoKit applications involve the manipulation of data:

- Exoplaneteary Datasets (including downloading, re-loading and updating) obtained from [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) or the [Encyclopaedia of exoplanetary systems](https://exoplanet.eu/home/).

- Exoplanetary Systems created within the context of the datasets.

## Examples
Some `notebook` format tutorials can be found at in the [tutorials](./docs/source/tutorials/) directory.

More information available in the [documentation](https://resokit.readthedocs.io/en/latest/).


## Future

The implementation of tools for analizyng standard n-body simulations output
(and the Mean Movement Resonacnes involved) is still under development, but will be the next feature of ResoKit.

Stay tuned for updates!

-------------------------------------------------------------------------------


## Requirements

You need Python `3.10`, or greater to run ResoKit.

### Standard Installation

You could find **ResoKit** at PyPI. The standard installation via pip:

```bash
$ python -m pip install resokit
```

#### Extra dependencies

If query online database features will be used with the [Encyclopaedia of exoplanetary systems](https://exoplanet.eu/home/) dataset, package [astropy](https://pypi.org/project/astropy/) is needed.

You can install all extra ResoKit requirements via pip:

```bash
$ python -m pip install resokit[all]
```

### Development Install

Clone this repo and then inside the local directory execute

```bash
$ git clone https://github.com/gianuzzi/resokit.git
$ cd resokit
$ python -m pip install -e .
```

## Authors
- Emmanuel Gianuzzi [egianuzzi@unc.edu.ar](egianuzzi@unc.edu.ar) ([IATE-OAC-CONICET][], [FaMAF-UNC][]).
- Matías Cerioni [matias.cerioni@unc.edu.ar](matias.cerioni@unc.edu.ar) ([IATE-OAC-CONICET][]).



  [IATE-OAC-CONICET]: http://iate.oac.uncor.edu/
  [OAC-CONICET]: https://oac.unc.edu.ar/
  [FaMAF-UNC]: https://www.famaf.unc.edu.ar/
