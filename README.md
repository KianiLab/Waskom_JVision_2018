# Code and data from Waskom, Asfour, & Kiani (2018)

This repository contains data and analysis code for the following paper:

Waskom ML, Asfour JW, Kiani R (2018). [Perceptual insensitivity to higher-order statistical moments of coherent random dot motion](http://jov.arvojournals.org/article.aspx?articleid=2685931). *Journal of Vision* 18(6):9 1-10.

## Data

The [`trial_data.csv`](./trial_data.csv) file has a tidy table with data corresponding to each trial. Fields are as follows:

| Column        | Description                                   |
|---------------|-----------------------------------------------|
| subject       | subject id                                    |
| condition     | odd motion condition name                     |
| correct       | was odd motion correctly identified?          |
| rt            | reaction time (in seconds)                    |
| odd_{x,y}     | coordinates of odd aperture center, in deg    |
| sacc_{x,y}    | coordinates of saccade landing point, in deg  |
| odd_{m,v,s,k} | trialwise statistics of odd dot displacements |

## Behavioral analyses

The [`behavioral_analyses.ipynb`](./behavioral_analyses.ipynb) notebook contains figures and statistical models for analyzing the behavioral data.

## Motion energy model

The [`motionenergy.py`](./motionenergy.py) module contains a Python implementation of the Adelson Bergen spatiotemporal energy model. It depends on numpy and scipy. The [`motionenergy_tutorial.ipynb`](./motionenergy_tutorial.ipynb) notebook contains a tutorial demonstration that shows how to use the model implementation and gives some intuition for its parameters. The tutorial depends on the [`stimulus.py`](./stimulus.py) module, which makes random dot and drifting grating movies. While you can view the static tutorial notebook, it contains interactive elements that will be much more informative if you download and run locally. The tutorial is written for Python 3.

## Dependencies

A list of software dependencies and versions corresponding to the paper can be found in [`requirements.txt`](./requirements.txt).

## License

These files are being released openly in the hope that they might be useful but with no promise of support. If using them leads to a publication, please cite the paper.

The dataset is released under a CC-BY 4.0 license.

The code is released under a [BSD license](./LICENSE.md).
