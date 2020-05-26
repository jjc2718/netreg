# netscape

Exploring the **net**work/pathway regularization land**scape** for gene expression data

Overall goal of this project: evaluate network regularized methods for
a variety of practical problems using a variety of biological networks.

At present, this repo contains an adaptation of the pipeline used in
[BioBombe](https://github.com/greenelab/BioBombe) for predicting gene
alteration status using TCGA gene expression data.

## Setup

We recommend using the conda environment specified in the `environment.yml` file to run these analyses. To build and activate this environment, run:

```shell
# conda version 4.5.0
conda env create --file environment.yml

conda activate netscape
```

To run PLIER, you'll have to install it from GitHub using the
`devtools` library. Run the following from an R interactive shell/
REPL inside the `netscape` Conda environment:

```R
library(devtools)
install_github('wgmao/PLIER')

```

You should then be able to run the `library(PLIER)` command
without errors.

## Running tests

Running the tests requires the `pytest` module (included in the specified
Conda environment). Once this module is installed, you can run the tests
by executing the command

```shell
pytest tests/
```

from the repo root.
