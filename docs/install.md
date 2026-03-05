---
jupytext:
  main_language: bash
  cell_metadata_filter: -all
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
---

# Installation

`lassoinf` provides selective inference tools leveraging a high-performance C++ backend. It integrates seamlessly with both Python and R.

## Requirements

### Python
- Python >= 3.10
- `numpy`, `scipy`, `matplotlib`
- `meson-python`, `ninja`, `pybind11` (for building the C++ extension)

### R
- R >= 4.0
- `Rcpp`, `RcppEigen`

### General
- A C++14 compliant compiler

## Installation for Python

```
pip install lassoinf
```

To build from source, you can install `lassoinf` and build its C++ extension directly using `pip`:

```
pip install .
```

For an editable install during development (using the `meson-python` backend), run:

```
pip install --no-build-isolation -Csetup-args=-Dbuildtype=debug -e .
```

## Installation for R

The R package bindings are located in the `R_pkg/` directory. You can install the R package from the terminal using `R CMD INSTALL`:

```
cd R_pkg
R CMD INSTALL .
```

Or from within an R session using `devtools`:

```
%%R
devtools::install("R_pkg")
```

### Testing the R Package

To run the R test suite, you can use the `testthat` package from the command line:

```
cd R_pkg
Rscript -e 'testthat::test_dir("tests/testthat")'
```
