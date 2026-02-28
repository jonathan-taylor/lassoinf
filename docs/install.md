# Installation

`lassoinf` provides selective inference tools leveraging a high-performance C++ backend. It integrates seamlessly with both Python and R.

## Requirements

### Python
- Python >= 3.9
- `numpy`, `scipy`, `matplotlib`
- `meson-python`, `ninja`, `pybind11` (for building the C++ extension)

### R
- R >= 4.0
- `Rcpp`, `RcppEigen`

### General
- A C++14 compliant compiler

## Installation for Python

You can install `lassoinf` and build its C++ extension directly using `pip` from the repository root:

```bash
pip install .
```

For an editable install during development (using the `meson-python` backend), run:

```bash
pip install --no-build-isolation -Csetup-args=-Dbuildtype=debug -e .
```

## Installation for R

The R package bindings are located in the `R_pkg/` directory. You can install the R package from the terminal using `R CMD INSTALL`:

```bash
cd R_pkg
R CMD INSTALL .
```

Alternatively, from within an R session using `devtools`:

```R
devtools::install("R_pkg")
```

### Testing the R Package

To run the R test suite, you can use the `testthat` package from the command line:

```bash
cd R_pkg
Rscript -e 'testthat::test_dir("tests/testthat")'
```
