# lassoinf

`lassoinf` provides selective inference tools leveraging a high-performance C++ backend. It is a hybrid codebase written in Python and C++, utilizing `pybind11` for bindings and the `Eigen` library for fast linear algebra.

## Features

- **Affine Constraints**: Core numerical tools for inference tasks.
- **Dual Implementation**: Logic provided in pure Python (`lassoinf.affine_constraints`) and fast C++ counterparts (`lassoinf_cpp`).
- **High Performance**: The C++ extensions target C++14 and integrate seamlessly with numpy and scipy.

## Requirements

### Python
- Python >= 3.10
- `numpy`, `scipy`, `matplotlib`
- `meson-python`, `ninja`, `pybind11` (for building the extension)

### R
- R >= 4.0
- `Rcpp`, `RcppEigen`

### General
- A C++14 compliant compiler

## Installation

### Python

You can install `lassoinf` and build its C++ extension directly using `pip`:

```bash
pip install .
```

For an editable install during development (using the `meson-python` backend):

```bash
pip install --no-build-isolation -Csetup-args=-Dbuildtype=debug -e .
```

### R

The R package `lassoinf` requires `Rcpp`, `RcppEigen`, and `R6`. You can install the package from the terminal or directly from the source directory.

**From the terminal:**

```bash
# Build and install the package
R CMD INSTALL R_pkg
```

**From within an R session:**

```R
# Install required dependencies if not already present
install.packages(c("Rcpp", "RcppEigen", "R6", "mvtnorm"))

# Install the package from the source directory
install.packages("R_pkg", repos = NULL, type = "source")
```

Or using `devtools`:

```R
devtools::install("R_pkg")
```

### Testing the R Package

To run the R test suite, you can use the `testthat` package from the command line:

```bash
cd R_pkg
Rscript -e 'testthat::test_dir("tests/testthat")'
```

## Documentation

For more information, refer to the documentation in the `docs/` folder. For the R package, you can view the vignette:

```R
vignette("gaussian_lasso_boot", package = "lassoinf")
```

Alternatively, you can browse the `pkgdown` documentation site in `R_pkg/docs/index.html`.

## License

This project is licensed under the BSD 3-Clause License. See the `LICENSE` file for details.
