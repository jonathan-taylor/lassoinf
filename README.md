# lassoinf

`lassoinf` provides selective inference tools leveraging a high-performance C++ backend. It is a hybrid codebase written in Python and C++, utilizing `pybind11` for bindings and the `Eigen` library for fast linear algebra.

## Features

- **Selective Inference**: Core numerical tools for inference tasks.
- **Dual Implementation**: Logic provided in pure Python (`lassoinf.selective_inference`) and fast C++ counterparts (`lassoinf_cpp`).
- **High Performance**: The C++ extensions target C++14 and integrate seamlessly with numpy and scipy.

## Requirements

### Python
- Python >= 3.9
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

The R package bindings are located in the `R_pkg/` directory. You can install the R package from the terminal using `R CMD INSTALL`:

```bash
cd R_pkg
R CMD INSTALL .
```

Or from within an R session using `devtools`:

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

For more information, refer to the documentation in the `docs/` folder.

## License

This project is licensed under the BSD 3-Clause License. See the `LICENSE` file for details.
