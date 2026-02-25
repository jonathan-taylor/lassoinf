# lassoinf

`lassoinf` provides selective inference tools leveraging a high-performance C++ backend. It is a hybrid codebase written in Python and C++, utilizing `pybind11` for bindings and the `Eigen` library for fast linear algebra.

## Features

- **Selective Inference**: Core numerical tools for inference tasks.
- **Dual Implementation**: Logic provided in pure Python (`lassoinf.selective_inference`) and fast C++ counterparts (`lassoinf_cpp`).
- **High Performance**: The C++ extensions target C++14 and integrate seamlessly with numpy and scipy.

## Requirements

- Python >= 3.9
- `numpy`, `scipy`, `matplotlib`
- `meson-python`, `ninja`, `pybind11` (for building the extension)
- A C++14 compliant compiler

## Installation

You can install `lassoinf` and build its C++ extension directly using `pip`:

```bash
pip install .
```

For an editable install during development (using the `meson-python` backend):

```bash
pip install --no-build-isolation -Csetup-args=-Dbuildtype=debug -e .
```

## Documentation

For more information, refer to the documentation in the `docs/` folder.

## License

This project is licensed under the BSD 3-Clause License. See the `LICENSE` file for details.
