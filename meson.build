project('lassoinf', 'cpp',
  version : '0.1.0',
  default_options : ['warning_level=3', 'cpp_std=c++14'])

py = import('python').find_installation(pure: false)

pybind11_dep = dependency('pybind11')

py.install_sources(
  'python/lassoinf/__init__.py',
  'python/lassoinf/discrete_family.py',
  'python/lassoinf/gaussian_family.py',
  'python/lassoinf/selective_inference.py',
  subdir: 'lassoinf'
)

py.install_sources(
  'python/lassoinf/operators/__init__.py',
  'python/lassoinf/operators/composite.py',
  'python/lassoinf/operators/xtvx.py',
  subdir: 'lassoinf/operators'
)

py.extension_module(
  'lassoinf_cpp',
  'cpp/src/selective_inference.cpp',
  'cpp/src/lasso_post_selection_constraints.cpp',
  'cpp/src/discrete_family.cpp',
  'cpp/src/gaussian_family.cpp',
  'python/discrete_family_py.cpp',
  'python/gaussian_family_py.cpp',
  'python/lassoinf_py.cpp',
  include_directories: include_directories('cpp/include', 'subprojects/eigen'),
  dependencies: [pybind11_dep],
  install: true
)