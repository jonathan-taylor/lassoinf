## Submission

This is a new submission.

## Test environments

* local macOS (aarch64), R 4.5.3

## R CMD check results

0 errors | 0 warnings | 1 note

* checking CRAN incoming feasibility ... NOTE
  Maintainer: 'Jonathan Taylor <jtaylo@stanford.edu>'
  New submission

## Notes for the reviewers

* The C++ sources in `src/lassoinf/` are shared with the package's Python implementation
  (https://github.com/jonathan-taylor/lassoinf).
* The truncated bivariate normal computations are adapted from MIT-licensed code by
  Sifan Liu (listed as contributor and copyright holder); see `inst/COPYRIGHTS`.
