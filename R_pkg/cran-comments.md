## Submission

This is a new submission.

## Test environments

* local macOS (aarch64), R 4.5.3
* GitHub Actions: macOS (release), Windows (release), Ubuntu (devel, release, oldrel-1):
  Status OK
* win-builder: Windows, R-devel (2026-10-05 r90641 ucrt): 1 note (below)

## R CMD check results

0 errors | 0 warnings | 1 note

* checking CRAN incoming feasibility ... NOTE
  Maintainer: 'Jonathan Taylor <jtaylo@stanford.edu>'
  New submission

  Possibly misspelled words in DESCRIPTION: Liu, Panigrahi, Tian, estimands

  These are correctly spelled: the surnames of the authors of the cited papers, and
  "estimands", the standard statistical term for the quantities being estimated.

## Notes for the reviewers

* The C++ sources in `src/lassoinf/` are shared with the package's Python implementation
  (https://github.com/jonathan-taylor/lassoinf).
* The truncated bivariate normal computations are adapted from MIT-licensed code by
  Sifan Liu (listed as contributor and copyright holder); see `inst/COPYRIGHTS`.
