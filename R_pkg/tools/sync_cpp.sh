#!/bin/sh
# Copy the shared C++ library (../cpp) into src/lassoinf so the R package is
# self-contained, as R CMD build / CRAN require. Run from R_pkg/ after editing
# anything under ../cpp.
set -e
cd "$(dirname "$0")/.."
rm -rf src/lassoinf
mkdir -p src/lassoinf/include src/lassoinf/src
cp ../cpp/include/*.h ../cpp/include/*.hpp src/lassoinf/include/
cp ../cpp/src/*.cpp src/lassoinf/src/
