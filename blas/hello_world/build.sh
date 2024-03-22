#!/bin/bash

BASEDIR=$(cd "$(dirname -- "$0")" >/dev/null; pwd -P)
BLAS_ROOT_PATH="$BASEDIR/../.."
HALIDE_ROOT_PATH="$BLAS_ROOT_PATH/tools/Halide"

build_single_kernel() {
    cd "$BASEDIR/../$1" && mkdir -p build && cd build
    cmake .. && make "oneapi_s${2}_tiny_a10"
    mkdir -p "$BASEDIR/build/include/$1"
    cp "$BASEDIR/../reconfigurable_${2}/bin/exposed_funcs.hpp" "$BASEDIR/build/include/$1/"
}

build_single_kernel dot dotprod
build_single_kernel axpy vecadd

cd "$BASEDIR/build"
icpx -fsycl -fintelfpga \
    -I include \
    -I "$BLAS_ROOT_PATH/include" \
    -I "$HALIDE_ROOT_PATH/include" \
    -L "$HALIDE_ROOT_PATH/lib" \
    -lm -ldl -lsycl -lHalide \
    ../kernel.cpp -o kernel
cp kernel ../
