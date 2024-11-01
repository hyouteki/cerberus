#!/bin/bash

set -xe

CC="gcc"
LIBS="-lm"
BUILD_NAME="cerberus"
INCLUDE_DIRS="malpractice"

mkdir -p ./build
$CC "${BUILD_NAME}.c" -o "./build/${BUILD_NAME}" -I $INCLUDE_DIRS $LIBS
./build/$BUILD_NAME
