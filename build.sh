#!/bin/bash

set -xe

CC="gcc"
LIBS=("m" "pthread")
BUILD_NAME="cerberus"
INCLUDE_DIRS=("malpractice" "malpractice/lodge")

CFLAGS=""
for dir in "${INCLUDE_DIRS[@]}"; do
	CFLAGS="${CFLAGS} -I${dir}"
done

LFLAGS=""
for lib in "${LIBS[@]}"; do
	LFLAGS="${LFLAGS} -l${lib}"
done

mkdir -p ./build
$CC "${BUILD_NAME}.c" $CFLAGS -o "./build/${BUILD_NAME}" $LFLAGS
./build/$BUILD_NAME
