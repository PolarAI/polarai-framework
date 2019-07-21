---
id: installation
title: Installation
---

# Building Athena from source

Prerequisites:
- Git
- CMake 3.13+
- A C++17 capable compiler
- LLVM 8
- Python 3

## Using build script

1. Checkout repository
2. `path/to/source/scripts/build.py --build-type Release --disable-tests path/to/build/dest path/to/source`

To see other build options use `scripts/build.py --help`

## Using CMake

1. Checkout repository
2. Create a build directory and `cd` there
3. Configure project with `cmake -G "Ninja" path/to/source`
4. Build project `cmake --build . -j4`
5. Run tests `ctest`
6. Install `cmake --build . --target install`

Athena defines the following optional config options

|Option|Default|Description|
|------|-------|-----------|
|ATHENA_DISABLE_TESTS|OFF|Disable tests build|
|ATHENA_DOCS_ONLY|OFF|Build docs without building library|
|ATHENA_BUILD_STATIC|OFF|Build static library|
|ATHENA_USE_SANITIZERS|OFF|Build library with sanitizers support. Valid values: seq, par|
|FORCE_BLIS|OFF|Force CPU runtime to use BLIS as BLAS implementation|
|FORCE_UBLAS|OFF|Force CPU runtime to use uBLAS as BLAS implementation|
|ENABLE_COVERAGE|OFF|Generate code coverage info|

