# [WIP] Athena

[![Build Status](https://travis-ci.org/athenaml/athena.svg?branch=master)](https://travis-ci.org/athenaml/athena)

**Warning** The project is still heavy work in progress. Not suitable for production use.

## What is Athena?

Athena is a high performance scalable deep learning framework. It represents computation
through computation graph and operates tensors.

## Installation
### Building from source

Prerequisites:
- Git
- CMake 3.13+
- A C++17 capable compiler
- LLVM 8
- Python 3 (optional)

Building from source

1. Checkout repository
2. `path/to/source/scripts/build.py --build-type Release --disable-tests path/to/build/dest path/to/source`

To see other build options use `scripts/build.py --help`

Alternatively, you can use pure CMake flow to build the library.
