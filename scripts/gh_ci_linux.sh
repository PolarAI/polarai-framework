#!/usr/bin/env bash

export COMMAND_WRAPPER="docker run -v $GITHUB_WORKSPACE:$GITHUB_WORKSPACE -e ATHENA_TEST_ENVIRONMENT=ci -e ATHENA_BINARY_DIR=$GITHUB_WORKSPACE/install_$BUILD_TYPE -e CC=clang-8 -e CXX=clang++-8 athenaml/build"

mkdir -p "build_$BUILD_TYPE"
$COMMAND_WRAPPER "$GITHUB_WORKSPACE"/scripts/build.py --install-dir="$GITHUB_WORKSPACE"/install_"$BUILD_TYPE" --build-type="$BUILD_TYPE" "$GITHUB_WORKSPACE"/build_"$BUILD_TYPE" "$GITHUB_WORKSPACE"
$COMMAND_WRAPPER cmake --build "$GITHUB_WORKSPACE"/build_"$BUILD_TYPE" --target install
$COMMAND_WRAPPER "$GITHUB_WORKSPACE"/scripts/test.sh "$GITHUB_WORKSPACE"/build_"$BUILD_TYPE"
