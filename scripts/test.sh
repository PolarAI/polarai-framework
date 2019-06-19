#!/bin/bash

if [[ -z "$1" ]]
then
    cd build
else
    cd $1
fi

ctest -T Test
