#!/bin/bash
# script tesing project with google test framework

# make project 
make

# create build directory to make tests
cd ./tests
mkdir build

# copy Make Lists from bridge dir
cp ../bridge/CMakeLists_GTEST.txt ./build/CMakeLists.txt
echo

# make tests
echo Make Tests
cd build 
cmake CMakeLists.txt
make
echo 

# run tests
echo Running Tests 
./runTests


