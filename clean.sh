#!/bin/bash
# script restoring original view of project

echo make clean
make clean

# delete doxygen
echo delete doxygen
rm -f CMakeLists.txt Doxyfile
rm -rf cbuild/

# delete gtest
echo delete gtest
rm -rf ./tests/build