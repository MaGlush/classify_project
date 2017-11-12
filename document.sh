#!/bin/bash
# script documenting project with doxygen

# get CMakeLists
cp ./bridge/CMakeLists_DOXYGEN.txt ./CMakeLists.txt

# get Doxyfile
cp ./bridge/Doxyfile ./Doxyfile

#make project
mkdir cbuild; cd cbuild; cmake ../; make; cd ..
