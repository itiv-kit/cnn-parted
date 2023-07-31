#!/bin/bash


# stop on error
set -e


# Store the current working directory
current_dir=$(pwd)

# Create the build directory inside the tools/DRAMsim3 directory
mkdir -p tools/DRAMsim3/build
cd tools/DRAMsim3/build

# Run CMake to generate build files
cmake ..

# Build dramsim3 library and executables
make -j4

# Optionally, install the built targets (uncomment the line below if needed)
# make install

# Return to the original directory
cd "$current_dir"
