#!/bin/bash

echo "Setting up timeloop"


# stop on error
set -e

# Timeloop-Python
if [ ! -d "cnnparted/tools/timeloop/pat" ]; then
    ln -sf ../pat-public/src/pat cnnparted/tools/timeloop/src
fi

scons -C cnnparted/tools/timeloop -j4 --accelergy
