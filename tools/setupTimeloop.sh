#!/bin/bash

echo "Settin up timeloop"


# stop on error
set -e

# Timeloop-Python
if [ ! -d "tools/timeloop/pat" ]; then
    ln -sf ../pat-public/src/pat tools/timeloop/src
fi

scons -C tools/timeloop -j4 --accelergy
