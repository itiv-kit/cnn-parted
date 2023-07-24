#!/bin/bash

# stop on error
set -e

# Timeloop-Python
if [ ! -d "tools/timeloop/pat" ]; then
    ln -sf ../pat-public/src/pat tools/timeloop/src
fi

scons-3.6 -C tools/timeloop -j4 --accelergy
