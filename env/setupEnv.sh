#!/bin/bash

export SIM_ROOT_DIR=$PWD

source $PWD/.venv/bin/activate
export LD_LIBRARY_PATH=$PWD/cnnparted/tools/timeloop/lib:${LD_LIBRARY_PATH}
