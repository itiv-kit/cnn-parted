#!/bin/bash

export SIM_ROOT_DIR=$PWD

source $PWD/pythonEnv/bin/activate
export LD_LIBRARY_PATH=$PWD/tools/timeloop/lib:${LD_LIBRARY_PATH}