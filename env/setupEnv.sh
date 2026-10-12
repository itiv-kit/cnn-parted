#!/bin/bash

export PROJECT_ROOT=$PWD

source $PWD/.venv/bin/activate
#export LD_LIBRARY_PATH=$PWD/cnnparted/tools/timeloop/lib:${LD_LIBRARY_PATH}
#export LD_LIBRARY_PATH=/usr/local/lib:$PWD/cnnparted/tools/accelergy-timeloop-infrastructure/src/timeloop/lib:${LD_LIBRARY_PATH}

export CPATH=$PROJECT_ROOT/.external_libs/usr/local/include:$CPATH
export LIBRARY_PATH=$PROJECT_ROOT/.external_libs/usr/local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$PROJECT_ROOT/.external_libs/usr/local/lib:$PWD/cnnparted/tools/accelergy-timeloop-infrastructure/src/timeloop/lib:${LD_LIBRARY_PATH}
