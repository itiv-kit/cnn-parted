#!/bin/bash

# stop on error
set -e

function usage() {
    cat <<USAGE

    Usage: $0 [--skip-timeloop]

    Options:
        --skip-timeloop:      skip installation of timeloop
USAGE
    exit 1
}

SKIP_TIMELOOP=false

while [ "$1" != "" ]; do
    case $1 in
    --skip-timeloop)
        SKIP_TIMELOOP=true
        ;;
    -h | --help)
        usage
        ;;
    *)
        usage
        exit 1
        ;;
    esac
    shift
done

# Basic python setup
python3 -m venv pythonEnv
source pythonEnv/bin/activate
pip3 install --upgrade pip
pip3 install setuptools
pip3 install libconf
pip3 install numpy
pip3 install pyyaml

# PyTorch
pip3 install torchvision    # installs typing-extensions, dataclasses, torch, pillow at once
pip3 install torchinfo


if [[ $SKIP_TIMELOOP == false ]]; then
    # Pull submodules
    git submodule update --init --recursive

    cd tools

    # Cacti
    cd cacti
    make
    chmod -R 777 .
    cd ..

    # Timeloop-Python
    cd timeloop/src
    if [ ! -d "pat" ]; then
        ln -s ../pat-public/src/pat .
    fi
    cd ..
    scons -j4 --accelergy
    cd ..

    # Accelergy
    cd accelergy
    pip3 install .
    cd ..
    cd accelergy-aladdin-plug-in
    pip3 install .
    cd ..
    cd accelergy-cacti-plug-in
    pip3 install .
    cd ..
    cd accelergy-table-based-plug-ins
    pip3 install .
fi
