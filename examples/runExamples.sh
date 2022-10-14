#!/bin/bash

# stop on error
set -e

cd $SIM_ROOT_DIR
python cnnparted.py -p examples/squeezenet1_1.yaml squeezenet1_1
python cnnparted.py -p examples/fcn_resnet50.yaml fcn_resnet50
python cnnparted.py -p examples/googlenet.yaml googlenet
