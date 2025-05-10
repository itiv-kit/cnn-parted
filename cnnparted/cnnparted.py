#! /usr/bin/env python3
import itertools
import argparse
from genericpath import isfile
import tempfile
import shutil
import torch
import os
import sys
import importlib
import numpy as np
from timeit import default_timer as timer

from framework import ConfigHelper
from framework.stages.artifacts import Artifacts
from framework.stages.stage_base import STAGE_DEPENDENCIES;
from framework.constants import MODEL_PATH, ROOT_DIR, WORKLOAD_FOLDER

def parse_pipeline(config):
    stages = config["pipeline"]["stages"]
    
    stage_classes = []
    enabled_stages = []
    package = importlib.import_module("framework.stages")
    for stage in stages:
        stage_class = getattr(package, stage)
        enabled_stages.append(stage_class)
        
        # Check if required stages are available
        if stage_class in STAGE_DEPENDENCIES:
            dependencies_exist = False
            deps = STAGE_DEPENDENCIES[stage_class]
            deps_tup = [(dep, ) if not isinstance(dep, tuple) else dep for dep in deps]
            deps_prod = itertools.product(*deps_tup)
            # Some stages can take either one or more stages as requirements, these are denoted by a tuple
            # E.g. deps = ['GraphAnalysis', ('NodeEvaluation', 'DesignPartitioningOptimization'), 'SystemParser']
            # Construct the cartesian product of these
            for deps_candidate in deps_prod:
                if set(deps_candidate).issubset(enabled_stages):
                    dependencies_exist = True
                    break
            if not dependencies_exist:
                raise RuntimeError(f"Dependencies for \"{stage_class.__name__}\" not found. Required are {deps} to be instantiated previously.")
            
        stage_classes.append(stage_class())

    return stage_classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='Explore partition point metrics for DNNs')
    parser.add_argument('conf_file_path', help='Path to config file')
    parser.add_argument('run_name',
                        type=str,
                        default='run',
                        help='Name of run',)
    parser.add_argument('-p',
                        action='store_true',
                        help='Show progress of run')
    parser.add_argument('-n',
                        '--num-runs',
                        type=int,
                        default=1,
                        help='Number of runs')
    parser.add_argument('--cuda',
                        action='store_true',
                        help='Use CUDA')
    parser.add_argument('--cid',
                        type=int,
                        default=0,
                        help='CUDA ID')
    parser.add_argument('-clp',
                        '--clear-partitioning',
                        action='store_true',
                        help="Use to clean results of partitioning evaluation")
    args = parser.parse_args()

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cid)
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '5'
    np.set_printoptions(precision=2)


    # Setup working directory based on configuration file
    conf_helper = ConfigHelper(args.conf_file_path)
    config = conf_helper.get_config()
    main_conf = config.get('general')
    if work_dir_str := main_conf.get('work_dir'):
        work_dir = work_dir_str
        if os.path.isdir(work_dir) and not main_conf.get("keep_dir", True):
            shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)
    else:
        work_dir_tmp = tempfile.TemporaryDirectory(dir=ROOT_DIR)
        work_dir = work_dir_tmp.name.split(os.path.sep)[-1]
    
    if args.clear_partitioning:
        rn = args.run_name
        files = [rn+"_non_optimals.npy",
                 rn+"_paretos.npy",
                 rn+"_objectives_all.csv",
                 rn+"_objectives_nondom.csv",
                 rn+"_result_all.csv",
                 rn+"_result_nondom.csv",
                 rn+"_robustness.csv"
                ]
        for file in files:
            file_p = os.path.join(ROOT_DIR, work_dir, file)
            if os.path.isfile(file_p):
                os.remove(file_p)
        sys.exit()

        
    # Determine stages that should be run
    stages = parse_pipeline(config)
        
    step_runtime = [timer()]

    # Setup a class to keep track of results of stages
    artifacts = Artifacts(config, vars(args), device, step_runtime)
    artifacts.config["work_dir"] = work_dir

    try:
        for stage in stages:
            stage.run(artifacts)
            step_runtime.append(timer())
    except KeyboardInterrupt:
        print("Interrupted")
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)


