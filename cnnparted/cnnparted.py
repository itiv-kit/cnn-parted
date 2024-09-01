#! /usr/bin/env python3
import argparse
import tempfile
import shutil
import torch
import os
import sys
import subprocess
import importlib
import numpy as np
import pandas as pd
from typing import Callable
from timeit import default_timer as timer

from framework import ConfigHelper, NodeThread, GraphAnalyzer, PartitioningOptimizer, RobustnessOptimizer, AccuracyEvaluator
from framework.constants import MODEL_PATH, ROOT_DIR, WORKLOAD_FOLDER


def main(args):
    # Step 0 - Initialize CNNParted
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cid)
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    conf_helper = ConfigHelper(args.conf_file_path)
    config = conf_helper.get_config()
    main_conf = config.get('general')
    keep_dir = main_conf.get('keep_dir', True)
    if work_dir_str := main_conf.get('work_dir'):
        work_dir = work_dir_str
        if os.path.isdir(work_dir) and not keep_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)
    else:
        work_dir_tmp = tempfile.TemporaryDirectory(dir=ROOT_DIR)
        work_dir = work_dir_tmp.name.split(os.path.sep)[-1]

    node_components, link_components = conf_helper.get_system_components()
    accuracy_function = setup_workload(args.run_name, config['neural-network'])
    step_runtime = [timer()]

    # Step 1 - Analysis
    ga = GraphAnalyzer(work_dir, args.run_name, tuple(config['neural-network']['input-size']), args.p)
    ga.find_schedules(main_conf.get('num_topos'))
    step_runtime.append(timer())

    # Step 2 - Robustness Analysis
    q_constr = {}
    if config.get('accuracy').get('robustness'):
        robustnessAnalyzer = RobustnessOptimizer(work_dir, args.run_name, ga.torchmodel, accuracy_function, config.get('accuracy'), device, args.p)
        q_constr = robustnessAnalyzer.optimize()
    step_runtime.append(timer())

    # Step 3 - Layer Evaluation
    nodeStats = node_eval(ga, node_components, work_dir, args.run_name, args.p)
    num_platforms = len(nodeStats)
    step_runtime.append(timer())

    # Step 4 - Find pareto-front
    num_pp = main_conf.get('num_pp')
    if num_pp == -1:
        num_pp = len(nodeStats[list(nodeStats.keys())[0]]["eval"]["design_0"]["layers"].keys()) - 1
    elif num_platforms == 1:
        num_pp = 0
    optimizer = PartitioningOptimizer(ga, num_pp, nodeStats, link_components, args.p)
    n_constr, n_var, sol = optimizer.optimize(q_constr, main_conf)
    step_runtime.append(timer())

    # Step 5 - Accuracy Evaluation (only non-dominated solutions)
    if accuracy_cfg := config.get('accuracy'):
        print("Found: ")
        for pareto, sched in sol.items():
            print(pareto, len(sched))
        print("Evaluating accuracy...")

        quant = AccuracyEvaluator(ga.torchmodel, nodeStats, accuracy_cfg, device, args.p)
        quant.eval(sol["nondom"], n_constr, n_var, ga.schedules, accuracy_function)
        for i, p in enumerate(sol["dom"]): # achieving aligned csv file
            sol["dom"][i] = np.append(p, float(0))
    step_runtime.append(timer())

    # Step 6 - Output exploration results
    write_files(work_dir, args.run_name, n_constr, n_var, sol, ga.schedules, num_platforms)
    num_pp_schemes = write_log_file(work_dir, args.run_name, step_runtime, sol, num_platforms)

    if num_pp_schemes == 0:
        print()
        print("### [CNNParted] No valid partitioning found! ###")
        print()


def setup_workload(run_name : str, model_settings: dict) -> Callable:
    try:
        filename = os.path.join(MODEL_PATH, run_name, "model.onnx")
        subprocess.check_call(['mkdir', '-p', os.path.join(MODEL_PATH, run_name)])

        if os.path.isfile(model_settings['name']) and model_settings['name'].endswith(".onnx"):
            shutil.copy(model_settings['name'], filename)

            #TODO This is just a temporary workaround, accuracy eval currently not used
            accuracy_function = importlib.import_module(
                f"{WORKLOAD_FOLDER}.alexnet", package=__package__
            ).accuracy_function
            return accuracy_function


        model = importlib.import_module(
            f"{WORKLOAD_FOLDER}.{model_settings['name']}", package=__package__
        ).model

        accuracy_function = importlib.import_module(
            f"{WORKLOAD_FOLDER}.{model_settings['name']}", package=__package__
        ).accuracy_function

        input_size= model_settings['input-size']
        x = torch.randn(input_size)
        torch.onnx.export(model, x, filename, verbose=False, input_names=['input'], output_names=['output'])

        return accuracy_function

    except KeyError:
        print()
        print('\033[1m' + 'Workload not available' + '\033[0m')
        print()
        quit(1)

def node_eval(ga : GraphAnalyzer, node_components : list, work_dir: str, run_name : str, progress : bool) -> dict:
    nodeStats = {}
    node_threads = [
            NodeThread(component.get('id'), ga, component, work_dir, run_name, progress)
            for component in node_components
        ]

    for t in node_threads:
        if not t.config.get("timeloop"):
            t.start()

    for t in node_threads:
        if t.config.get("timeloop"): # run them simply on main thread
            t.run()
        else:
            t.join()

    for node_thread in node_threads:
        id,stats = node_thread.getStats()
        nodeStats[id] = stats

    return nodeStats

def write_files(work_dir: str, run_name : str, n_constr : int, n_var : int, results : dict, schedules : list, num_platforms: int) -> None:
    rows = []
    idx_max = 1 + num_platforms + 1
    for pareto, sched in results.items():
        for sd in sched:
            data = np.append(sd, pareto)
            data = data.astype('U256')
            data[0:idx_max] = data[0:idx_max].astype(float).astype(int)
            data[n_constr+1:n_constr+n_var+1] = data[n_constr+1:n_constr+n_var+1].astype(float).astype(int)
            for i in range(n_constr+1,int(n_var/2)+n_constr+1):
                data[i] = schedules[int(data[0])][int(data[i])-1]
            rows.append(data)
        if pareto == "nondom":
            df = pd.DataFrame(rows)
            df.to_csv( os.path.join(work_dir, run_name + "_" + "result_nondom.csv"), header=False)
    df = pd.DataFrame(rows)
    df.to_csv( os.path.join(work_dir, run_name + "_" + "result_all.csv"), header=False)

def write_log_file(work_dir: str, run_name : str, step_runtimes : list, sol : dict, num_platforms : int) -> int:
    log_file = os.path.join(work_dir, run_name + ".log")
    f = open(log_file, "a")

    # Runtime of each step
    step_runtimes = [x - step_runtimes[0] for x in step_runtimes[1:]]
    t_prev = 0
    for i, x in enumerate(step_runtimes, 1):
        f.write("Step " + str(i) +  ": " + str(x - t_prev) + ' s \n')
        t_prev = x

    # Number of solutions found
    num_pp_schemes = 0
    for pareto, scheme in sol.items():
        f.write(str(pareto) + ": " + str(len(scheme)) + '\n')
        num_pp_schemes += len(scheme)
    if num_pp_schemes > 0:
        num_real_pp = [int(scheme[num_platforms+1]) for scheme in sol["nondom"]]
        for i in range(0, max(num_real_pp)+1):
            f.write(str(i) + " Partitioning Point(s): " + str(num_real_pp.count(i)) + '\n')
    else:
        f.write("No valid partitioning found! \n")

    f.close()
    return num_pp_schemes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='Explore partition point metrics for DNNs')
    parser.add_argument('conf_file_path', help='Path to config file')
    parser.add_argument('run_name',
                        type=str,
                        default='run',
                        help='Name of run')
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
    args = parser.parse_args()

    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '5'
    np.set_printoptions(precision=2)

    try:
        main(args)
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)

