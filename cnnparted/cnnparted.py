#! /usr/bin/env python3
import argparse
import torch
import os
import sys
import subprocess
import importlib
import numpy as np
import pandas as pd
from typing import Callable

from framework import ConfigHelper, NodeThread, GraphAnalyzer, PartitioningOptimizer, RobustnessOptimizer, AccuracyEvaluator
from framework.constants import MODEL_PATH, WORKLOAD_FOLDER


def main(args):
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cid)
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    conf_helper = ConfigHelper(args.conf_file_path)
    config = conf_helper.get_config()
    main_conf = config.get('general')
    node_components, link_components = conf_helper.get_system_components()
    accuracy_function = setup_workload(args.run_name, config['neural-network'])

    # Step 1 - Analysis
    ga = GraphAnalyzer(args.run_name, tuple(config['neural-network']['input-size']), args.p)
    ga.find_schedules(main_conf.get('num_topos'))

    # Step 2 - Robustness Analysis
    q_constr = {}
    if config.get('accuracy'):
        robustnessAnalyzer = RobustnessOptimizer(args.run_name, ga.torchmodel, accuracy_function, config.get('accuracy'), device, args.p)
        q_constr = robustnessAnalyzer.optimize()

    # Step 3 - Layer Evaluation
    nodeStats = node_eval(ga, node_components, args.run_name, args.p)

    # Step 4 - Find pareto-front
    num_pp = main_conf.get('num_pp')
    if num_pp == -1:
        num_pp = len(nodeStats[list(nodeStats.keys())[0]]) - 1
    optimizer = PartitioningOptimizer(ga, num_pp, nodeStats, link_components, args.p)
    n_constr, n_var, sol = optimizer.optimize(q_constr, main_conf)

    # Step 5 - Accuracy Evaluation (only non-dominated solutions)
    if config.get('accuracy'):
        quant = AccuracyEvaluator(ga.torchmodel, nodeStats, config.get('accuracy'), device, args.p)
        quant.eval(sol["nondom"], n_constr, n_var, ga.schedules, accuracy_function)
        for i, p in enumerate(sol["dom"]): # achieving aligned csv file
            sol["dom"][i] = np.append(p, float(0))

    # Step 6 - Find best partitioning point
    # objective = conf_helper.get_optimization_objectives(node_components, link_components)

    # Step 7 - Output exploration results
    write_files(args.run_name, n_constr, n_var, sol, ga.schedules)
    sols = 0
    for pareto, sched in sol.items():
        print(pareto, len(sched))
        sols += len(sched)

    if sols > 0:
        num_real_pp = [int(sched[1]) for sched in sol["nondom"]]
        for i in range(1, max(num_real_pp)+1):
            print(i, "Partition(s):", num_real_pp.count(i))
    else:
        print()
        print("### [CNNParted] No valid partitioning found! ###")
        print()


def setup_workload(run_name : str, model_settings: dict) -> Callable:
    try:
        model = importlib.import_module(
            f"{WORKLOAD_FOLDER}.{model_settings['name']}", package=__package__
        ).model

        accuracy_function = importlib.import_module(
            f"{WORKLOAD_FOLDER}.{model_settings['name']}", package=__package__
        ).accuracy_function

        input_size= model_settings['input-size']

        x = torch.randn(input_size)
        subprocess.check_call(['mkdir', '-p', os.path.join(MODEL_PATH, run_name)])
        filename = os.path.join(MODEL_PATH, run_name, "model.onnx")
        torch.onnx.export(model, x, filename, verbose=False, input_names=['input'], output_names=['output'])

        return accuracy_function

    except KeyError:
        print()
        print('\033[1m' + 'Workload not available' + '\033[0m')
        print()
        quit(1)

def node_eval(ga : GraphAnalyzer, node_components : list, run_name : str, progress : bool) -> dict:
    nodeStats = {}
    node_threads = [
            NodeThread(component.get('id'), ga, component, run_name, progress)
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

def write_files(run_name : str, n_constr : int, n_var : int, results : dict, schedules : list) -> None:
    rows = []
    for pareto, sched in results.items():
        for sd in sched:
            data = np.append(sd, pareto)
            data = data.astype('U256')
            data[0:2] = data[0:2].astype(float).astype(int)
            data[n_constr+1:n_constr+n_var+1] = data[n_constr+1:n_constr+n_var+1].astype(float).astype(int)
            for i in range(n_constr+1,int(n_var/2)+n_constr+1):
                data[i] = schedules[int(data[0])][int(data[i])-1]
            rows.append(data)
        if pareto == "nondom":
            df = pd.DataFrame(rows)
            df.to_csv(run_name + "_" + "result_nondom.csv", header=False)
    df = pd.DataFrame(rows)
    df.to_csv(run_name + "_" + "result_all.csv", header=False)


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

