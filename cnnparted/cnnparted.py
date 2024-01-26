#! /usr/bin/env python3
import argparse
import torch
import os
import subprocess
import importlib
import numpy as np
import pandas as pd
from typing import Callable

from framework import ConfigHelper, NodeThread, GraphAnalyzer, NSGA2_Optimizer, QuantizationEvaluator
from framework.constants import MODEL_PATH, WORKLOAD_FOLDER


def main(args):
    conf_helper = ConfigHelper(args.conf_file_path)
    config = conf_helper.get_config()
    node_components, link_components = conf_helper.get_system_components()
    accuracy_function = setup_workload(args.run_name, config['neural-network'])
    num_pp = len(node_components) - 1
    n_var = num_pp * 2 + 1

    # Step 1 - Analysis
    ga = GraphAnalyzer(args.run_name, tuple(config['neural-network']['input-size']), args.show_progress)

    # Step 2 - Layer Evaluation
    nodeStats = node_evaluation(ga, node_components, args.run_name, args.show_progress)

    # Step 3 - Find pareto-front
    optimizer = NSGA2_Optimizer(ga, nodeStats, link_components, args.show_progress)
    fixed_sys = True # do not change order of accelerators if true. TODO: add to config file
    sol = optimizer.optimize(fixed_sys)

    # Step 4 - Accuracy Evaluation
    quant = QuantizationEvaluator(ga.torchmodel, ga.input_size, config.get('accuracy'), args.show_progress)
    quant.eval(sol["nondom"][:n_var+1], ga.schedules, accuracy_function)

    # Step 5 - Find best partitioning point
    # objective = conf_helper.get_optimization_objectives(node_components, link_components)

    # Step 6 - Output exploration results
    write_files(args.run_name, n_var, sol, ga.schedules)
    for pareto, sched in sol.items():
        print(pareto, len(sched))


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

def node_evaluation(ga : GraphAnalyzer, node_components : list, run_name : str, progress : bool) -> dict:
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

def write_files(run_name : str, n_var : int, results : dict, schedules : list) -> None:
    rows = []
    for pareto, sched in results.items():
        for sd in sched:
            data = np.append(sd, pareto)
            data[:n_var+1] = data[:n_var+1].astype(float).astype(int)
            data[1] = schedules[int(data[0])][int(data[1])-1]
            rows.append(data)
    df = pd.DataFrame(rows)
    df.to_csv(run_name + "_" + "result.csv", header=False)
    df = pd.DataFrame(schedules)
    df.to_csv(run_name + "_" + "schedules.csv", header=False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='Explore partition point metrics for DNNs')
    parser.add_argument('conf_file_path', help='Path to config file')
    parser.add_argument('run_name',
                        type=str,
                        default='run',
                        help='Name of run')
    parser.add_argument('-p',
                        '--show-progress',
                        action='store_true',
                        help='Show progress of run')
    parser.add_argument('-n',
                        '--num-runs',
                        type=int,
                        default=1,
                        help='Number of runs')
    parser.add_argument('--accuracy', action='store_true', default=False, help='Compute with accuracy')
    args = parser.parse_args()

    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '5'
    np.set_printoptions(precision=2)

    main(args)
