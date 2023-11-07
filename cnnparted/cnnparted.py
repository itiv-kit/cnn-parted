#! /usr/bin/env python3
import argparse
from framework import DNNAnalyzer, ModuleThreadInterface, NodeThread, LinkThread, Evaluator#, Dif_Evaluator
from framework.Optimizer.NSGA2 import NSGA2_Optimizer
from framework.helpers.ConfigHelper import ConfigHelper
import torch
from framework.constants import MODEL_PATH
import sys
import os
import json
import importlib
from framework import DNNAnalyzer, NodeThread, LinkThread, Evaluator ,MemoryNodeThread, QuantizationEvaluator
from framework.model.modelHelper import modelHelper
MODEL_FOLDER = "workloads"

def setup_workload(model_settings: dict) -> tuple:
    model = importlib.import_module(
        f"{MODEL_FOLDER}.{model_settings['name']}", package=__package__
    ).model

    accuracy_function = importlib.import_module(
        f"{MODEL_FOLDER}.{model_settings['name']}", package=__package__
    ).accuracy_function

    input_size= model_settings['input-size']

    x = torch.randn(input_size)
    torch.onnx.export(model, x, MODEL_PATH, verbose=False, input_names=['input'], output_names=['output'])

    return model, accuracy_function

def main():
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
    args = parser.parse_args()
    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '5'
    conf_helper = ConfigHelper(args.conf_file_path)
    config = conf_helper.get_config()
    node_components,link_components = conf_helper.get_system_components()



    try:
        model, accuracy_function = setup_workload(config['neural-network'])
    except KeyError:
        print()
        print('\033[1m' + 'Workload not available' + '\033[0m')
        print()
        quit(1)

    sim_times = []
    mem_sim_times=[]

    dnn = DNNAnalyzer(MODEL_PATH, tuple(config['neural-network']['input-size']), conf_helper)#,node_components,link_components ,constraints)

    if len(dnn.partpoints_filtered) == 0:
        print("ERROR: No partitioning points found. Please check your system constraints: max_memory_size, max_out_size ")
        quit(1)


    accStats = {}
    if 'accuracy' in config.keys():
        accEval = QuantizationEvaluator(dnn, config.get('accuracy'), accuracy_function, args.show_progress)
        accStats = accEval.get_stats()
        print(accStats)

    objective = conf_helper.get_optimization_objectives(node_components,link_components)
    first_component_id = node_components[0]['id']

    nodeStats={}
    linkStats={}
    memoryStats={}

    memory_threads= [MemoryNodeThread(first_component_id-1, dnn, None,False, args.run_name, args.show_progress)]
    for memt in memory_threads:
            memt.start()

    for i in range (0, args.num_runs):

        node_threads = [
                NodeThread(component.get('id'), dnn, component,component['id'] != first_component_id, args.run_name, args.show_progress)
                for component in node_components
            ]
        link_threads = [
                LinkThread(component.get('id'), dnn, component,False, args.run_name, args.show_progress)
                for component in link_components
            ]


        for t in node_threads:
            if not t.config.get("timeloop"):
                t.start()

        for t in link_threads:
            t.start()

        for t in node_threads:
            if t.config.get("timeloop"): # run them simply on main thread
                t.run()

        for t in node_threads and link_threads:
            if not t.config.get("timeloop"):
                t.join()

        for node_thread in node_threads:
            id,stats = node_thread.getStats()
            if id not in nodeStats:
                nodeStats[id] = {}
            nodeStats[id][i] = stats

        for link_thread in  link_threads:
            id,stats = link_thread .getStats()
            if id not in linkStats:
                linkStats[id] = {}
            linkStats[id][i] = stats


    for memt in memory_threads:
            memt.join()

    for mem_thread in memory_threads:
        id,stats = mem_thread.getStats()
        if id not in memoryStats:
            memoryStats[id]={}
        memoryStats[id][0]=stats


    e = Evaluator(dnn, nodeStats, linkStats, memoryStats,accStats)
    e.print_sim_time()
    e.export_csv(args.run_name)

    nodes = e.get_all_layer_stats()

    if len(nodes) == 0:
        print("No benificial partitioning point found: check the bandwidth and memory constraints: ")
        sys.exit()


    nsga2 = NSGA2_Optimizer(nodes)
    optimizer,paretos = nsga2.optimize(objective)

    data = {
    "best partitioning Point": optimizer,
    "Paretos": [layer['layer'] for layer in paretos]
    }

    with open(args.run_name + '_optimals.json', 'w') as jsonfile:
        json.dump(data, jsonfile, indent=4)

    print("best partitioning Point: ")
    print(optimizer)
    print("Paretos:")
    for layer in paretos:
        print(layer['layer'])


if __name__ == '__main__':
    main()
