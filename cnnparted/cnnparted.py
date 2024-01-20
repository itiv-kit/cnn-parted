#! /usr/bin/env python3
import argparse
from framework.Optimizer.NSGA2 import NSGA2_Optimizer
from framework.helpers.ConfigHelper import ConfigHelper
import torch
from framework.constants import MODEL_PATH, WORKLOAD_FOLDER
import os
import subprocess
import importlib
from framework import NodeThread, GraphAnalyzer


def main(args):
    conf_helper = ConfigHelper(args.conf_file_path)
    config = conf_helper.get_config()
    node_components, link_components = conf_helper.get_system_components()

    accuracy_function = setup_workload(args.run_name, config['neural-network'])

    ga = GraphAnalyzer(args.run_name, tuple(config['neural-network']['input-size']))

    nodeStats = node_evaluation(ga, node_components, args.run_name, args.show_progress)

    nsga2 = NSGA2_Optimizer(ga, nodeStats, link_components)
    objective = conf_helper.get_optimization_objectives(node_components, link_components)
    nsga2.optimize(objective)


def setup_workload(run_name : str, model_settings: dict) -> tuple:
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

    main(args)



    # dnn = DNNAnalyzer(MODEL_PATH, tuple(config['neural-network']['input-size']), conf_helper)#,node_components,link_components ,constraints)

    # if len(dnn.partpoints_filtered) == 0:
    #     print("ERROR: No partitioning points found. Please check your system constraints: max_memory_size, max_out_size ")
    #     quit(1)


    # accStats = {}
    # if args.accuracy:
    #     if 'accuracy' in config.keys():
    #         accEval = QuantizationEvaluator(dnn, config.get('accuracy'), node_components, accuracy_function, args.show_progress)
    #         accStats = accEval.get_stats()

    # objective = conf_helper.get_optimization_objectives(node_components,link_components)
    # first_component_id = node_components[0]['id']

    # nodeStats={}
    # linkStats={}

    # for i in range (0, args.num_runs):

    #     node_threads = [
    #             NodeThread(component.get('id'), dnn, component,component['id'] != first_component_id, args.run_name, args.show_progress)
    #             for component in node_components
    #         ]
    #     link_threads = [LinkComputationThread(component.get('id'), dnn, component,False, args.run_name, args.show_progress)
    #              for component in link_components

    #         ]

    #     for t in node_threads:
    #         if not t.config.get("timeloop"):
    #             t.start()

    #     for t in link_threads:
    #         t.start()

    #     for t in node_threads:
    #         if t.config.get("timeloop"): # run them simply on main thread
    #             t.run()

    #     for t in node_threads and link_threads:
    #         if not t.config.get("timeloop"):
    #             t.join()

    #     for node_thread in node_threads:
    #         id,stats = node_thread.getStats()
    #         if id not in nodeStats:
    #             nodeStats[id] = {}
    #         nodeStats[id][i] = stats

    #     for link_thread in  link_threads:
    #         id,stats = link_thread.getStats()
    #         if id not in linkStats:
    #             linkStats[id] = {}
    #         linkStats[id][i] = stats

    # e = Evaluator(dnn, nodeStats, linkStats, accStats)
    # e.print_sim_time()
    # e.export_csv(args.run_name)

    # nodes = e.get_all_layer_stats()

    # if len(nodes) == 0:
    #     print("No benificial partitioning point found: check the bandwidth and memory constraints: ")
    #     sys.exit()


    # nsga2 = NSGA2_Optimizer(nodes)
    # optimizer,paretos = nsga2.optimize(objective)

    # data = {
    # "best partitioning Point": optimizer,
    # "Paretos": [layer['layer'] for layer in paretos]
    # }

    # with open(args.run_name + '_optimals.json', 'w') as jsonfile:
    #     json.dump(data, jsonfile, indent=4)

    # print("best partitioning Point: ")
    # print(optimizer)
    # print("Paretos:")
    # for layer in paretos:
    #     print(layer['layer'])
