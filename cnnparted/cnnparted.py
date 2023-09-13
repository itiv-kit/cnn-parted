#! /usr/bin/env python3

import argparse
from framework import DNNAnalyzer, ModuleThreadInterface, NodeThread, LinkThread, Evaluator #, Dif_Evaluator
from framework.Optimizer.NSGA2 import NSGA2_Optimizer
from framework.helpers.ConfigHelper import ConfigHelper
#from framework.node.Dramsim import Dramsim
import yaml
import torch
from framework.constants import MODEL_PATH
import sys

import importlib

from framework import DNNAnalyzer, ModuleThreadInterface, NodeThread, LinkThread, Evaluator#, QuantizationEvaluator

MODEL_FOLDER = "workloads"

def setup_workload(model_settings: dict) -> tuple:
    model = importlib.import_module(
        f"{MODEL_FOLDER}.{model_settings['name']}", package=__package__
    ).model
    input_size= model_settings['input-size']
    x = torch.randn(input_size)
    torch.onnx.export(model, x, MODEL_PATH, verbose=False, input_names=['input'], output_names=['output'])

    # accuracy_function = importlib.import_module(
    #     f"{MODEL_FOLDER}.{model_settings['name']}", package=__package__
    # ).accuracy_function

    return MODEL_PATH,None# accuracy_function

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

    conf_helper = ConfigHelper(args.conf_file_path)
    config = conf_helper.get_config()
    #model = conf_helper.get_model(main)
    constraints = conf_helper.get_constraints()
    

    try:
        model_path, accuracy_function = setup_workload(config['neural-network'])
    except KeyError:
        print()
        print('\033[1m' + 'Workload not available' + '\033[0m')
        print()
        quit(1)



    dnn = DNNAnalyzer(model_path, (tuple(config['neural-network']['input-size'])),
                      constraints)
    if len(dnn.partpoints_filtered) == 0:
        quit(1)

    # accStats = {}
    # if 'accuracy' in config.keys():
    #     accEval = QuantizationEvaluator(model, dnn, config.get('accuracy'), accuracy_function, args.show_progress)
    #     accStats = accEval.get_stats()

      
    node_components,link_components = conf_helper.get_system_components()
    first_component_id = node_components[0]['id']

    node_threads = [
                NodeThread(component.get('name', "Partition"+str(component.get('id', 'N/A'))).lower(), dnn, component,component['id'] != first_component_id, args.run_name, args.show_progress)
                for component in node_components
            ]
    link_threads = [
                LinkThread(component.get('name', "Link"+str(component.get('id', 'N/A'))).lower(), dnn, component,False, args.run_name, args.show_progress)
                for component in link_components
            ]

    ####Test####
    nodeStats={}
    linkStats={}
    for i in range (0, args.num_runs):
        node_threads = [
                NodeThread(component.get('name', "Partition"+str(component.get('id', 'N/A'))).lower(), dnn, component,component['id'] != first_component_id, args.run_name, args.show_progress)
                for component in node_components
            ]
        link_threads = [
                LinkThread(component.get('name', "Link"+str(component.get('id', 'N/A'))).lower(), dnn, component,False, args.run_name, args.show_progress)
                for component in link_components
            ]
        
        for t in node_threads:
            t.start()
        for t in link_threads:
            t.start()

        for t in node_threads:
            t.join()
        for t in link_threads:
            t.join()
        
        
        for index, node_thread in enumerate(node_threads):
            stats = node_thread.getStats()
            if index not in nodeStats:
                nodeStats[index] = {}
            nodeStats[index][i] = stats

        for index,link_thread in  enumerate(link_threads):
            stats = link_thread .getStats()
            if index not in linkStats:
                linkStats[index] = {}
            linkStats[index][i] = stats


    ############
    #threads = node_threads + link_threads

    # for i in range(0, args.num_runs):
    #     for t in threads:
    #         t.start()
    #     for t in threads:
    #         t.join()

    #     sensorStats[i] = threads[0].getStats()
    #     linkStats[i] = threads[2].getStats()
    #     edgeStats[i] = threads[1].getStats()

    #Evaluator should be modified to support more than 2 accelerators setting
    e = Evaluator(dnn, nodeStats[0], linkStats[0], nodeStats[1], {})
    e.print_sim_time()
    e.export_csv(args.run_name)
 
    nodes = e.get_all_layer_stats()
    
    if len(nodes) == 0:
        print("No benificial partitioning point found: check the bandwidth and memory constraints: ")
        sys.exit()

    nsga2 = NSGA2_Optimizer(nodes)
    optimizer = nsga2.optimize(conf_helper.get_optimization_objectives())

    print("best partioning Point: ")
    print(optimizer)
    


if __name__ == '__main__':
    main()
