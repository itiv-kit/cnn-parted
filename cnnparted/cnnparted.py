#! /usr/bin/env python3

import yaml
import argparse
import importlib

from framework import DNNAnalyzer, ModuleThreadInterface, NodeThread, LinkThread, Evaluator, QuantizationEvaluator

MODEL_FOLDER = "workloads"

def load_config(fname : str) -> dict:
    with open(fname) as f:
        return yaml.load(f, Loader = yaml.SafeLoader)

def setup_workload(model_settings: dict) -> tuple:
    model = importlib.import_module(
        f"{MODEL_FOLDER}.{model_settings['name']}", package=__package__
    ).model

    accuracy_function = importlib.import_module(
        f"{MODEL_FOLDER}.{model_settings['name']}", package=__package__
    ).accuracy_function

    return model, accuracy_function

def main():
    parser = argparse.ArgumentParser(description='Explore partition point metrics for DNNs')
    parser.add_argument('conf_file_path', help='Path to config file')
    parser.add_argument('run_name', type=str, default='run', help='Name of run')
    parser.add_argument('-p', '--show-progress', action='store_true',
                        help='Show progress of run')
    parser.add_argument('-n', '--num-runs', type=int, default=1,
                        help='Number of runs')
    args = parser.parse_args()

    config = load_config(args.conf_file_path)

    try:
        model, accuracy_function = setup_workload(config['neural-network'])
    except KeyError:
        print()
        print('\033[1m' + 'Workload not available' + '\033[0m')
        print()
        quit(1)

    try:
        max_size = config['constraints']['max_out_size']
    except KeyError:
        max_size = 0

    dnn = DNNAnalyzer(model, (tuple(config['neural-network']['input-size'])), max_size)
    if len(dnn.partpoints_filtered) == 0:
        quit(1)

    accStats = {}
    if 'accuracy' in config.keys():
        accEval = QuantizationEvaluator(model, dnn, config.get('accuracy'), accuracy_function)
        accStats = accEval.get_stats()

    sensorStats = {}
    linkStats   = {}
    edgeStats   = {}
    for i in range(0, args.num_runs):
        threads : list[ModuleThreadInterface] = [
            NodeThread('sensor', dnn, config.get('sensor'), args.run_name, args.show_progress),
            LinkThread('link'  , dnn, config.get('link')  , args.run_name, args.show_progress),
            NodeThread('edge'  , dnn, config.get('edge')  , args.run_name, args.show_progress)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        sensorStats[i] = threads[0].getStats()
        linkStats[i]   = threads[1].getStats()
        edgeStats[i]   = threads[2].getStats()

    e = Evaluator(dnn, sensorStats, linkStats, edgeStats, accStats)
    e.print_sim_time()
    e.export_csv(args.run_name)


if __name__ == '__main__':
    main()
