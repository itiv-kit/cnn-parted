#! /usr/bin/env python3

import argparse
from framework import DNNAnalyzer, ModuleThreadInterface, NodeThread, LinkThread, Evaluator, Dif_Evaluator
from framework.Optimizer.NSGA2 import NSGA2_Optimizer
from framework.helpers.ConfigHelper import ConfigHelper


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
    model = conf_helper.get_model(main)
    constraints = conf_helper.get_constraints()

    dnn = DNNAnalyzer(model, (tuple(config['neural-network']['input-size'])),
                      constraints)
    if len(dnn.partpoints_filtered) == 0:
        quit(1)

    sensorStats = {}
    linkStats = {}
    edgeStats = {}
    for i in range(0, args.num_runs):
        threads: list[ModuleThreadInterface] = [
            NodeThread('sensor', dnn, config.get('sensor'), args.run_name,
                       args.show_progress),
            LinkThread('link', dnn, config.get('link'), args.run_name,
                       args.show_progress),
            NodeThread('edge', dnn, config.get('edge'), args.run_name,
                       args.show_progress)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        sensorStats[i] = threads[0].getStats()
        linkStats[i] = threads[1].getStats()
        edgeStats[i] = threads[2].getStats()

    e = Evaluator(dnn, sensorStats, linkStats, edgeStats)
    e.print_sim_time()
    e.export_csv(args.run_name)
    #de = Dif_Evaluator(dnn, sensorStats)
    #de.save_stats(args.run_name, 'otherdnns.csv')

    nodes = e.get_all_layer_stats()
    nsga2 = NSGA2_Optimizer(nodes)
    optimizer = nsga2.optimize(conf_helper.get_optimization_objectives())

    print("best partioning Point: ")
    print(optimizer)
    


if __name__ == '__main__':
    main()
