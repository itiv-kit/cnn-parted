import csv
import os
import numpy as np
import itertools
import pandas as pd
import argparse
import glob
import matplotlib.pyplot as plt

from framework.constants import ROOT_DIR
from framework.helpers.DesignMetrics import calc_metric, get_metric_info, SUPPORTED_METRICS
from framework.helpers.Visualizer import plotMetricPerConfigPerLayer, COLOR_SEQUENCE, MARKER_SEQUENCE

def objective_to_str(objective: str):
    if objective in ["latency", "energy", "throughput", "area"]:
        return objective.capitalize()
    elif objective == "link_latency":
        return "Link Latency"
    elif objective == "link_energy":
        return "Link Energy"
    else:
        return "undef"

def objective_to_unit(objective: str):
    if objective in ["latency", "link_latency"]:
        return "$ms$"
    elif objective in ["energy", "link_energy"]:
        return "$mJ$"
    elif objective == "throughput":
            return "$FPS$"
    elif objective == "area":
        return "$mm^2$"
    else:
        return "unit"


# These functions are all copied from somewhere in cnnparted, mostly NodeThread.py
def read_layer_csv(filename: str) -> dict:
    designs = {}
    stats = {}

    with open(filename, 'r', newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        current_design_tag = "design_0"
        stats["layers"] = {}
        for row in reader:
            if row["Design Tag"] != current_design_tag:
                designs[current_design_tag] = stats
                current_design_tag = row["Design Tag"]

                stats = {}
                stats["layers"] = {}

            layer_name = row['Layer']
            stats["layers"][layer_name] = {}
            stats["layers"][layer_name]['latency'] = float(row['Latency [ms]'])
            stats["layers"][layer_name]['energy'] = float(row['Energy [mJ]'])
            stats["layers"][layer_name]['area'] = float(row['Area [mm2]'])

    # Append stats for final design
    tag = row["Design Tag"]
    designs[tag] = stats
    return designs


def prune_accelerator_designs(stats: list[dict], top_k: int, metric: str, is_dse: bool):
    # If there are less designs than top_k simply return the given list
    if len(stats) <= top_k or not is_dse:
        pruned_stats = []
        for design in stats:
            tag = design["tag"] 
            layers = design["layers"]
            #arch_config = design["arch_config"]
            pruned_stats.append({"tag": tag, "layers": layers})
        return stats

    # The metric_per_design array has this structure, with
    # every cell holding EAP, EDP or some other metric:
    #  x | l0 | l1 | l2 | l3 |
    # ------------------------
    # d0 | ...| ...| ...| ...|
    # d1 | ...| ...| ...| ...|
    metric_per_design = []
    energy_per_design = []
    latency_per_design = []
    area_per_design = []

    for design in stats:
        tag = design["tag"]
        layers = design["layers"]
        energy_per_layer = []
        latency_per_layer = []
        for name, layer in layers.items():
            energy_per_layer.append(layer["energy"])    
            latency_per_layer.append(layer["latency"])    

        energy_per_design.append(energy_per_layer)
        latency_per_design.append(latency_per_layer)
        area_per_design.append([layer["area"]])

    metric_per_design = calc_metric(np.array(energy_per_design), np.array(latency_per_design), np.array(area_per_design), metric, reduction=False)

    # Now, we need to find the top_k designs per layer
    design_candidates = []
    for col in metric_per_design.T:
        metric_for_layer = col.copy()
        metric_for_layer = np.argsort(metric_for_layer)

        for i in metric_for_layer[0:top_k]:
            design_candidates.append(f"design_{i}")

    design_candidates = np.unique(design_candidates) 

    # Remove all designs that have not been found to be suitable design candidates
    pruned_stats = []
    for design in stats:
        if design["tag"] in design_candidates: 
            tag = design["tag"] 
            layers = design["layers"]
            #arch_config = design["arch_config"]
            pruned_stats.append({"tag": tag, "layers": layers})

    return pruned_stats


def apply_platform_constraints(stats: list[dict], constraints: dict):
    max_energy = constraints.get("energy", np.inf)
    max_latency = constraints.get("latency", np.inf)
    max_area = constraints.get("area", np.inf)

    energy_per_design = []
    latency_per_design = []
    area_per_design = []
    for design in stats:
        layers = design["layers"]
        energy_per_layer = []
        latency_per_layer = []
        for name, layer in layers.items():
            energy_per_layer.append(layer["energy"])    
            latency_per_layer.append(layer["latency"])    

        energy_per_design.append(energy_per_layer)
        latency_per_design.append(latency_per_layer)
        area_per_design.append([layer["area"]])

    energy_per_design = np.array(energy_per_design)
    latency_per_design = np.array(latency_per_design)
    area_per_design = np.array(area_per_design)
    
    total_energy = calc_metric(energy_per_design, latency_per_layer, area_per_design, "energy", reduction= True)
    total_latency = calc_metric(energy_per_design, latency_per_layer, area_per_design, "latency", reduction= True)
    total_area = area_per_design

    constrained_stats = [design for idx, design in enumerate(stats) if (total_area[idx] <= max_area and total_latency[idx] <= max_latency and total_energy[idx] <= max_energy)]
    if not constrained_stats:
        raise ValueError("After applying constraints no designs remain!")

    return constrained_stats

def get_individual_metrics(objectives):
    latency      = objectives[:, 0]
    energy       = objectives[:, 1]
    throughput   = objectives[:, 2]
    area         = objectives[:, 3]
    link_latency = objectives[:, 4]
    link_energy  = objectives[:, 5]
    return latency, energy, throughput, area, link_latency, link_energy

def stats_to_dict(stats: list[dict]) -> dict:
    statsdict = {}
    for i, stat in enumerate(stats):
        key = f"design_{i}"
        design = {k: v for k, v in stat.items()}
        statsdict[key] = design
    return statsdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser( prog='CNNParted Plotter for Evaluation')
    parser.add_argument("-in", "--indir", type=str, help="Path to the evaluation directory")
    parser.add_argument("-out", "--outdir", type=str, help="Path to which results will be saved")
    parser.add_argument("-d", "--designs", type=int, nargs="+", help="Which designs to consider, defaults to all", default=None)
    parser.add_argument("-t", "--plottype", choices=["metric", "partitioning", "compare_base", "compare_multiple", "print_opt"], required=True)
    parser.add_argument("-s", "--system", choices = ["eyeriss_like", "simba_like", "gemmini_like"], nargs="+", help="Accelerators used in the system in order: [platform_0, platform_1, ...]")
    parser.add_argument("-c", "--constraints", type=str, help="Input yaml file that was used to start the simulation")
    parser.add_argument("-b", "--basedir", type=str, help="Directory which contains evaluation of baseline evaluation", default=None)
    parser.add_argument("-o", "--otherdirs", type=str, nargs="+", help="Directories to multiple other results which are also used for comparison")
    parser.add_argument("-l", "--labels", type=str, nargs="+", help="Labels for plots. Only used when plottype==compare_multiple", required=False, default=None)
    parser.add_argument("-n", "--network", type=str, help="String with network name")
    args = parser.parse_args()

    # Plot settings
    all_objectives = ["latency", "energy", "throughput", "area", "link_latency", "link_energy"]
    plot_combinations: list[tuple[str, str]] = list(itertools.permutations(all_objectives, 2)) #combinations

    assert(args.indir != args.outdir)
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
    outdir_figs = os.path.join(os.getcwd(), args.outdir, "figures")
    if not os.path.isdir(outdir_figs):
        os.makedirs(outdir_figs, exist_ok=True)

    
    #system_eval_dir = os.path.join(ROOT_DIR, args.indir, "system_evaluation")
    #num_platforms = len(glob.glob(system_eval_dir))
    num_pp = 1 # hardcoded for ASPDAC evaluation


    # Number of variables for the PartitioningOptimizer
    n_var = num_pp * 2 + 1 #x_len
    n_obj = 6 # f_len; latency, energy, throughput, area + link latency + link energy
    n_constr = (num_pp + 1) + 1 + (num_pp + 1) * 2 + (num_pp + 1) * 2 #g_len; num_accelerator_platforms + num_real_pp + latency/energy per partition + latency/energy per link

    # This evaluated every design layer by layer and generates according plots
    if args.plottype == "metric":
        # Read Timeloop results
        in_csv = glob.glob(os.path.join(ROOT_DIR, args.indir, "*"+args.system[0]+"_tl_layers.csv"))
        stats = read_layer_csv(in_csv[0])

        # Check if we should only evaluate specific designs
        if args.designs is not None:
            stats = [s for i, s in enumerate(stats) if i in args.designs]

        # Create plots for all combinations of metric, scale and plot type
        # Area is not plotted as it is constant for all layers
        #statsdict = stats_to_dict(stats)
        for type in ["line", "bar"]:
            for scale in ["linear", "log"]:
                for metric in SUPPORTED_METRICS:
                    if metric != "area":
                        plotMetricPerConfigPerLayer(stats, outdir_figs, metric, type=type, scale=scale, prefix=args.system[0]+"_"+scale+"_")

    # Generate the plots to evaluate the partitioning
    elif args.plottype == "partitioning":
        # Format of the pareto vector:
        #[index, schedule, 
        #        [[design_id, num_pp+1], num_real_pp, [l_pp, num_pp+1], [e_pp, num_pp+1], [l_pp_link, num_pp], [e_pp_link, num_pp]], 
        #               [[platform_pp, num_pp], pp_layer], 
        #                       [X, n_obj]]

        fname_results_nondom = glob.glob(os.path.join(ROOT_DIR, args.indir, "*result_nondom.csv"))
        fname_results_all = glob.glob(os.path.join(ROOT_DIR, args.indir, "*result_all.csv"))

        results_nondom = pd.read_csv(fname_results_nondom[0], header=None)
        paretos = results_nondom.to_numpy()

        color_pps = COLOR_SEQUENCE[0:1]

        idx = paretos[:, 0]
        schedule = paretos[:, 1]
        constraints = paretos[:, 2:n_constr+2]
        pp_infos = paretos[:, n_constr+2:n_constr+2+n_var]
        objectives = paretos[:, n_constr+2+n_var:-1]

        # Read layers of the network
        schedules_csv = glob.glob(os.path.join(ROOT_DIR, args.indir, "*schedules.csv"))[0]
        df = pd.read_csv(schedules_csv, header=None, index_col=0)
        schedules = df.values.tolist()[0]

        pps = pp_infos[:, 0]
        pp_layers = []
        pps_index = [schedules.index(pp) for pp in pps]

        # Get  results per platform
        design_ids = constraints[:, 0:2]
        in_csv_sensor = glob.glob(os.path.join(ROOT_DIR, args.indir, "*"+args.system[0]+"_tl_layers.csv"))
        in_csv_hpc = glob.glob(os.path.join(ROOT_DIR, args.indir, "*"+args.system[0]+"_tl_layers.csv"))
        sensor_platform_stats = read_layer_csv(in_csv_sensor[0])
        hpc_platform_stats = read_layer_csv(in_csv_hpc[0])

        sensor_platform_stats = [sensor_platform_stats[int(design_id[0])] for design_id in design_ids]
        hpc_platform_stats = [sensor_platform_stats[int(design_id[0])] for design_id in design_ids]

        # Iterate over layers to get metrics by platform
        # Layout:
        #          |             l0           | l1  |  l2 | ...
        # pareto0  | {energy, latency, area}  | ... | ... | ... 
        # pareto1  | {energy, latency, area}  | ... | ..  | ...
        pareto_sensor_metrics = []
        pareto_hpc_metrics = []
        for sensor, hpc, pp in zip(sensor_platform_stats, hpc_platform_stats, pps):
            after_pp = False
            sensor_platform_metrics = []
            hpc_platform_metrics = []
            for layer in schedules:
                if not after_pp:
                    if metrics := sensor["layers"].get(layer):
                        sensor_platform_metrics.append(metrics)
                else:
                    if metrics := hpc["layers"].get(layer):
                        hpc_platform_metrics.append(metrics)

                if layer == pp:
                    after_pp = True
            
            pareto_sensor_metrics.append(sensor_platform_metrics)
            pareto_hpc_metrics.append(hpc_platform_metrics)
        
    elif args.plottype == "compare_base":
        fname_results_nondom = glob.glob(os.path.join(ROOT_DIR, args.indir, "*result_nondom.csv"))
        fname_results_all = glob.glob(os.path.join(ROOT_DIR, args.indir, "*result_all.csv"))

        fname_results_base_nondom = glob.glob(os.path.join(ROOT_DIR, args.basedir, "*result_nondom.csv"))
        fname_results_base_all = glob.glob(os.path.join(ROOT_DIR, args.basedir, "*result_all.csv"))

        results_all_dse = pd.read_csv(fname_results_all[0], header=None).to_numpy()
        results_dom_dse = np.array([res for res in results_all_dse if res[-1]=="dom"])

        results_all_base = pd.read_csv(fname_results_base_all[0], header=None).to_numpy()
        results_dom_base = np.array([res for res in results_all_base if res[-1]=="dom"])

        results_pareto_dse = pd.read_csv(fname_results_nondom[0], header=None).to_numpy()
        results_pareto_base = pd.read_csv(fname_results_base_nondom[0], header=None).to_numpy()

        dse_dom_objectives = results_dom_dse[:, n_constr+2+n_var:-1]
        base_dom_objectives=results_dom_base[:, n_constr+2+n_var:-1]

        base_dom_latency, base_dom_energy, base_dom_throughput, base_dom_area, base_dom_link_latency, base_dom_link_energy = get_individual_metrics(base_dom_objectives)
        dse_dom_latency, dse_dom_energy, dse_dom_throughput, dse_dom_area, dse_dom_link_latency, dse_dom_link_energy = get_individual_metrics(dse_dom_objectives)

        # Extract objective values form the gathered results
        # layout of the objective vector:
        #   [latency, energy, throughput, area, link_latency, link_energy]
        dse_idx =           results_pareto_dse[:, 0]
        dse_schedule =      results_pareto_dse[:, 1]
        dse_constraints =   results_pareto_dse[:, 2:n_constr+2]
        dse_pp_infos =      results_pareto_dse[:, n_constr+2:n_constr+2+n_var]
        dse_objectives =    results_pareto_dse[:, n_constr+2+n_var:-1]

        base_idx =           results_pareto_base[:, 0]
        base_schedule =      results_pareto_base[:, 1]
        base_constraints =   results_pareto_base[:, 2:n_constr+2]
        base_pp_infos =      results_pareto_base[:, n_constr+2:n_constr+2+n_var]
        base_objectives =    results_pareto_base[:, n_constr+2+n_var:-1]

        base_latency, base_energy, base_throughput, base_area, base_link_latency, base_link_energy = get_individual_metrics(base_objectives)
        dse_latency, dse_energy, dse_throughput, dse_area, dse_link_latency, dse_link_energy = get_individual_metrics(dse_objectives)

        # Latency, energy and area are the metrics we need to evaluate
         
        # Do the actual plotting
        fig = plt.figure(dpi=1200)
        ax1 = fig.add_subplot(111)
        #plt.gca().set_prop_cycle(color=["lightgrey", "lightgrey", COLOR_SEQUENCE[0], COLOR_SEQUENCE[1]] )
        for ptype in plot_combinations:
            ax1.scatter( eval(f"dse_dom_{ptype[0]}"), eval(f"dse_dom_{ptype[1]}"), marker="o", color="lightgrey")
            ax1.scatter( eval(f"base_dom_{ptype[0]}"), eval(f"base_dom_{ptype[1]}"), marker="o", color="lightgrey", label="Dominated" )
            ax1.scatter( eval(f"base_{ptype[0]}"), eval(f"base_{ptype[1]}"), label="Base", marker="+", color=COLOR_SEQUENCE[0])
            ax1.scatter( eval(f"dse_{ptype[0]}"), eval(f"dse_{ptype[1]}"), label="DSE", marker="o", color=COLOR_SEQUENCE[1])
            ax1.set_xlabel(f"{objective_to_str(ptype[0])} [{objective_to_unit(ptype[0])}]")
            ax1.set_ylabel(f"{objective_to_str(ptype[1])} [{objective_to_unit(ptype[1])}]")
            ax1.legend()
            figname = f"{objective_to_str(ptype[0])}_over_{objective_to_str(ptype[1])}.pdf"
            fig.savefig(os.path.join(outdir_figs, figname))
            ax1.clear()
        plt.close()

        # Plat additional metrics
        fig = plt.figure(dpi=1200)
        ax1 = fig.add_subplot(111)

        base_dom_edap = base_dom_area * base_dom_latency * base_dom_energy
        dse_dom_edap = dse_dom_area * dse_dom_latency * dse_dom_energy
        base_edap = base_area * base_latency * base_energy
        dse_edap = dse_area * dse_latency * dse_energy
        
        ax1.scatter(base_dom_edap ,base_dom_edap, marker="o", color="lightgrey",  )
        ax1.scatter(dse_dom_edap ,dse_dom_edap, marker="o", color="lightgrey", label="Dominated" )
        ax1.scatter(dse_edap ,dse_edap, marker="o", color=COLOR_SEQUENCE[0], label="DSE" )
        ax1.scatter(base_edap ,base_edap, marker="+", color=COLOR_SEQUENCE[1], label="Base" )
        ax1.legend()
        fig.savefig(os.path.join(outdir_figs, "edap.pdf"))

    elif args.plottype == "compare_multiple":

        dse_dom_latencies = []
        dse_dom_energies = []
        dse_dom_throughputs = []
        dse_dom_areas = []
        dse_dom_link_latencies = []
        dse_dom_link_energies = []
        dse_dom_edaps = []
        dse_dom_edps = []
        dse_dom_eaps = []

        dse_latencies = []
        dse_energies = []
        dse_throughputs = []
        dse_areas = []
        dse_link_latencies = []
        dse_link_energies = []
        dse_edaps = []
        dse_edps = []
        dse_eaps = []

        def append_results(latencies, energies, throughputs, areas, link_latencies, link_energies,
                           latency, energy, throughput, area, link_latency, link_energy):
            latencies.append(latency)
            energies.append(energy)
            throughputs.append(throughput)
            areas.append(area)
            link_latencies.append(link_latency)
            link_energies.append(link_energy)

        num_odirs = len(args.otherdirs)

        # For all specified run directories, gather data
        for odir in args.otherdirs:
            fname_results_nondom = glob.glob(os.path.join(ROOT_DIR, odir, "*result_nondom.csv"))
            fname_results_all = glob.glob(os.path.join(ROOT_DIR, odir, "*result_all.csv"))
            results_pareto_dse = pd.read_csv(fname_results_nondom[0], header=None).to_numpy()

            results_all_dse = pd.read_csv(fname_results_all[0], header=None).to_numpy()
            results_dom_dse = np.array([res for res in results_all_dse if res[-1]=="dom"])

            dse_dom_objectives = results_dom_dse[:, n_constr+2+n_var:-1]
            dse_dom_latency, dse_dom_energy, dse_dom_throughput, dse_dom_area, dse_dom_link_latency, dse_dom_link_energy = get_individual_metrics(dse_dom_objectives)

            append_results(dse_dom_latencies, dse_dom_energies, dse_dom_throughputs, dse_dom_areas, dse_dom_link_latencies, dse_dom_link_energies, *get_individual_metrics(dse_dom_objectives))
            dse_dom_edaps.append(dse_dom_area * dse_dom_latency * dse_dom_energy)
            dse_dom_edps.append(dse_dom_energy * dse_dom_latency)
            dse_dom_eaps.append(dse_dom_energy * dse_dom_area)

            # Extract objective values form the gathered results
            # layout of the objective vector:
            #   [latency, energy, throughput, area, link_latency, link_energy]
            dse_idx =           results_pareto_dse[:, 0]
            dse_schedule =      results_pareto_dse[:, 1]
            dse_constraints =   results_pareto_dse[:, 2:n_constr+2]
            dse_pp_infos =      results_pareto_dse[:, n_constr+2:n_constr+2+n_var]
            dse_objectives =    results_pareto_dse[:, n_constr+2+n_var:-1]
            dse_latency, dse_energy, dse_throughput, dse_area, dse_link_latency, dse_link_energy = get_individual_metrics(dse_objectives)
            append_results(dse_latencies, dse_energies, dse_throughputs, dse_areas, dse_link_latencies, dse_link_energies, *get_individual_metrics(dse_objectives))
            dse_edaps.append(dse_area * dse_latency * dse_energy)
            dse_edps.append(dse_energy * dse_latency)
            dse_eaps.append(dse_energy * dse_area)

        # Collect results from a baseline, i.e. without DSE enabled
        fname_results_base_nondom = glob.glob(os.path.join(ROOT_DIR, args.basedir, "*result_nondom.csv"))
        fname_results_base_all = glob.glob(os.path.join(ROOT_DIR, args.basedir, "*result_all.csv"))

        results_all_base = pd.read_csv(fname_results_base_all[0], header=None).to_numpy()
        results_dom_base = np.array([res for res in results_all_base if res[-1]=="dom"])

        base_dom_objectives = results_dom_base[:, n_constr+2+n_var:-1]
        base_dom_latency, base_dom_energy, base_dom_throughput, base_dom_area, base_dom_link_latency, base_dom_link_energy = get_individual_metrics(base_dom_objectives)

        results_pareto_base = pd.read_csv(fname_results_base_nondom[0], header=None)
        results_pareto_base = results_pareto_base.to_numpy()
        base_objectives =    results_pareto_base[:, n_constr+2+n_var:-1]
        base_latency, base_energy, base_throughput, base_area, base_link_latency, base_link_energy = get_individual_metrics(base_objectives)

        base_edap = base_energy * base_latency * base_area
        base_edp = base_energy * base_latency
        base_eap = base_energy * base_area

        def get_list_name(objective: str) -> str:
            if objective == "latency":
                return "latencies"
            elif objective == "energy":
                return "energies"
            elif objective == "throughput":
                return "throughputs"
            elif objective == "area":
                return "areas"
            elif objective == "link_latency":
                return "link_latencies"
            elif objective == "link_energy":
                return "link_energies"
            
            raise RuntimeError("Invalid objective specified")

        # Do the actual plotting
        fig = plt.figure(dpi=1200)
        ax1 = fig.add_subplot(111)
        marker_size= 10**2

        for ptype in plot_combinations:

            ax1.scatter( eval(f"base_dom_{ptype[0]}"), eval(f"base_dom_{ptype[1]}"), marker=".", color="lightgrey", label="Dominated", s=marker_size )

            # iterate over other results that were specified
            # First we print only the dominated results so they are in the background
            for ores in range(num_odirs):
                list_names = [get_list_name(ptype[0]), get_list_name(ptype[1])]
                objective_dom_list0, objective_dom_list1 = eval(f"dse_dom_{list_names[0]}"), eval(f"dse_dom_{list_names[1]}")
                ax1.scatter( objective_dom_list0[ores], objective_dom_list1[ores],  marker=".", color="lightgrey", s=marker_size)

            # iterate over other results that were specified
            plt.gca().set_prop_cycle(color=COLOR_SEQUENCE[1:]) #, marker=MARKER_SEQUENCE[1:] 
            for ores in range(num_odirs):
                list_names = [get_list_name(ptype[0]), get_list_name(ptype[1])]
                objective_list0, objective_list1 = eval(f"dse_{list_names[0]}"), eval(f"dse_{list_names[1]}")
                ax1.scatter( objective_list0[ores], objective_list1[ores], label=args.labels[ores], marker="o", s=marker_size) #MARKER_SEQUENCE[1+ores]

            # Plot base results
            ax1.scatter( eval(f"base_{ptype[0]}"), eval(f"base_{ptype[1]}"), label="Base", marker=MARKER_SEQUENCE[0], color=COLOR_SEQUENCE[0], s=marker_size)
            
            # General settings
            ax1.set_xlabel(f"{objective_to_str(ptype[0])} [{objective_to_unit(ptype[0])}]", fontsize=18)
            ax1.set_ylabel(f"{objective_to_str(ptype[1])} [{objective_to_unit(ptype[1])}]", fontsize=18)
            ax1.legend()
            figname = f"{objective_to_str(ptype[1])}_over_{objective_to_str(ptype[0])}.pdf"
            fig.savefig(os.path.join(outdir_figs, figname))
            ax1.clear()
        plt.close()
            
        # plot metrics such as EDAP
        #fig = plt.figure(dpi=1200)
        #ax1 = fig.add_subplot(111)

        #dse_dom_all_edaps = np.concatenate(dse_dom_edaps)
        #ax1.scatter(dse_dom_all_edaps, dse_dom_all_edaps, label="Dominated", color="lightgrey", marker=".")

        #plt.gca().set_prop_cycle(color=COLOR_SEQUENCE[1:])
        #for ores in range(num_odirs):
        #    #ax1.scatter(base_dom_edap ,base_dom_edaps, marker="o", color="lightgrey",  )
        #    #ax1.scatter(dse_dom_edaps[ores] ,dse_dom_edaps[ores], marker="o", color="lightgrey", label=args.labels[ores] )
        #    ax1.scatter(dse_edaps[ores], dse_edaps[ores], label=args.labels[ores], marker=MARKER_SEQUENCE[1+ores])

        #ax1.scatter(base_edap, base_edap, marker=MARKER_SEQUENCE[0], color=COLOR_SEQUENCE[0], label="Base" )
        #ax1.legend()
        #fig.savefig(os.path.join(outdir_figs, "edap.pdf"))
        #plt.close()

    elif args.plottype == "print_opt":
        # This requires a base and a dse directory
        fname_results_all = glob.glob(os.path.join(ROOT_DIR, args.indir, "*result_all.csv"))
        fname_results_base_all = glob.glob(os.path.join(ROOT_DIR, args.basedir, "*result_all.csv"))

        results_all_dse = pd.read_csv(fname_results_all[0], header=None).to_numpy()
        results_all_base = pd.read_csv(fname_results_base_all[0], header=None).to_numpy()

        base_all_objectives = results_all_base[:, n_constr+2+n_var:-1]
        dse_all_objectives = results_all_dse[:, n_constr+2+n_var:-1]

        base_latency, base_energy, base_throughput, base_area, base_link_latency, base_link_energy = get_individual_metrics(base_all_objectives)
        dse_latency, dse_energy, dse_throughput, dse_area, dse_link_latency, dse_link_energy = get_individual_metrics(dse_all_objectives)

        # Results for baseline
        base_idx_opt_latency = np.argmin(base_latency)
        base_opt_latency_params = [base_latency[base_idx_opt_latency], base_energy[base_idx_opt_latency], base_area[base_idx_opt_latency], base_throughput[base_idx_opt_latency]]

        base_idx_opt_energy = np.argmin(base_energy)
        base_opt_energy_params = [base_latency[base_idx_opt_energy], base_energy[base_idx_opt_energy], base_area[base_idx_opt_energy], base_throughput[base_idx_opt_energy]]

        # Results for DSE
        dse_idx_opt_latency = np.argmin(dse_latency)
        dse_opt_latency_params = [dse_latency[dse_idx_opt_latency], dse_energy[dse_idx_opt_latency], dse_area[dse_idx_opt_latency], dse_throughput[dse_idx_opt_latency]]

        dse_idx_opt_energy = np.argmin(dse_energy)
        dse_opt_energy_params = [dse_latency[dse_idx_opt_energy], dse_energy[dse_idx_opt_energy], dse_area[dse_idx_opt_energy], dse_throughput[dse_idx_opt_energy]]

        def export_tex(energy_opt_base: list, latency_opt_base: list, energy_opt_dse: list, latency_opt_dse: list):
            flist_energy_base = ["%.2f" % elem for elem in energy_opt_base]
            fl_energy_base = "   &   ".join(flist_energy_base)

            flist_latency_base = ["%.2f" % elem for elem in latency_opt_base]
            fl_latency_base = "   &   ".join(flist_latency_base)

            fl_energy_dse  = ["%.2f" % elem for elem in energy_opt_dse]
            fl_energy_dse =  "   &   ".join(fl_energy_dse)

            fl_latency_dse = ["%.2f" % elem for elem in latency_opt_dse]
            fl_latency_dse = "   &   ".join(fl_latency_dse)

            network = str(args.network) + " Base"
            line = f"\multirow{{2}}{{*}}{{{network}}}" + "  &  " + "  Latency opt.  &  " + fl_latency_base + "      \\\\  \n"
            line += "                                  " +  "  &  " + "  Energy opt.  &  " + fl_energy_base + "      \\\\  \n"

            network = str(args.network) + " DSE"
            line += f"\multirow{{2}}{{*}}{{{network}}}" + "  &  " + "  Latency opt.  &  " + fl_latency_dse + "      \\\\  \n"
            line += "                                  " +  "  &  " + "  Energy opt.  &  " + fl_energy_dse + "      \\\\  \n"
            print(line)

        def pprint_list(l: list):
            format_list = ["%.2f" % elem for elem in l]
            print(format_list)

        export_tex(base_opt_energy_params, base_opt_latency_params, dse_opt_energy_params, dse_opt_latency_params)
        #print("Base Opt Latency")
        #pprint_list(base_opt_latency_params)
        #print("Base Opt Energy")
        #pprint_list(base_opt_energy_params)

        #print("DSE Opt Latency")
        #pprint_list(dse_opt_latency_params)
        #print("DSE Opt Energy")
        #pprint_list(dse_opt_energy_params)


