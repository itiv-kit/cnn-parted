import pickle
import importlib
import os
import sys
import subprocess
import libconf
import yaml
import glob
import re
import tqdm
import shutil

from tools.timeloop.scripts.parse_timeloop_output import parse_timeloop_stats

from framework.constants import ROOT_DIR
from framework.helpers.Visualizer import plotMetricPerConfigPerLayer
from framework.helpers.DesignMetrics import calc_metric, SUPPORTED_METRICS
from framework.dse.ArchitectureMutator import ArchitectureMutator

class Timeloop:
    # Output file names.
    out_prefix = "timeloop-mapper."
    exec_path = os.path.join(ROOT_DIR, 'tools', 'timeloop', 'build', 'timeloop-mapper')
    configs_dir = os.path.join(ROOT_DIR, 'configs', 'tl_configs')

    def __init__ (self, tl_config : dict) -> None:
        log_file_name = self.out_prefix + "log"
        stats_file_name = self.out_prefix + "stats.txt"
        xml_file_name = self.out_prefix + "map+stats.xml"
        map_txt_file_name = self.out_prefix + "map.txt"
        map_cfg_file_name = self.out_prefix + "map.cfg"
        map_cpp_file_name = self.out_prefix + "map.cpp"
        accelergy_log_name = self.out_prefix + "accelergy.log"
        accelergy_art_sum_name = self.out_prefix + "ART_summary.yaml"
        accelergy_art_name = self.out_prefix + "ART.yaml"
        accelergy_ert_sum_name = self.out_prefix + "ERT_summary.yaml"
        accelergy_ert_name = self.out_prefix + "ERT.yaml"
        flatt_arch_name = self.out_prefix + "flattened_architecture.yaml"

        self.output_file_names = [  log_file_name,
                                    stats_file_name,
                                    xml_file_name,
                                    map_txt_file_name,
                                    map_cfg_file_name,
                                    map_cpp_file_name,
                                    accelergy_log_name,
                                    accelergy_art_sum_name,
                                    accelergy_art_name,
                                    accelergy_ert_sum_name,
                                    accelergy_ert_name,
                                    flatt_arch_name ]

        self.accname = tl_config['accelerator']
        self.prob_name = tl_config['layer']
        self.freq = tl_config['frequency']
        self.mapper_cfg = {} if not tl_config.get('mapper') else tl_config['mapper']
        self.type_cfg = '.yaml'
        self.runroot = tl_config['run_root']
        self.dse_config = tl_config.get('dse', None)
        self.tl_cfg = tl_config
        self.stats = {} #[]

        self.mutator: ArchitectureMutator = None
        if self.dse_config:
            mutator_cfg = self.dse_config
            mutator_cfg["tl_in_configs_dir"] = self.configs_dir

            mutator_name = self.dse_config["mutator"]
            package = importlib.import_module(f"framework.dse.{mutator_name}")
            mutator_cls = getattr(package, self.dse_config["mutator"])
            self.mutator = mutator_cls(self.dse_config)

    def run(self, layers : dict, progress : bool = False):
        if self.mutator is not None:
            stats = {}
            
            print(f"There are a total of {len(self.mutator.design_space)} designs to be evaluated!")
            
            for i, design in enumerate(self.mutator.design_space):
                if os.path.exists(os.path.join(self.runroot, "design"+str(i))):
                    shutil.rmtree(os.path.join(self.runroot, "design"+str(i)))
                self._run_design(layers, progress, stats, i, design)
            
            # Plot all metrics in all combinations of line/bar, scale/log
            for m in SUPPORTED_METRICS:
                if m != "area":
                    self._plot_all_of_metric(stats, m)

        else:
            stats = {"design_0": {}}
            stats["design_0"]["layers"] = {}
            #stats["0"]["design_params"] = ArchitectureConfig.from_yaml()
            for layer in tqdm.tqdm(layers, self.accname, disable=(not progress)):
                layer_name = layer.get("name")
                output = self._run_single(self.runroot, layer, tl_files_path=None)

                stats["design_0"]["layers"][layer_name] = {}
                stats["design_0"]["layers"][layer_name]["latency"] = output["latency_ms"]
                stats["design_0"]["layers"][layer_name]["energy"] = output["energy_mJ"]
                stats["design_0"]["layers"][layer_name]["area"] = output["area_mm2"]

        # Gather results
        self.stats = {tag: results for tag, results in stats.items()}
        with open(os.path.join(self.runroot, f"results_{self.accname}.pkl"), "wb") as f:
            pickle.dump(self.stats, f)

    def _run_design(self, layers: dict, progress: bool, stats: dict, i: int, design):
        design_runroot = os.path.join(self.runroot, "design"+str(i))
        tl_design_dir = os.path.join(design_runroot, "tl_config")
        tl_design_dir_arch = os.path.join(design_runroot, "tl_config", "archs")
        tl_design_dir_constraints = os.path.join(design_runroot, "tl_config", "constraints")
        os.makedirs(tl_design_dir_arch)
        os.makedirs(tl_design_dir_constraints)

        self.mutator.run_from_config(design, outdir=tl_design_dir)
        stats[f"design_{i}"] = {}
        stats[f"design_{i}"]["layers"] = {}
        stats[f"design_{i}"]["arch_config"] = design.get_config()
        with open(os.path.join(design_runroot, "arch_config.yaml"), "w") as f:
            y = yaml.safe_dump(design.get_config(), sort_keys=False)
            f.write(y)

        for layer in tqdm.tqdm(layers, self.accname, disable=(not progress)):
            layer_name = layer.get("name")
            output = self._run_single(design_runroot, layer, tl_files_path=tl_design_dir)

            stats[f"design_{i}"]["layers"][layer_name] = {}
            stats[f"design_{i}"]["layers"][layer_name]["latency"] = output["latency_ms"]
            stats[f"design_{i}"]["layers"][layer_name]["energy"] = output["energy_mJ"]
            stats[f"design_{i}"]["layers"][layer_name]["area"] = output["area_mm2"]

    def _run_single(self,
            runroot: str,
            layer : dict,
            logfile : str = 'timeloop.log',
            tl_files_path: str = None
        ) -> dict:
        if os.path.isfile(os.path.join(self.configs_dir, 'archs', (self.accname + '.cfg'))):
            self.type_cfg = '.cfg'

        runname = layer.get('name')[1:]
        dirname = os.path.join(ROOT_DIR, runroot, runname)
        subprocess.check_call(['mkdir', '-p', dirname])
        #os.chdir(dirname)

        prob_name = self.prob_name
        map_fname = os.path.join(self.configs_dir, 'mapper', ('template'+self.type_cfg))
        nmap_fname = os.path.join(dirname, ('mapper'+self.type_cfg))
        prob_fname = os.path.join(self.configs_dir, 'probs', prob_name+self.type_cfg)
        nprob_fname = os.path.join(dirname, prob_name+self.type_cfg)
        configfile_path  = os.path.join(dirname, (self.accname+self.type_cfg))
        logfile_path = os.path.join(dirname, logfile)

        if tl_files_path is None:
            tl_files_path = self.configs_dir
        else:
            tl_files_path = os.path.join(ROOT_DIR, tl_files_path)

        if self.mapper_cfg:
            self._rewrite_mapper_cfg(map_fname, nmap_fname)
        else:
            subprocess.check_call(['cp', map_fname, nmap_fname])

        self._rewrite_workload_bounds(prob_fname, nprob_fname, layer)
        self._exportConfig(tl_files_path, nprob_fname, nmap_fname, configfile_path)

        with open(logfile_path, "w") as outfile:
            status = subprocess.call([self.exec_path, configfile_path], stdout = outfile, stderr = outfile, cwd=dirname)
            if status != 0:
                subprocess.check_call(['cat', logfile_path])
                print('Did you remember to build timeloop and set up your environment properly?')
                sys.exit(1)

        #os.chdir(ROOT_DIR)
        timeloop_stats = self._parse_stats(dirname)

        # Workaround for batched matmul operations. We assume that the accelerator would 
        # just perform each batch separately. Since they then all have the same
        # shape the mapping is also the same and we can just multiply the results of one batch.
        # For acclerators where this is not the case, a new mapping configuration must be designed
        if gemm_params := layer.get('gemm_params'):
            if batch_size := gemm_params.get('b'):
                timeloop_stats["energy_mJ"] = timeloop_stats["energy_mJ"] * batch_size
                timeloop_stats["latency_ms"] = timeloop_stats["latency_ms"] * batch_size
        return timeloop_stats


    def _plot_all_of_metric(self, stats, metric: str):
        plotMetricPerConfigPerLayer(stats, self.tl_cfg["work_dir"], metric,                          prefix=self.accname+"_")
        plotMetricPerConfigPerLayer(stats, self.tl_cfg["work_dir"], metric,             scale="log", prefix=self.accname+"_log_")
        plotMetricPerConfigPerLayer(stats, self.tl_cfg["work_dir"], metric, type="bar",              prefix=self.accname+"_")
        plotMetricPerConfigPerLayer(stats, self.tl_cfg["work_dir"], metric, type="bar", scale="log", prefix=self.accname+"_log_")

    def _rewrite_mapper_cfg(self, src : str, dst : str) -> None:
        with open(src, "r") as f:
            if "cfg" in src:
                config = libconf.load(f)
            elif "yaml" in src:
                config = yaml.load(f, Loader = yaml.SafeLoader)

        for key in self.mapper_cfg.keys():
            config['mapper'][key] = self.mapper_cfg[key]

        with open(dst, "w") as f:
            if "cfg" in src:
                f.write(libconf.dumps(config))
            elif "yaml" in src:
                f.write(yaml.dump(config))

    def _get_timeloop_params(self, config: dict, layer: dict) -> list:
        if layer.get('conv_params'):
            conv_params = layer.get('conv_params')
            q = conv_params.get('q')
            p = conv_params.get('p')
            c = conv_params.get('c')  # input channels
            n = conv_params.get('n')  # input batch_size
            m = conv_params.get('m')
            s = conv_params.get('s')
            r = conv_params.get('r')
            wstride = conv_params.get('wstride')
            hstride = conv_params.get('hstride')
        else:
            gemm_params = layer.get('gemm_params')
            c = gemm_params.get('c')
            n = gemm_params.get('n')
            m = gemm_params.get('m')
            q = 1
            p = 1
            s = 1
            r = 1
            wstride = 1
            hstride = 1


        config['problem']['instance']['R'] = r
        config['problem']['instance']['S'] = s
        config['problem']['instance']['P'] = p
        config['problem']['instance']['Q'] = q
        config['problem']['instance']['C'] = c
        if 'M' in config['problem']['instance'].keys():
            config['problem']['instance']['M'] = m
        else:
            config['problem']['instance']['K'] = m
        config['problem']['instance']['N'] = n
        config['problem']['instance']['Wstride'] = wstride
        config['problem']['instance']['Hstride'] = hstride
        config['problem']['instance']['Wdilation'] = 1
        config['problem']['instance']['Hdilation'] = 1

    def _rewrite_workload_bounds(self, src : str, dst : str, layer : dict) -> None:
        with open(src, "r") as f:
            if "cfg" in src:
                config = libconf.load(f)
            elif "yaml" in src:
                config = yaml.load(f, Loader = yaml.SafeLoader)

        self._get_timeloop_params(config, layer)

        with open(dst, "w") as f:
            if "cfg" in src:
                f.write(libconf.dumps(config))
            elif "yaml" in src:
                f.write(yaml.dump(config))

    def _load_accelerator_files(self, dir : str) -> list:
        arch_fnames = os.path.join(dir, 'archs', (self.accname + '.yaml'))
        constraint_fname = os.path.join(dir, 'constraints', (self.accname + '_' + '*'))

        input_fnames = [arch_fnames, constraint_fname]
        input_files = []
        for fname in input_fnames:
            input_files += glob.glob(fname, recursive = True)

        if not input_files:
            raise Exception("Accelerator " + self.accname + " not available.")

        return input_files

    def _exportConfig(self, config_dir : str, prob_fname : str, map_fname : str, confname : str) -> None:
        config_files = self._load_accelerator_files(config_dir)
        config_files += [prob_fname, map_fname]

        with open(confname, 'w') as cf:
            for fname in config_files:
                if not os.path.isfile(fname):
                    continue
                with open(fname, 'r') as f:
                    cf.write(f.read())
                    cf.write('\n')

    def _parse_area_stats(self, filename : str) -> float:
        area = 0.0
        with open(filename, "r") as f:
            area_stats = yaml.load(f, Loader = yaml.SafeLoader)
            for comp in area_stats['ART']['tables']:
                single_area = float(comp['area'])

                elements_regex = re.compile(r'(\d)..(\d\d?\d?)')
                num_elements = 1
                pos = 0
                while 1:
                    res = elements_regex.search(comp['name'], pos=pos)
                    if res is not None:
                        num_elements *= int(res.group(2)) + int(res.group(1)) + 1
                        pos = res.span(2)[-1]
                    else:
                        break

                area += num_elements * single_area

        return area

    def _parse_stats(self, dirname : str) -> dict:
        output = parse_timeloop_stats(dirname)
        area = self._parse_area_stats(os.path.join(dirname, self.output_file_names[8])) # *.ART.yaml

        output["energy_mJ"] = output["energy_pJ"] / 1e9
        output["latency_ms"] = output["cycles"] / self.freq * 1e3
        output["area_mm2"] = area / 1e6

        return output
