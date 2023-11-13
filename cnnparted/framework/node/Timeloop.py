import os
import sys
import subprocess
import libconf
import yaml
import glob

from tools.timeloop.scripts.parse_timeloop_output import parse_timeloop_stats

from framework.constants import ROOT_DIR


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

        self.stats = {}

    def run(self, layers : dict, progress : bool = False):
        for layer in layers:
            layer_name = layer.get("name")
            output = self._run_single(layer)

            self.stats[layer_name] = {}
            self.stats[layer_name]["latency"] = output["latency_ms"]
            self.stats[layer_name]["energy"] = output["energy_mJ"]

            if progress:
                layer_i = layers.index(layer) + 1
                print("Finished", layer_i, "/", len(layers), self.prob_name, "layers")

    def _run_single(self,
            layer : dict,
            logfile : str = 'timeloop.log'
        ) -> dict:
        if os.path.isfile(os.path.join(self.configs_dir, 'archs', (self.accname + '.cfg'))):
            self.type_cfg = '.cfg'

        runname = layer.get('name') + self.accname
        dirname = os.path.join(ROOT_DIR, self.runroot, runname)
        subprocess.check_call(['mkdir', '-p', dirname])
        os.chdir(dirname)

        prob_name = self.prob_name
        map_fname = os.path.join(self.configs_dir, 'mapper', ('template'+self.type_cfg))
        nmap_fname = os.path.join(dirname, ('mapper'+self.type_cfg))
        prob_fname = os.path.join(self.configs_dir, 'probs', prob_name+self.type_cfg)
        nprob_fname = os.path.join(dirname, prob_name+self.type_cfg)
        configfile_path  = os.path.join(dirname, (self.runroot+self.type_cfg))
        logfile_path = os.path.join(dirname, logfile)

        if self.mapper_cfg:
            self._rewrite_mapper_cfg(map_fname, nmap_fname)
        else:
            subprocess.check_call(['cp', map_fname, nmap_fname])

        self._rewrite_workload_bounds(prob_fname, nprob_fname, layer)
        self._exportConfig(self.configs_dir, nprob_fname, nmap_fname, configfile_path)

        with open(logfile_path, "w") as outfile:
            status = subprocess.call([self.exec_path, configfile_path], stdout = outfile, stderr = outfile)
            if status != 0:
                subprocess.check_call(['cat', logfile_path])
                print('Did you remember to build timeloop and set up your environment properly?')
                sys.exit(1)

        os.chdir(ROOT_DIR)
        return self._parse_stats(dirname)


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

        conv_params = layer.get('conv_params')
        q = conv_params.get('q')
        p = conv_params.get('p')
        c = conv_params.get('c')  # input channels
        n = conv_params.get('n')  # input batch_size
        m = conv_params.get('m')
        s = conv_params.get('s')
        r = conv_params.get('r')
        wpad = conv_params.get('wpad')
        hpad = conv_params.get('hpad')
        wstride = conv_params.get('wstride')
        hstride = conv_params.get('hstride')


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
        arch_fnames = os.path.join(dir, 'archs', (self.accname + '*'))
        constraint_fname = os.path.join(dir, 'constraints', (self.accname + '*'))

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

    def _parse_stats(self, filename : str) -> dict:
        output = parse_timeloop_stats(filename)

        output["energy_mJ"] = output["energy_pJ"] / 1e9
        output["latency_ms"] = output["cycles"] / self.freq * 1e3

        return output
