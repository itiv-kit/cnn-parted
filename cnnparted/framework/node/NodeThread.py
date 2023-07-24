from framework.ModuleThreadInterface import ModuleThreadInterface
from .GenericNode import GenericNode
from .Timeloop import Timeloop

import csv

class NodeThread(ModuleThreadInterface):
    def _eval(self) -> None:
        if not self.config:
            return

        if self.name == 'edge':
            self.reverse = True
        else:
            self.reverse = False

        if self.config.get('timeloop'):
            self._run_timeloop(self.config['timeloop'])
        else:
            self._run_generic(self.config)


    def _run_generic(self, config : dict) -> None:
        part_points = self.dnn.partition_points
        input_size = self.dnn.input_size

        gn = GenericNode(config)

        layer_name = 'Identity'
        if self.reverse:
            layers = part_points.copy()
        else: # Set stats to zero for Identity
            self.stats[layer_name] = {}
            self.stats[layer_name]['latency'] = 0
            self.stats[layer_name]['latency_iqr'] = 0
            self.stats[layer_name]['energy'] = 0

        # only iterate through filtered list to save time
        for point in self.dnn.partpoints_filtered:
            layer_name = point.get_layer_name(False, True)

            if not self.reverse:
                idx = part_points.index(point) + 1
                layers = part_points[:idx]
            else:
                idx = layers.index(point) + 1
                del layers[:idx]
                input_size = point.output_size

            output = gn.run(layers, input_size)
            self.stats[layer_name] = {}
            self.stats[layer_name]['latency'] = output['latency_ms']
            self.stats[layer_name]['latency_iqr'] = output['latency_iqr']
            self.stats[layer_name]['energy'] = output['energy_mJ']

            if self.show_progress:
                layer_i = self.dnn.partpoints_filtered.index(point) + 1
                print("Finished", layer_i, "/", len(self.dnn.partpoints_filtered), self.name, "models" )


    def _run_timeloop(self, config : dict) -> None:
        overall_latency = 0
        overall_energy = 0

        conv_layers = self.dnn.get_conv2d_layers()

        if self.reverse:
            conv_layers = conv_layers[::-1]

        runroot = 'run' + self.name
        config['run_root'] = runroot

        tl = Timeloop(config)

        for layer in conv_layers:
            output = tl.run(layer)
            overall_latency += output['latency_ms']
            overall_energy += output['energy_mJ']

            layer_name = layer.get_layer_name(False, True)
            partpoint_name = self.dnn.search_partition_point(layer).get_layer_name(False, True)

            if not partpoint_name in self.stats.keys():
                self.stats[partpoint_name] = {}

            self.stats[partpoint_name]['latency'] = overall_latency
            self.stats[partpoint_name]['latency_iqr'] = 0
            self.stats[partpoint_name]['energy'] = overall_energy

            # save single layer stats
            self.stats[partpoint_name][layer_name] = {}
            self.stats[partpoint_name][layer_name]['latency'] = output['latency_ms']
            self.stats[partpoint_name][layer_name]['latency_iqr'] = 0
            self.stats[partpoint_name][layer_name]['energy'] = output['energy_mJ']

            if self.show_progress:
                layer_i = conv_layers.index(layer) + 1
                print("Finished", layer_i, "/", len(conv_layers), self.name, "layers" )

        # Fill up dict with partitioning points not containing CONVs
        prev_latency = 0
        prev_energy = 0
        if self.reverse:
            plist = self.dnn.partition_points[::-1]
        else:
            plist = self.dnn.partition_points

        for point in plist:
            l_name = point.get_layer_name(False, True)
            if l_name in self.stats.keys():
                prev_latency = self.stats[l_name]['latency']
                prev_energy = self.stats[l_name]['energy']
            else:
                self.stats[l_name] = {}
                self.stats[l_name]['latency'] = prev_latency
                self.stats[l_name]['latency_iqr'] = 0
                self.stats[l_name]['energy'] = prev_energy

        self._write_timeloop_csv(self.runname + "_" + self.name + "_tl_layers.csv")


    def _write_timeloop_csv(self, filename : str) -> None:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            header = [
                "No.",
                "Partitioning Point",
                "Layer",
                "Latency [ms]",
                "Energy [mJ]"
            ]
            writer.writerow(header)
            row_num = 1
            for pp in self.stats.keys():
                for l in self.stats[pp].keys():
                    if isinstance(self.stats[pp][l], dict):
                        row = [
                            row_num,
                            pp,
                            l,
                            str(self.stats[pp][l]['latency']),
                            str(self.stats[pp][l]['energy'])
                        ]
                        writer.writerow(row)
                        row_num += 1
