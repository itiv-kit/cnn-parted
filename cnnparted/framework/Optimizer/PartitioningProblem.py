from pymoo.core.problem import ElementwiseProblem
from copy import deepcopy
import numpy as np

from ..link.Link import Link


class PartitioningProblem(ElementwiseProblem):
    def __init__(self, nodeStats : dict, schedule : list, fixed_sys : bool, layer_dict : dict, layer_params : dict, link_confs : list):
        self.nodeStats = nodeStats
        self.num_acc = len(nodeStats)
        self.num_pp = self.num_acc - 1
        self.schedule = schedule
        self.fixed_sys = fixed_sys
        self.num_layers = len(schedule)
        self.layer_dict = layer_dict
        self.layer_params = layer_params

        self.links = []
        for link_conf in link_confs:
            self.links.append(Link(link_conf))


        n_var = self.num_pp * 2 + 1 # Number of max. partitioning points + device ID
        n_obj = 3 + self.num_pp + self.num_acc # latency, energy, throughput + bandwidth + memory

        xu_pp = np.empty(self.num_pp)
        xu_pp.fill(self.num_layers + 0.49)
        xu_acc = np.empty(self.num_pp + 1)
        xu_acc.fill(self.num_acc + 0.49)

        xu = np.append(xu_pp, xu_acc)

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=1,
                         xl=0.51,
                         xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        p = []
        for i in x:
            p.append(int(np.round(i)))

        latency = 0.0
        energy = 0.0
        throughput = 0.0
        bandwidth = np.full((self.num_pp), np.inf)
        mem = np.full((self.num_acc), np.inf)

        if not np.array_equal(np.sort(p[:self.num_pp]), p[:self.num_pp]): # keep order of partitioning points
            out["G"] = x[0]
        elif np.unique(p[-self.num_acc:]).size != np.asarray(p[-self.num_acc:]).size:   # only use accelerator once
            out["G"] = x[0]
        elif self.fixed_sys and not np.array_equal(np.sort(p[-self.num_acc:]), p[-self.num_acc:]): # keep order of Accelerators
            out["G"] = x[0]
        else:
            out["G"] = -x[0]

            l_pp = []
            e_pp = []
            th_pp = []
            successors = []
            successors.append(self.schedule[0])

            i = -1
            last_pp = -1
            for i, pp in enumerate(p[0:self.num_pp], self.num_pp):
                mem[i-self.num_pp] = self._eval_partition(p[i], last_pp, pp, l_pp, e_pp, th_pp, successors)

                # evaluate link
                link_l, link_e, bandwidth[i-self.num_pp] = self._get_link_metrics(i-self.num_pp, last_pp, pp, successors)
                l_pp.append(link_l)
                e_pp.append(link_e)
                th_pp.append(self._zero_division(1000.0, link_l)) # FPS - latency in ms

                last_pp = pp

            mem[i+1-self.num_pp] = self._eval_partition(p[i+1], last_pp, self.num_layers, l_pp, e_pp, th_pp, successors)

            latency = sum(l_pp)
            energy = sum(e_pp)
            throughput = min(th_pp) * -1

        out["F"] = [latency, energy, throughput] + list(bandwidth) + list(mem)

    def _eval_partition(self, acc : int, last_pp : int, pp : int, l_pp : list, e_pp : list, th_pp : list, successors : list) -> int:
        acc -= 1
        acc_latency = 0.0
        acc_energy = 0.0
        part_l_params = 0

        ls = {}
        dmem = []
        for j in range(last_pp + 1, pp + 1):
            layer = self.schedule[j-1]
            acc_latency += self._get_layer_latency(acc, layer)
            acc_energy += self._get_layer_energy(acc, layer)
            if layer in self.layer_params.keys():
                part_l_params += self.layer_params[layer]

            while layer in successors: successors.remove(layer)
            layer_successors = self.layer_dict[layer]["successors"]
            successors += layer_successors

            ifms = [] # Input Feature Maps
            afms = [] # Active Feature Maps
            for k in list(ls.keys()):
                v = ls[k]
                if layer in v:
                    ifms.append(np.prod(self.layer_dict[k]["output_size"]))
                    ls[k].remove(layer)
                elif v:
                    afms.append(np.prod(self.layer_dict[k]["output_size"]))
                else:
                    del ls[k]

            ofm = np.prod(self.layer_dict[layer]["output_size"]) # Output Feature Maps
            dmem.append(sum([ofm] + ifms + afms))   # Feature Map Memory Consumption per layer

            ls[layer] = deepcopy(layer_successors)

        l_pp.append(acc_latency)
        e_pp.append(acc_energy)
        th_pp.append(self._zero_division(1000.0, acc_latency)) # FPS - latency in ms
        return part_l_params + max(dmem, default=0) # mem evaluation

    def _get_layer_latency(self, acc : int, layer_name : str) -> float:
        acc = list(self.nodeStats.keys())[acc]
        if self.nodeStats[acc].get(layer_name):
            return float(self.nodeStats[acc][layer_name]['latency'])
        else:
            return 0

    def _get_layer_energy(self, acc : int, layer_name : str) -> float:
        acc = list(self.nodeStats.keys())[acc]
        if self.nodeStats[acc].get(layer_name):
            return float(self.nodeStats[acc][layer_name]['energy'])
        else:
            return 0

    def _zero_division(self, a : float, b : float) -> float:
        return a / b if b else np.inf

    def _get_link_metrics(self, link_idx : int, last_pp : int, pp : int, successors : list) -> (float, float, float):
        if len(self.links) == 0 or last_pp == pp:
            return 0, 0, 0

        layers = []
        for layer in np.unique(successors):
            layers += self.layer_dict[layer]["predecessors"]
        layers = np.unique([layer for layer in layers if layer not in successors])

        data_sizes = [np.prod(self.layer_dict[layer]["output_size"]) for layer in layers]

        if len(self.links) == 1:
            return self.links[0].eval(np.sum(data_sizes))
        else:
            return self.links[link_idx].eval(np.sum(data_sizes))
