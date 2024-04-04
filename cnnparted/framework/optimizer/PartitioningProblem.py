from pymoo.core.problem import ElementwiseProblem
from copy import deepcopy
import numpy as np
from collections import deque, defaultdict

from ..link.Link import Link


class PartitioningProblem(ElementwiseProblem):
    def __init__(self, num_pp : int, nodeStats : dict, schedule : list, q_constr : dict, fixed_sys : bool, acc_once : bool, layer_dict : dict, layer_params : dict, link_confs : list):
        self.nodeStats = nodeStats
        self.num_acc = len(nodeStats)
        self.num_pp = num_pp
        self.schedule = schedule
        self.q_constr = q_constr
        self.fixed_sys = fixed_sys
        self.acc_once = acc_once
        self.num_layers = len(schedule)
        self.layer_dict = layer_dict
        self.layer_params = layer_params

        self.links = []
        for link_conf in link_confs:
            self.links.append(Link(link_conf))

        n_var = self.num_pp * 2 + 1 # Number of max. partitioning points + device IDs
        n_obj = 5 # latency, energy, throughput + link latency + link energy
        n_constr = 1 + (self.num_pp + 1) * 2 + (self.num_pp + 1) * 2 # num_real_pp + latency/energy per partition + latency/energy per link

        xu_pp = np.empty(self.num_pp)
        xu_pp.fill(self.num_layers)
        xu_acc = np.empty(self.num_pp + 1)
        xu_acc.fill(self.num_acc)
        xu = np.append(xu_pp, xu_acc)

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=1, xu=xu)

    def _evaluate(self, x : np.ndarray, out : dict, *args, **kwargs) -> None:
        valid = True
        num_real_pp = 0
        curr_latency = 0.0 # used for throughput calculation

        l_pp = []
        e_pp = []

        latency = energy = throughput = link_latency = link_energy = 0.0
        bandwidth = np.full((self.num_pp + 1), np.inf)
        mem = np.full((self.num_pp + 1), np.inf)

        p : list = x.tolist()
        p.insert(self.num_pp, self.num_layers)
        if not np.array_equal(np.sort(p[:self.num_pp]), p[:self.num_pp]): # keep order of partitioning points
            valid = False
        elif self.acc_once and np.unique(p[-self.num_pp-1:]).size != np.asarray(p[-self.num_pp-1:]).size:   # only use accelerator once
            valid = False
        elif self.fixed_sys and not np.array_equal(np.sort(p[-self.num_pp-1:]), p[-self.num_pp-1:]): # keep order of Accelerators
            valid = False
        else:
            th_pp = []
            part_latency = deque()
            l_pp_link = []
            e_pp_link = []
            successors = [self.schedule[0]]
            i = last_pp = last_acc = -1
            for i, pp in enumerate(p[0:self.num_pp+1], self.num_pp + 1):
                v, mem[i-self.num_pp-1] = self._eval_partition(p[i], last_pp, pp, l_pp, e_pp, successors)
                valid &= v

                # evaluate link
                if last_pp != pp and last_acc != p[i]:
                    if last_acc != -1:
                        link_l, link_e, bandwidth[i-self.num_pp-1] = self._get_link_metrics(i-self.num_pp-1, successors)
                        l_pp_link.append(link_l)
                        e_pp_link.append(link_e)
                        th_pp.append(self._zero_division(1000.0, link_l)) # FPS - latency in ms
                    else:
                        l_pp_link.append(0.0)
                        e_pp_link.append(0.0)
                        bandwidth[i-self.num_pp-1] = 0

                    if last_pp != 1:
                        num_real_pp += 1
                        if last_pp != -1:
                            part_latency.append([last_acc, sum(l_pp[:-1]) - curr_latency])
                            curr_latency = sum(l_pp[:-1])
                else:
                    l_pp_link.append(0.0)
                    e_pp_link.append(0.0)
                    bandwidth[i-self.num_pp-1] = 0

                if pp == self.num_layers:
                    part_latency.append([p[i], sum(l_pp) - curr_latency])

                if last_pp != pp:
                    last_pp = pp
                    if last_pp != 1: # if last_pp not input
                        last_acc = p[i]

            link_latency = sum(l_pp_link)
            link_energy = sum(e_pp_link)
            latency = sum(l_pp) + link_latency
            energy = sum(e_pp) + link_energy
            throughput = self._get_throughput(th_pp, part_latency) * -1

        out["F"] = [latency, energy, throughput, link_latency, link_energy] #+ list(bandwidth) #+ list(mem)

        if valid:
            out["G"] = [-num_real_pp] + [i * (-1) for i in l_pp] + [i * (-1) for i in e_pp] + [i * (-1) for i in l_pp_link] + [i * (-1) for i in e_pp_link]
        else:
            out["G"] = [1] + [i for i in range(self.num_pp+1)] + [i for i in range(self.num_pp+1)] + [i for i in range(self.num_pp+1)] + [i for i in range(self.num_pp+1)]

    def _eval_partition(self, acc : int, last_pp : int, pp : int, l_pp : list, e_pp : list, successors : list) -> tuple[bool, int]:
        valid = True
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
            valid &= self._check_layer_bitwidth(acc, layer)
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
        return valid, part_l_params + max(dmem, default=0) # mem evaluation

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

    def _check_layer_bitwidth(self, acc : int, layer_name : str) -> bool:
        if 'input' in layer_name or 'output' in layer_name:
            return True
        acc = list(self.nodeStats.keys())[acc]
        bit_width = self.nodeStats[acc].get("bits")
        return bit_width >= max([self.q_constr[x] for x in self.q_constr.keys() if layer_name in x], default=0)


    def _zero_division(self, a : float, b : float) -> float:
        return a / b if b else np.inf

    def _get_link_metrics(self, link_idx : int, successors : list) -> tuple[float, float, float]:
        if len(self.links) == 0:
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

    def _get_throughput(self, th_pp : list, part_latency : deque) -> float:
        acc_index = defaultdict(list)
        for i, pair in enumerate(part_latency):
            acc_index[pair[0]].append(i)

        for _, indexes in acc_index.items():
            acc_latency = 0.0
            for i in range(min(indexes), max(indexes)+1):
                acc_latency += part_latency[i][1]

            th_pp.append(self._zero_division(1000.0, acc_latency))

        return min(th_pp)
