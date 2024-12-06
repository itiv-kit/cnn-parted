import math
from pymoo.core.problem import ElementwiseProblem
from copy import deepcopy
import numpy as np
from collections import deque, defaultdict

from framework.link.Link import Link
from framework.helpers.DesignMetrics import calc_metric


class PartitioningProblem(ElementwiseProblem):
    def __init__(self, num_pp : int, nodeStats : dict, schedule : list, q_constr : dict, fixed_sys : bool, acc_once : bool, layer_dict : dict, layer_params : dict, link_confs : list, system_constraints: dict):
        self.nodeStats = nodeStats
        self.num_platforms = len(nodeStats)
        self.num_pp = num_pp
        self.schedule = schedule
        self.q_constr = q_constr
        self.fixed_sys = fixed_sys
        self.acc_once = acc_once
        self.num_layers = len(schedule)
        self.layer_dict = layer_dict
        self.layer_params = layer_params
        self.system_constraints = system_constraints

        self.links = []
        for link_conf in link_confs:
            self.links.append(Link(link_conf))

        n_var = self.num_pp * 2 + 1 # platform IDs + Number of max. partitioning points
        n_obj = 6 # latency, energy, throughput, area + link latency + link energy
        n_constr = self.num_platforms + 1 + (self.num_pp + 1) * 2 + (self.num_pp + 1) * 2 # num_accelerator_platforms + num_real_pp + latency/energy per partition + latency/energy per link

        xu = self.num_platforms * self.num_layers - 1

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=0, xu=xu)

    def _evaluate(self, x : np.ndarray, out : dict, *args, **kwargs) -> None:
        valid = True
        num_real_pp = 0

        l_pp = []
        e_pp = []

        # Init dict
        design_tag = {}
        for key in self.nodeStats.keys():
            design_tag[key] = 'design_0'

        latency = energy = throughput = area = link_latency = link_energy = 0.0
        bandwidth = np.full((self.num_pp + 1), np.inf)
        mem = np.full((self.num_pp + 1), np.inf)

        # [ [partitioning_point, num_pp], [map_partition_to_platform, num_pp+1] ]
        p : list = x.tolist()
        p[0:self.num_pp] = np.divide(p[0:self.num_pp], self.num_platforms)
        p[self.num_pp:] = np.divide(p[self.num_pp:], self.num_layers)
        p = np.floor(p).astype(int)
        # insert a partitioning point after last layer as otherwise evaluation of final partition is not performed
        # e.g. if there are 10 layers and pp=5, the inserted point is needed to trigger
        # evaluation of partition from layer 6-10
        p = np.insert(p, self.num_pp, self.num_layers-1)

        if not np.array_equal(np.sort(p[:self.num_pp]), p[:self.num_pp]): # keep order of partitioning points
            valid = False
        elif self.acc_once and np.unique(p[-self.num_pp-1:]).size != np.asarray(p[-self.num_pp-1:]).size: # only use accelerator once
            valid = False
        elif self.fixed_sys and not np.array_equal(np.sort(p[-self.num_pp-1:]), p[-self.num_pp-1:]): # keep order of Accelerators
            valid = False
        else:
            th_pp = []
            partitions = []
            part_latency = deque()
            l_pp_link = np.full((self.num_pp + 1), 0, dtype=float)
            e_pp_link = np.full((self.num_pp + 1), 0, dtype=float)
            successors = [self.schedule[0]]
            i = last_pp = last_platform = -1
            for i, pp in enumerate(p[0:self.num_pp+1], self.num_pp + 1):
                partitions.append([p[i], last_pp, pp])

                # link evaluation
                if len(partitions) > 1: # do not calculate time to transfer input data
                    if partitions[-2][0] != partitions[-1][0] or last_platform != partitions[-1][0]: #and last_platform != -1): # not the same platform
                        if partitions[-1][1] != partitions[-1][2] and partitions[-1][1] != 0: # not the same layer and not the input
                            num_real_pp += 1 # set number of real partitioning points
                            last_platform = partitions[-1][0]

                            link_l, link_e, bandwidth[i-self.num_pp-1] = self._get_link_metrics(i-self.num_pp-1, last_pp, successors)
                            l_pp_link[i-self.num_pp-1] = link_l
                            e_pp_link[i-self.num_pp-1] = link_e
                            th_pp.append(self._zero_division(1000.0, link_l)) # FPS - latency in ms
                else:
                    last_platform = partitions[-1][0]

                # node evaluation
                v, optimal_design_tag, mem[i-self.num_pp-1] = self._eval_partition(p[i], last_pp, pp, l_pp, e_pp, successors)
                valid &= v
                part_latency.append([p[i], l_pp[-1]]) # required for throughput calculation

                current_platform = list(self.nodeStats.keys())[p[i]]
                design_tag[current_platform] = optimal_design_tag

                # update last pp and acc
                if last_pp != pp:
                    last_pp = pp

            link_latency = sum(l_pp_link)
            link_energy = sum(e_pp_link)
            latency = sum(l_pp) + link_latency
            energy = sum(e_pp) + link_energy
            throughput = self._get_throughput(th_pp, part_latency) * -1
            area = self._get_area(partitions, design_tag)
            design_tag = [self._get_tag_as_int(tag) for tag in design_tag.values()]

        out["F"] = [latency, energy, throughput, area, link_latency, link_energy] #+ list(bandwidth) #+ list(mem)

        valid &= self._check_system_constraints(self.system_constraints, 
                                                energy+link_energy,
                                                latency+link_latency,
                                                throughput,
                                                area)

        if valid:
            out["G"] = [i * (-1) for i in design_tag] + [-num_real_pp] + [i * (-1) for i in l_pp] + [i * (-1) for i in e_pp] + [i * (-1) for i in l_pp_link] + [i * (-1) for i in e_pp_link]
        else:
            out["G"] = [i for i in design_tag] + [1] + [i for i in range(self.num_pp+1)] + [i for i in range(self.num_pp+1)] + [i for i in range(self.num_pp+1)] + [i for i in range(self.num_pp+1)]

    def _eval_partition(self, platform : int, last_pp : int, pp : int, l_pp : list, e_pp : list, successors : list) -> tuple[bool, int]:
        platform = list(self.nodeStats.keys())[platform]

        area_per_design = []
        latency_per_design = []
        energy_per_design = []

        platform_latency_per_design = []
        platform_energy_per_design = []

        design_tags: list[str] = list(self.nodeStats[platform]["eval"].keys())

        for design_tag, _ in self.nodeStats[platform]["eval"].items():
            valid = True
            platform_latency = 0.0
            platform_energy = 0.0
            part_l_params = 0

            energy_per_layer = []
            latency_per_layer = []

            for j in range(last_pp + 1, pp+1):
                layer = self.schedule[j]

                layer_latency = self._get_layer_latency(platform, design_tag, layer)
                layer_energy = self._get_layer_energy(platform, design_tag, layer)

                latency_per_layer.append(layer_latency)
                energy_per_layer.append(layer_energy)

                platform_latency += layer_latency
                platform_energy += layer_energy

            latency_per_design.append(latency_per_layer)
            energy_per_design.append(energy_per_layer)
            area_per_design.append([self._get_area_platform(platform, design_tag)])

            platform_energy_per_design.append(platform_energy)
            platform_latency_per_design.append(platform_latency)

        # Check bit-width, analyze memory requirements and generate successors needed for link analysis
        ls = {}
        dmem = []

        for j in range(last_pp + 1, pp+1):
            layer = self.schedule[j]

            valid &= self._check_layer_bitwidth(platform, layer)
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

        # Decide which design should be used
        metric_per_design = calc_metric(np.array(energy_per_design), np.array(latency_per_design), np.array(area_per_design), "edp", reduction=True)

        #optimal_design_id  = np.argmin(metric_per_design)
        optimal_design_idx = np.argmin(metric_per_design)
        optimal_design_tag = design_tags[optimal_design_idx]

        l_pp.append(platform_latency_per_design[optimal_design_idx])
        e_pp.append(platform_energy_per_design[optimal_design_idx])
        return valid, optimal_design_tag, part_l_params + max(dmem, default=0) # mem evaluation

    def _get_tag_as_int(self, design_tag: str):
        #tag: str = self.nodeStats[platform]["eval"][design_id]["tag"]
        tag = design_tag.split("_")[-1]
        return int(tag)

    def _get_layer_latency(self, platform : int, design_tag: str, layer_name : str) -> float:
        if self.nodeStats[platform]["eval"][design_tag]["layers"].get(layer_name):
            return float(self.nodeStats[platform]["eval"][design_tag]["layers"][layer_name]['latency'])
        else:
            return 0

    def _get_layer_energy(self, platform : int, design_tag: str,  layer_name : str) -> float:
        if self.nodeStats[platform]["eval"][design_tag]["layers"].get(layer_name):
            return float(self.nodeStats[platform]["eval"][design_tag]["layers"][layer_name]['energy'])
        else:
            return 0

    def _check_layer_bitwidth(self, platform : int, layer_name : str) -> bool:
        if 'input' in layer_name or 'output' in layer_name:
            return True
        bit_width = self.nodeStats[platform].get("bits")

        if isinstance(bit_width, int):
            return bit_width >= max([self.q_constr[x] for x in self.q_constr.keys() if layer_name in x], default=0)
        else:
            assert(isinstance(bit_width, dict))
            if layer_name in bit_width:
                return bit_width[layer_name] / 2 >= max([self.q_constr[x] for x in self.q_constr.keys() if layer_name in x], default=0)
            else:
                return True

    def _zero_division(self, a : float, b : float) -> float:
        return a / b if b else np.inf

    def _get_link_metrics(self, link_idx : int, pp : int, successors : list) -> tuple[float, float, float]:
        if len(self.links) == 0 or successors == []:
            return 0, 0, 0

        layers = []
        for layer in np.unique(successors):
            layers += self.layer_dict[layer]["predecessors"]
        layers = np.unique([layer for layer in layers if layer not in successors])

        # Only consider already executed layers
        layers = np.unique([layer for layer in layers if layer in list(self.layer_dict.keys())[:pp+1]])

        data_sizes = [np.prod(self.layer_dict[layer]["output_size"]) for layer in layers]

        if len(self.links) == 1:
            return self.links[0].eval(np.sum(data_sizes))
        else:
            return self.links[link_idx].eval(np.sum(data_sizes))

    def _get_throughput(self, th_pp : list, part_latency : deque) -> float:
        platform_index = defaultdict(list)
        for i, tup in enumerate(part_latency):
            platform_index[tup[0]].append(i)

        for _, indexes in platform_index.items():
            platform_latency = 0.0
            for i in range(min(indexes), max(indexes)+1):
                platform_latency += part_latency[i][1]

            th_pp.append(self._zero_division(1000.0, platform_latency))

        return np.min(th_pp)

    def _get_area(self, partitions : list, design_tag: list[str]) -> float:
        parts = {}
        for platform in range(self.num_platforms):
            parts[platform] = []
        for p in partitions:
            parts[p[0]].append(p[1:])

        area = 0.0
        for key in parts.keys():
            #platform = [*self.nodeStats][key]
            platform = list(self.nodeStats.keys())[key]
            tag = design_tag[platform]
            if self.nodeStats[platform]['type'] == 'mnsim':
                for part in parts[key]:
                    for l in self.schedule[part[0]+1:part[1]+1]:
                        if l in [*self.nodeStats[platform]["eval"][tag]["layers"]]:
                            area += float(self.nodeStats[platform]["eval"][tag]["layers"][l]['area'])
            else: # timeloop
                #part: partition
                for part in parts[key]:
                    if part[0] != part[1] and part[1] != 0: #partitions stores [platform, layer_ids], here check if layers are executed on platform
                        tag = design_tag[platform]
                        first_layer = [*self.nodeStats[platform]["eval"][tag]["layers"]][0]
                        area += float(self.nodeStats[platform]["eval"][tag]["layers"][first_layer]['area'])
                        break

        return area

    def _get_area_platform(self, platform: int, design_id: int):
        first_layer = [*self.nodeStats[platform]["eval"][design_id]["layers"]][0]
        area = float(self.nodeStats[platform]["eval"][design_id]["layers"][first_layer]['area'])
        return area

    def _check_system_constraints(self, constraints, energy, latency, throughput, area) -> bool:
        valid = True
        if (latency_constraint := constraints.get("latency", math.inf)) and latency > latency_constraint:
            valid = False
        if (energy_constraint := constraints.get("energy", math.inf)) and energy > energy_constraint:
            valid = False
        if (throughput_constraint := constraints.get("throughput", 0)) and abs(throughput) < throughput_constraint:
            # throughput is passed as a negative value
            valid = False
        if (area_constraint := constraints.get("area", math.inf)) and area > area_constraint:
            valid = False
        
        return valid
