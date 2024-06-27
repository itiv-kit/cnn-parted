from pymoo.core.problem import ElementwiseProblem
from copy import deepcopy
import numpy as np
from collections import deque, defaultdict

from framework.link.Link import Link


class PartitioningProblem(ElementwiseProblem):
    def __init__(self, num_pp : int, nodeStats : dict, schedule : list, q_constr : dict, fixed_sys : bool, acc_once : bool, layer_dict : dict, layer_params : dict, link_confs : list):
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

        self.links = []
        for link_conf in link_confs:
            self.links.append(Link(link_conf))

        n_var = self.num_pp * 2 + 1 # Number of max. partitioning points + platform IDs
        n_obj = 6 # latency, energy, throughput, area + link latency + link energy
        n_constr = (self.num_pp + 1) + 1 + (self.num_pp + 1) * 2 + (self.num_pp + 1) * 2 # num_accelerator_platforms + num_real_pp + latency/energy per partition + latency/energy per link

        xu = self.num_platforms * self.num_layers - 1

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=1, xu=xu)

    def _evaluate(self, x : np.ndarray, out : dict, *args, **kwargs) -> None:
        valid = True
        num_real_pp = 0
        curr_latency = 0.0 # used for throughput calculation

        l_pp = []
        e_pp = []

        latency = energy = throughput = area = link_latency = link_energy = 0.0
        bandwidth = np.full((self.num_pp + 1), np.inf)
        mem = np.full((self.num_pp + 1), np.inf)

        p : list = x.tolist()
        p[0:self.num_pp] = np.divide(p[0:self.num_pp], self.num_platforms)
        p[self.num_pp:] = np.divide(p[self.num_pp:], self.num_layers)
        p = np.floor(p).astype(int) + 1
        p = np.insert(p, self.num_pp, self.num_layers)

        if not np.array_equal(np.sort(p[:self.num_pp]), p[:self.num_pp]): # keep order of partitioning points
            valid = False
        elif self.acc_once and np.unique(p[-self.num_pp-1:]).size != np.asarray(p[-self.num_pp-1:]).size:   # only use accelerator once
            valid = False
        elif self.fixed_sys and not np.array_equal(np.sort(p[-self.num_pp-1:]), p[-self.num_pp-1:]): # keep order of Accelerators
            valid = False
        else:
            #breakpoint()
            design_id = [] #should be list of len num_platform - num_links
            th_pp = []
            partitions = []
            part_latency = deque()
            l_pp_link = []
            e_pp_link = []
            successors = [self.schedule[0]]
            i = last_pp = last_platform = -1
            for i, pp in enumerate(p[0:self.num_pp+1], self.num_pp + 1):
                #breakpoint()
                v, optimal_design_id, mem[i-self.num_pp-1] = self._eval_partition(p[i], last_pp, pp, l_pp, e_pp, successors)
                valid &= v

                design_id.append(optimal_design_id)
                partitions.append([p[i], last_pp, pp])

                # evaluate link
                if last_pp != pp and last_platform != p[i]:
                    if last_platform != -1:
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
                            part_latency.append([last_platform, sum(l_pp[:-1]) - curr_latency])
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
                        last_platform = p[i]

            link_latency = sum(l_pp_link)
            link_energy = sum(e_pp_link)
            latency = sum(l_pp) + link_latency
            energy = sum(e_pp) + link_energy
            throughput = self._get_throughput(th_pp, part_latency) * -1
            area = self._get_area(partitions)

        out["F"] = [latency, energy, throughput, area, link_latency, link_energy] #+ list(bandwidth) #+ list(mem)

        if valid:
            out["G"] = [i * (-1) for i in design_id] + [-num_real_pp] + [i * (-1) for i in l_pp] + [i * (-1) for i in e_pp] + [i * (-1) for i in l_pp_link] + [i * (-1) for i in e_pp_link]
        else:
            out["G"] = [i for i in range(self.num_pp+1)] + [1] + [i for i in range(self.num_pp+1)] + [i for i in range(self.num_pp+1)] + [i for i in range(self.num_pp+1)] + [i for i in range(self.num_pp+1)]

    def _eval_partition(self, platform : int, last_pp : int, pp : int, l_pp : list, e_pp : list, successors : list) -> tuple[bool, int]:
        platform = list(self.nodeStats.keys())[platform-1]

        area_per_design = []
        latency_per_design = []
        energy_per_design = []

        platform_latency_per_design = []
        platform_energy_per_design = []

        for design_id, _ in enumerate(self.nodeStats[platform]["eval"]):
            valid = True
            platform_latency = 0.0
            platform_energy = 0.0
            part_l_params = 0

            ls = {}
            dmem = []

            energy_per_layer = []
            latency_per_layer = []

            for j in range(last_pp + 1, pp + 1):
                layer = self.schedule[j-1]
                #could be called for diff designs, then use best
                layer_latency = self._get_layer_latency(platform, design_id, layer)
                layer_energy = self._get_layer_energy(platform, design_id, layer)

                latency_per_layer.append(layer_latency)
                energy_per_layer.append(layer_energy)

                platform_latency += layer_latency
                platform_energy += layer_energy

            latency_per_design.append(latency_per_layer)
            energy_per_design.append(energy_per_layer)
            area_per_design.append([self._get_area_platform(platform, design_id)])

            platform_energy_per_design.append(platform_latency)
            platform_latency_per_design.append(platform_latency)

        # Check bit-width, analyze memory requirements and generate successors needed for link analysis
        for j in range(last_pp + 1, pp + 1):
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
        # The x_per_design arrays have this layout:
        #  x | l0 | l1 | l2 | l3 |
        # ------------------------
        # d0 | ...| ...| ...| ...|
        # d1 | ...| ...| ...| ...|
        edp_per_design_per_layer = np.multiply(np.array(energy_per_design), np.array(latency_per_design))
        edap_per_design_per_layer = np.multiply(edp_per_design_per_layer, np.array(area_per_design))

    	# Use Energy-Delay-Area-Product as criterium for optimality
        # edap_per_design is a column vector, each row has the EDAP for one design
        edap_per_design = np.sum(edap_per_design_per_layer, axis=1)
        optimal_design_id = np.argmax(edap_per_design)

        l_pp.append(platform_latency_per_design[optimal_design_id])
        e_pp.append(platform_energy_per_design[optimal_design_id])
        return valid, optimal_design_id, part_l_params + max(dmem, default=0) # mem evaluation

    def _get_layer_latency(self, platform : int, design_id: int, layer_name : str) -> float:
        if self.nodeStats[platform]["eval"][design_id].get(layer_name):
            return float(self.nodeStats[platform]["eval"][design_id][layer_name]['latency'])
        else:
            return 0

    def _get_layer_energy(self, platform : int, design_id: int,  layer_name : str) -> float:
        if self.nodeStats[platform]["eval"][design_id].get(layer_name):
            return float(self.nodeStats[platform]["eval"][design_id][layer_name]['energy'])
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
        platform_index = defaultdict(list)
        for i, tup in enumerate(part_latency):
            platform_index[tup[0]].append(i)

        for _, indexes in platform_index.items():
            platform_latency = 0.0
            for i in range(min(indexes), max(indexes)+1):
                platform_latency += part_latency[i][1]

            th_pp.append(self._zero_division(1000.0, platform_latency))

        return min(th_pp)

    def _get_area(self, partitions : list) -> float:
        parts = {}
        for platform in range(self.num_platforms):
            parts[platform] = []
        for p in partitions:
            parts[p[0]-1].append(p[1:])

        area = 0.0
        for key in parts.keys():
            platform = [*self.nodeStats][key]
            if self.nodeStats[platform]['type'] == 'mnsim':
                for part in parts[key]:
                    for l in self.schedule[part[0]:part[1]]:
                        if l in [*self.nodeStats[platform]]:
                            area += float(self.nodeStats[platform][l]['area'])
            else: # timeloop
                for part in parts[key]:
                    if part[0] != part[1]:
                        first_layer = [*self.nodeStats[platform]["eval"][0]][0]
                        area += float(self.nodeStats[platform]["eval"][0][first_layer]['area']) #TODO This should be adapted to account for different designs
                        break

        return area

    def _get_area_platform(self, platform: int, design_id: int):
        first_layer = [*self.nodeStats[platform]["eval"][design_id]][0]
        area = float(self.nodeStats[platform]["eval"][design_id][first_layer]['area'])
        return area
