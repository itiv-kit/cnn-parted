import math
import itertools as it

from framework.model.onnx_model import OnnxModel
from framework.noc.noc import NocTopology
from framework.model.layer_tracker import LayerBindings
from framework.model.layers.compute_layer import ComputationLayer
from framework.scheduling import SystemSchedule
from framework.node.node_evaluator import SystemResult


def _score_even_finish(schedules: SystemSchedule) -> float:
    sched_finish_times = [sched.cur_time for sched in schedules.node_schedules.values()]
    latest_finish = max(sched_finish_times)
    total_idling_at_end = 0.0
    for sched in schedules.node_schedules.values():
        total_idling_at_end += (latest_finish - sched.cur_time)
    return total_idling_at_end

def _score_idle_times(schedules: SystemSchedule) -> float:
    # Prevent idle times during execution
    idle_times = [sched.get_idle_time() for sched in schedules.node_schedules.values()]
    return sum(idle_times)

def _score_max_latency(schedules: SystemSchedule) -> float:
    latencies = [sched.cur_time for sched in schedules.node_schedules.values()]
    return max(latencies)


def _score_edp(node_stats: SystemResult, tracker: LayerBindings) -> float:
    # For each layer we want to find the most efficient platform
    total_edp = 0.0    
    for nw, layer, bind in tracker:
        # TODO Currently assumes that there is only one design
        layer_data = node_stats[bind].designs[0].networks[nw].layers[layer]
        total_edp += layer_data.edp
    return total_edp

def _score_binding_energy(node_stats: SystemResult, tracker: LayerBindings) -> float:
    # For each layer we want to find the most efficient platform
    total_energy = 0.0    
    for nw, layer, bind in tracker:
        # TODO Currently assumes that there is only one design
        layer_data = node_stats[bind].designs[0].networks[nw].layers[layer]
        total_energy += layer_data.energy
    return total_energy

def _score_binding_latency(node_stats: SystemResult, tracker: LayerBindings) -> float:
    # For each layer we want to find the most efficient platform
    total_latency = 0.0    
    for nw, layer, bind in tracker:
        # TODO Currently assumes that there is only one design
        layer_data = node_stats[bind].designs[0].networks[nw].layers[layer]
        total_latency += layer_data.latency
    return total_latency

def _score_static_max_latency(node_stats: SystemResult, tracker:LayerBindings) -> float:
    latencies  = {id: 0 for id in node_stats.get_platform_ids()}
    for nw, layer, binding in tracker:
        layer_data = node_stats[binding].designs[0].networks[nw].layers[layer]
        latencies[binding] += layer_data.latency
    return max(latencies.values())

def _score_latency_fairness_jain(node_stats: SystemResult, tracker: LayerBindings) -> float:
    """Calculate jains fairness index -> to be maximized"""
    latencies  = {id: 0 for id in node_stats.get_platform_ids()}
    for nw, layer, binding in tracker:
        layer_data = node_stats[binding].designs[0].networks[nw].layers[layer]
        latencies[binding] += layer_data.latency #* self.models[nw].frames
    lats = list(latencies.values())
    
    x = sum(lats)**2
    y = node_stats.get_num_platforms() * (sum([lat**2 for lat in lats])) 
    J = x / y
    return J

def _score_latency_fairness_std_dev(node_stats: SystemResult, tracker: LayerBindings) -> float:
    """Calculate std deviation of latency on different platforms"""
    latencies  = {id: 0 for id in node_stats.get_platform_ids()}
    for nw, layer, binding in tracker:
        layer_data = node_stats[binding].designs[0].networks[nw].layers[layer]
        latencies[binding] += layer_data.latency #* self.models[nw].frames
    lats = list(latencies.values())
    mean = (1/len(lats)) * sum(lats)
    var = (1/len(lats)) * sum( (lat - mean)**2 for lat in latencies) 
    return math.sqrt(var)

def _score_latency_fairness_gini(node_stats: SystemResult,  tracker: LayerBindings) -> float:
    """Calculate Gini coefficient to measure equality of latency on different platforms"""
    latencies  = {id: 0 for id in node_stats.get_platform_ids()}
    for nw, layer, binding in tracker:
        layer_data = node_stats[binding].designs[0].networks[nw].layers[layer]
        latencies[binding] += layer_data.latency #* self.models[nw].frames
    lats = list(latencies.values())

    mean = (1/len(lats)) * sum(lats)
    x = sum([abs(li - lj) for li, lj in it.permutations(lats, 2)])
    y = 2 * node_stats.get_num_platforms()**2*mean
    gini = x / y
    return gini

def _score_latency_fairness_max_min_space(node_stats: SystemResult,  tracker: LayerBindings) -> float:
    """Calculate difference of max and min accelerators to score equality of latency on different platforms"""
    latencies  = {id: 0 for id in node_stats.get_platform_ids()}
    for nw, layer, binding in tracker:
        layer_data = node_stats[binding].designs[0].networks[nw].layers[layer]
        latencies[binding] += layer_data.latency #* self.models[nw].frames
    lats = list(latencies.values())

    return max(lats) - min(lats)


def _score_scheduling_proxy(node_stats: SystemResult, tracker: LayerBindings, models: dict[str, OnnxModel]) -> float:

    # Determine when the last accelerator finishes
    latest_platform_finish_time = _score_static_max_latency(tracker)

    # Determine when the last network finishes
    network_finish_time: dict[str, float] = {}
    for network, model in models.items():
        layer_finish_time: dict[str, float] = {layer.layer_name: 0 for layer in model.nodes}
        for layer in model.nodes:

            # Set the base value for the layer finish time: the time to process it
            if isinstance(layer, ComputationLayer):
                # assumption that only computation layers have an execution time
                layer_binding = tracker.get_binding(network, layer.layer_name)
                layer_finish_time[layer.layer_name] = node_stats[layer_binding].design.networks[network].layers[layer.layer_name].latency
            else:
                layer_finish_time[layer.layer_name] = 0

            pred_latencies = []
            for pred in model.predecessors(layer):
                pred_latencies.append(layer_finish_time[pred.layer_name])

            if len(pred_latencies) > 0:
                layer_finish_time[layer.layer_name] += max(pred_latencies)

        network_finish_time[network] = max(layer_finish_time.values())
    
    # The network that takes longest to execute
    latest_network_finish_time = max(network_finish_time.values())

    return max(latest_platform_finish_time, latest_network_finish_time)

def _score_tensor_transfer_energy(tracker: LayerBindings, models: dict[str, OnnxModel], noc: NocTopology,  ) -> float:
    total_energy = 0.0    
    for nw, layer, bind in tracker:
        # Besides executing the layer, consider the energy to transfer the input and output
        # tensors between the platform and the DRAM
        platform_coord = noc.id_to_node[bind].coord
        num_hops = noc.get_min_hops_to_dram(platform_coord)
        model_layer = models[nw].node_by_layer_name(layer)
        tensor_num_elem = [math.prod(tensor.shape) for tensor in model_layer.inputs + [model_layer.output]]
        tensor_transfer_eval = [noc.default_link.eval(num_elem) for num_elem in tensor_num_elem]
        tensor_transfer_energy = [energy * num_hops for (latency, energy, slice_size) in tensor_transfer_eval]
        total_energy += sum(tensor_transfer_energy)
    return total_energy

def _score_total_binding_energy(tracker: LayerBindings, models: dict[str, OnnxModel], noc: NocTopology,) -> float:
    return _score_binding_energy(tracker) + _score_tensor_transfer_energy(noc, models, tracker)

_SCORING_REGISTRY = {
    "schedule_even_finish": _score_even_finish,
    "schedule_idle_times": _score_idle_times,
    "schedule_max_latency": _score_max_latency,
    
    "edp": _score_edp,
    "binding_energy": _score_binding_energy,
    "binding_latency": _score_binding_latency,
    "latency": _score_binding_latency, #alias for now
    "static_max_latency": _score_static_max_latency,
    "jain": _score_latency_fairness_jain,
    "std_dev": _score_latency_fairness_std_dev,
    "gini": _score_latency_fairness_gini,
    "max_min": _score_latency_fairness_max_min_space,

    "crit_path": _score_scheduling_proxy,

    "tensor_energy": _score_tensor_transfer_energy,
    "energy": _score_total_binding_energy,
}

def scoring_dispatch(score_fn: str, node_stats: SystemResult, tracker: LayerBindings,
                     models: dict[str, OnnxModel], noc: NocTopology, schedules: SystemSchedule = None):
    assert score_fn in _SCORING_REGISTRY, f"Unknown scoring function {score_fn}, supported are: {_SCORING_REGISTRY.keys()}"

    # To resolve the different function signatures, dispatch is performed through this match statement
    match score_fn:
        case "schedule_even_finish" | "schedule_idle_times" | "schedule_max_latency":
            return _SCORING_REGISTRY[score_fn](schedules)

        case "edp" | "binding_energy" | "latency" | "static_max_latency" | "jain" | "std_dev" | "gini" | "max_min":
            return _SCORING_REGISTRY[score_fn](node_stats, tracker)

        case "crit_path":
            return _SCORING_REGISTRY[score_fn](node_stats, tracker, models)
        
        case "tensor_energy" | "energy":
            return _SCORING_REGISTRY[score_fn](tracker, models, noc)
