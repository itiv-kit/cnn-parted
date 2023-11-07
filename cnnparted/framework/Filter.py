import numpy as np

class Filter:
    def __init__(self, memoryInfo, config_helper, partition_points, mems):
        self.memoryInfo = memoryInfo
        self.config_helper = config_helper
        self.partition_points = partition_points
        self.node_components,_ = config_helper.get_system_components()
        self.constraints = config_helper.get_constraints()
        self.mems = mems
        self.num_bytes = config_helper.get_num_bytes()

        self.part_1_memory = {}
        self.part_2_memory = {}
        self.nodes_memory = {}
        self.part_1_memory_filtered = []
        self.part_2_memory_filtered = []
        self.part_max_layer = {}
        self.partpoints_filtered = []

    def apply_filter(self):
        # Get memories for each partition
        self.part_1_memory, self.part_2_memory = self.memoryInfo.get_memory_for_2_partitions(
            self.partition_points, self.mems, self.num_bytes
        )
        self.nodes_memory = {
            self.node_components[0]["id"]: self.part_1_memory,
            self.node_components[1]["id"]: self.part_2_memory
        }

        # Get memory size limits for nodes
        memory_sizes = self.config_helper.get_node_memory_sizes(self.node_components)

        # Filter partition points
        self.part_1_memory_filtered = [
            layer for layer in self.part_1_memory if self.part_1_memory[layer] <= memory_sizes[0]
        ]
        self.part_2_memory_filtered = [
            layer for layer in self.part_2_memory if self.part_2_memory[layer] <= memory_sizes[1]
        ]

        # Identify maximum layer based on memory constraints
        self.part_max_layer = {
            self.node_components[0]["id"]: self.part_1_memory_filtered[-1] if self.part_1_memory_filtered else None,
            self.node_components[1]["id"]: self.part_2_memory_filtered[0] if self.part_2_memory_filtered else None
        }

        # Apply filter for partition points based on output size and memory constraints
        max_size = self.constraints.get("max_out_size", 0)  # Default to 0 if not specified
        self.partpoints_filtered = [
            layer for layer in self.partition_points
            if ((np.prod(layer.get("output_size", [])) <= max_size or max_size == 0) and
                self.part_1_memory.get(layer["name"], float('inf')) <= memory_sizes[0] and
                self.part_2_memory.get(layer["name"], float('inf')) < memory_sizes[1])
        ]

        return self.partpoints_filtered, self.part_max_layer,self.nodes_memory
