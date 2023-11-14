import collections
import configparser
import math

import sys
import os
from framework.constants import ROOT_DIR
sys.path.append(os.path.join(ROOT_DIR, "tools", "MNSIM-2.0"))

from MNSIM.Interface.network import NetworkGraph
from MNSIM.Interface.interface import TrainTestInterface
from MNSIM.Latency_Model.Model_latency import Model_latency
from MNSIM.Energy_Model.Model_energy import Model_energy

class MNSIMInterface(TrainTestInterface):
    def __init__ (self, layers : list, config : dict, input_size : list) -> None:
        self.SimConfig = ROOT_DIR + config.get('conf_path')
        self.input_size = input_size

        # load simconfig
        ## xbar_size, input_bit, weight_bit, quantize_bit
        xbar_config = configparser.ConfigParser()
        xbar_config.read(self.SimConfig, encoding = 'UTF-8')
        self.hardware_config = collections.OrderedDict()
        # xbar_size
        xbar_size = list(map(int, xbar_config.get('Crossbar level', 'Xbar_Size').split(',')))
        self.xbar_row = xbar_size[0]
        self.xbar_column = xbar_size[1]
        self.hardware_config['xbar_size'] = xbar_size[0]
        # xbar bit
        self.xbar_bit = int(xbar_config.get('Device level', 'Device_Level'))
        self.hardware_config['weight_bit'] = math.floor(math.log2(self.xbar_bit))
        # input bit and ADC bit
        ADC_choice = int(xbar_config.get('Interface level', 'ADC_Choice'))
        DAC_choice = int(xbar_config.get('Interface level', 'DAC_Choice'))
        temp_DAC_bit = int(xbar_config.get('Interface level', 'DAC_Precision'))
        temp_ADC_bit = int(xbar_config.get('Interface level', 'ADC_Precision'))
        ADC_precision_dict = {
            -1: temp_ADC_bit,
            1: 10,
            # reference: A 10b 1.5GS/s Pipelined-SAR ADC with Background Second-Stage Common-Mode Regulation and Offset Calibration in 14nm CMOS FinFET
            2: 8,
            # reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
            3: 8,  # reference: A >3GHz ERBW 1.1GS/s 8b Two-Step SAR ADC with Recursive-Weight DAC
            4: 6,  # reference: Area-Efficient 1GS/s 6b SAR ADC with Charge-Injection-Cell-Based DAC
            5: 8,  # ASPDAC1
            6: 6,  # ASPDAC2
            7: 4  # ASPDAC3
        }
        DAC_precision_dict = {
            -1: temp_DAC_bit,
            1: 1,  # 1-bit
            2: 2,  # 2-bit
            3: 3,  # 3-bit
            4: 4,  # 4-bit
            5: 6,  # 6-bit
            6: 8  # 8-bit
        }
        self.input_bit = DAC_precision_dict[DAC_choice]
        self.quantize_bit = ADC_precision_dict[ADC_choice]
        self.hardware_config['input_bit'] = self.input_bit
        self.hardware_config['quantize_bit'] = self.quantize_bit
        # group num
        self.pe_group_num = int(xbar_config.get('Process element level', 'Group_Num'))
        self.tile_size = list(map(int, xbar_config.get('Tile level', 'PE_Num').split(',')))
        self.tile_row = self.tile_size[0]
        self.tile_column = self.tile_size[1]

        self.net = self._get_net(layers)

        self.stats = {}

    def _find_rel_layer_idx(self, layers : list, l_idx : int, name: str) -> int:
        for i in range(l_idx-1,0, -1):
            if layers[i].get('output')[0] == name:
                return i - l_idx

    def _get_net(self, layers : list, hardware_config : dict = None) -> NetworkGraph:
        if hardware_config == None:
            hardware_config = {'xbar_size': 512, 'input_bit': 2, 'weight_bit': 1, 'quantize_bit': 10}
        layer_config_list = []
        quantize_config_list = []
        input_index_list = []

        for l in layers:
            op_type = l.get('op_type')
            lyr = None
            if op_type == "Conv":
                conv_params = l.get('conv_params')
                ich = conv_params.get('c')
                och = conv_params.get('m')
                krl = conv_params.get('r')
                pad = conv_params.get('wpad')
                srd = conv_params.get('wstride')
                lyr = {'type': 'conv', 'in_channels': ich, 'out_channels': och, 'kernel_size': krl, 'padding': pad, 'stride': srd}
            elif op_type == "MaxPool":
                pool_params = l.get('pool_params')
                krl = pool_params.get('kernel')[0]
                pad = pool_params.get('padding')[0]
                srd = pool_params.get('stride')[0]
                lyr = {'type': 'pooling', 'mode': 'MAX', 'kernel_size': krl,  'padding': pad, 'stride': srd}
            elif op_type == "AveragePool":
                pool_params = l.get('pool_params')
                krl = pool_params.get('kernel')[0]
                srd = pool_params.get('stride')[0]
                lyr = {'type': 'pooling', 'mode': 'AVE', 'kernel_size': krl, 'stride': srd}
            elif op_type == "GlobalAveragePool":
                input_size = layers[layers.index(l) - 1].get('output_size')
                lyr = {'type': 'pooling', 'mode': 'AVE', 'kernel_size': input_size[2], 'stride': 1}
            elif op_type == "Relu":
                lyr = {'type': 'relu'}
            elif op_type == "Add":
                lyr = {'type': 'element_sum'}
            elif op_type == "Flatten":
                lyr = {'type': 'view'}
            elif op_type == "Gemm":
                input_size = layers[layers.index(l) - 1].get('output_size')
                num_classes = l.get('output_size')[1]
                lyr = {'type': 'fc', 'in_features': input_size[1], 'out_features': num_classes}
            elif op_type == "Concat":
                lyr = {'type': 'concat'}
            elif op_type == 1: # DNN input/output
                continue
            else:
                print("[MNSIMInterface] Warning: unknown Op_type ", op_type)
                continue

            inputs = l.get('input')
            l_idx = layers.index(l)
            input_idx = []
            for i in inputs:
                if i.endswith("output_0"):
                    input_idx.append(self._find_rel_layer_idx(layers, l_idx, i))
            if len(input_idx) != 0:
                lyr['input_index'] = sorted(input_idx, reverse=True)

            lyr['name'] = l.get('name')
            layer_config_list.append(lyr)

        for i in range(len(layer_config_list)):
            quantize_config_list.append({'weight_bit': 9, 'activation_bit': 9, 'point_shift': -2})
            if 'input_index' in layer_config_list[i]:
                input_index_list.append(layer_config_list[i]['input_index'])
            else:
                input_index_list.append([-1])
        input_params = {'activation_scale': 1. / 255., 'activation_bit': 9, 'input_shape': self.input_size}
        # add bn for every conv
        L = len(layer_config_list)
        for i in range(L-1, -1, -1):
            if layer_config_list[i]['type'] == 'conv':
                # continue
                layer_config_list.insert(i+1, {'name': 'bn' + str(i), 'type': 'bn', 'features': layer_config_list[i]['out_channels']})
                quantize_config_list.insert(i+1, {'weight_bit': 9, 'activation_bit': 9, 'point_shift': -2})
                input_index_list.insert(i+1, [-1])
                for j in range(i + 2, len(layer_config_list), 1):
                    for relative_input_index in range(len(input_index_list[j])):
                        if j + input_index_list[j][relative_input_index] < i + 1:
                            input_index_list[j][relative_input_index] -= 1
        net = NetworkGraph(hardware_config, layer_config_list, quantize_config_list, input_index_list, input_params)
        return net

    def run(self):
        struct_file = self.get_structure()
        mod_l = Model_latency(struct_file, self.SimConfig)
        mod_e = Model_energy(struct_file, self.SimConfig)
        mod_l.calculate_model_latency()

        for idx, layer in enumerate(struct_file):
            latency = (max(mod_l.finish_time[idx]) - max(mod_l.finish_time[idx - 1])) if idx > 0 else max(mod_l.finish_time[idx])
            energy = mod_e.arch_energy[idx]

            self.stats[layer[0][0].get('name')] = {}
            self.stats[layer[0][0].get('name')]["latency"] = latency / 1e6 # ns -> ms
            self.stats[layer[0][0].get('name')]["energy"] = energy / 1e6 # nJ -> mJ
