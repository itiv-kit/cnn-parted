# ONNX Runtime works with different hardware acceleration libraries through its extensible 
# Execution Providers (EP) framework to optimally execute the ONNX models on the hardware platform. 
# This interface enables flexibility for the AP application developer to deploy their ONNX models in 
# different environments in the cloud and the edge and optimize the execution by taking advantage of
# the compute capabilities of the platform.

#https://onnxruntime.ai/docs/execution-providers/


providers = {
    'cpu': 'CPUExecutionProvider',              # CPU execution provider for running ONNX models on CPU.
    'cuda': 'CUDAExecutionProvider',            # The CUDA Execution Provider enables hardware accelerated computation on Nvidia CUDA-enabled GPUs
    'tensorrt': 'TensorRTExecutionProvider',    # TensorRT execution provider for NVIDIA GPUs, optimized for deep learning inference.
    'directml': 'DirectMLExecutionProvider',    # DirectML execution provider for Windows devices, optimized for GPU acceleration.
    'openvino': 'OpenVINOExecutionProvider',    # OpenVINO execution provider for running on Intel hardware, optimized for deep learning.
    'nuphar': 'NupharExecutionProvider',        # Nuphar execution provider for FPGA and custom hardware acceleration.
    'dnnl': 'DnnlExecutionProvider',            # DNNL execution provider for deep learning acceleration on Intel architectures.
    'rocm': 'ROCmExecutionProvider',            # ROCm execution provider for AMD GPUs.
    'vitisai': 'VitisAIExecutionProvider',       # Vitis-AI execution provider for Xilinx FPGAs and AI acceleration.
    'azure' : 'AzureExecutionProvider'          #The Azure Execution Provider enables ONNX Runtime to invoke a remote Azure endpoint for inference
}
