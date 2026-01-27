[English](README.md) | [中文](README_ZH.md)

# TensorRT SAM3 (C++ Inference)

This is a TensorRT-based SAM3 inference repository (C++ implementation). It currently implements image preprocessing, image encoding, text encoding, decoder decoding, and post-processing processes, supporting multi-text prompt inference for images.

## Key Features:
- Uses TensorRT engine
- C++ + CUDA implementation of preprocessing/post-processing kernels, suitable for efficient GPU operation
- Supports mask/box output based on text prompts and geometric bounding boxes
- Utilizes batching and memory reuse to simultaneously recognize multiple text prompt categories
- Draw boxes on image A, recognize on image B

## ONNX Model and TensorRT Model Export
- Refer to the repository below to export ONNX models  
[https://github.com/jamjamjon/usls.git](https://github.com/jamjamjon/usls.git)

- Address of already exported ONNX models    
[https://huggingface.co/tangliyang/onnx_model_store](https://huggingface.co/tangliyang/onnx_model_store)

## Vision Encode Model Quantization
- Refer to the repository below to perform int8 quantization on the SAM3 vision encode model
[https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows/onnx_ptq/sam2](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows/onnx_ptq/sam2)

## Environment
- Server    
ubuntu 24.04
- GPU
NVIDIA GeForce RTX 4090
- Image  
nvcr.io/nvidia/tensorrt:25.10-py3

## Recognition Results
- Multi-word Text Prompts
Can simultaneously recognize multiple categories
<div align="center">
   <img src="https://raw.githubusercontent.com/leon0514/trt-sam3/refs/heads/main/workspace/assert/demo_multi_class.jpg" width="80%"/>
</div>

- Geometric Prompts
<div align="center">
   <img src="https://raw.githubusercontent.com/leon0514/trt-sam3/refs/heads/main/workspace/assert/demo_box.jpg" width="80%"/>
</div>

- Mixed Prompts
<div align="center">
   <img src="https://raw.githubusercontent.com/leon0514/trt-sam3/refs/heads/main/workspace/assert/demo_mixed.jpg" width="80%"/>
</div>

- Prompt boxes on image A, recognition on image B
<div align="center">
   <img src="https://raw.githubusercontent.com/leon0514/trt-sam3/refs/heads/main/workspace/assert/A.jpg" width="80%"/>
</div>

<div align="center">
   <img src="https://raw.githubusercontent.com/leon0514/trt-sam3/refs/heads/main/workspace/assert/B.jpg" width="80%"/>
</div>

## Speed
Around `50ms`

## Build and Run
```bash
cmake .. -DCMAKE_PREFIX_PATH="$(python3 -m pybind11 --cmakedir)"
make -j$(nproc)
```

## web UI
<div align="center">
   <img src="https://raw.githubusercontent.com/leon0514/trt-sam3/refs/heads/main/workspace/assert/web.jpg" width="80%"/>
</div>

## References
[https://github.com/jamjamjon/usls.git](https://github.com/jamjamjon/usls.git)

## License and Contributions
- This repository is an example for personal/research use, welcome issues.
