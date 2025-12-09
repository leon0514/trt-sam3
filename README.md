# TensorRT SAM3 (C++ 推理)

这是一个基于 TensorRT 的 SAM3 推理仓库（C++ 实现）。目前实现了图像预处理、图像编码、文本编码、decoder 解码和后处理流程，支持单张图片单文本提示的推理流程。

## 主要特点：
- 使用 TensorRT 引擎
- C++ + CUDA 实现预处理/后处理内核，适合在 GPU 上高效运行
- 支持基于文本提示和几何矩形框的 mask/box 输出
- 利用批处理和内存复用实现同时识别多个文本提示类别

## ONNX 模型以及 TensorRT 模型导出
参考 `https://github.com/jamjamjon/usls.git`

## 环境
- 服务器    
ubuntu 24.04
- 显卡
NVIDIA GeForce RTX 4090
- 镜像  
nvcr.io/nvidia/tensorrt:25.10-py3

## 识别效果
- 多单词 文本提示
可以同时识别多个类别
<div align="center">
   <img src="https://raw.githubusercontent.com/leon0514/trt-sam3/refs/heads/main/workspace/assert/demo_multi_class.jpg" width="80%"/>
</div>

- 几何提示
<div align="center">
   <img src="https://raw.githubusercontent.com/leon0514/trt-sam3/refs/heads/main/workspace/assert/demo_box.jpg" width="80%"/>
</div>

- 混合提示
<div align="center">
   <img src="https://raw.githubusercontent.com/leon0514/trt-sam3/refs/heads/main/workspace/assert/demo_mixed.jpg" width="80%"/>
</div>


## 速度
`50ms`左右

## 编译执行
生成可执行文件 `pro`
```
make pro
```
生成python绑定包 `trtsam3.so`
```
make all
```

## 引用
- 参考实现： `https://github.com/jamjamjon/usls.git`

## 许可与贡献
- 本仓库为个人/研究用途示例，欢迎 issue。
