# TensorRT SAM3 (C++ 推理)

这是一个基于 TensorRT 的 SAM3 推理仓库（C++ 实现）。目前实现了图像预处理、图像编码、文本编码、decoder 解码和后处理流程，支持单张图片单文本提示的推理流程。

## 主要特点：
- 使用 TensorRT 引擎
- C++ + CUDA 实现预处理/后处理内核，适合在 GPU 上高效运行
- 支持基于文本提示的 mask/box 输出


## ONNX 模型以及 TensorRT 模型导出
参考 `https://github.com/jamjamjon/usls.git`

## 环境
- 服务器    
ubuntu 24.04
- 镜像  
nvcr.io/nvidia/tensorrt:25.10-py3

## 编译执行
```shell
make pro
cd workspace
./pro
```
```
TensorRT-Engine 🌱 is Dynamic Shape model
Inputs: 2
        0.input_ids : {-1 x 32} [int64]
        1.attention_mask : {-1 x 32} [int64]
Outputs: 2
        0.text_features : {-1 x 32 x 256} [float32]
        1.text_mask : {-1 x 32} [bool]
------------------------------------------------------
------------------------------------------------------
TensorRT-Engine 🌱 is Dynamic Shape model
Inputs: 6
        0.fpn_feat_0 : {-1 x 256 x 288 x 288} [float32]
        1.fpn_feat_1 : {-1 x 256 x 144 x 144} [float32]
        2.fpn_feat_2 : {-1 x 256 x 72 x 72} [float32]
        3.fpn_pos_2 : {-1 x 256 x 72 x 72} [float32]
        4.prompt_features : {-1 x -1 x 256} [float32]
        5.prompt_mask : {-1 x -1} [bool]
Outputs: 4
        0.pred_masks : {-1 x 200 x 288 x 288} [float32]
        1.pred_boxes : {-1 x 200 x 4} [float32]
        2.pred_logits : {-1 x 200} [float32]
        3.presence_logits : {-1 x 1} [float32]
------------------------------------------------------
Inference engine loaded successfully.
Iteration 1: Detected 66 objects.
Iteration 2: Detected 66 objects.
Iteration 3: Detected 66 objects.
Iteration 4: Detected 66 objects.
Iteration 5: Detected 66 objects.
[⏰ 10 inferences] : 1227.49890 ms
Detected 66 objects.
```

## TODO
1. 多batch支持
2. 画框识别
3. 文本自动tokenizer

## 引用
- 参考实现： `https://github.com/jamjamjon/usls.git`

## 许可与贡献
- 本仓库为个人/研究用途示例，欢迎 issue。
