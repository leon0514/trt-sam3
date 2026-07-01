# TRT SAM3 Web API 文档

> 本文档描述 `workspace/app/` 中 FastAPI 服务暴露的所有 HTTP 接口。
> 基础路径：`http://<host>:8000`
> 服务启动：`python workspace/app/server.py`

---

## 目录

1. [通用说明](#1-通用说明)
2. [数据模型](#2-数据模型)
3. [接口列表](#3-接口列表)
   - [`GET /`](#31-get-)
   - [`POST /predict`](#32-post-predict)
   - [`POST /predict/file`](#33-post-predictfile)
   - [`POST /predict-obj-refine`](#34-post-predict-obj-refine)
4. [Mask RLE 格式](#4-mask-rle-格式)
5. [错误码](#5-错误码)
6. [调用示例](#6-调用示例)

---

## 1. 通用说明

### 1.1 服务启动与访问

```bash
cd workspace/app
python server.py
```

默认监听 `0.0.0.0:8000`：

- Web UI：`http://localhost:8000/`
- Swagger UI：`http://localhost:8000/docs`
- ReDoc：`http://localhost:8000/redoc`
- OpenAPI Schema：`http://localhost:8000/openapi.json`

### 1.2 CORS

当前配置为 `allow_origins=["*"]`，开发阶段允许任意来源跨域访问。生产环境请务必限制来源。

### 1.3 认证

当前未实现任何认证机制。若部署到公网，请在反向代理层补充 Basic Auth / OAuth / API Key 等认证。

### 1.4 图片格式

- **Base64 接口**：接收标准 Base64 编码字符串，支持带 `data:image/xxx;base64,` 前缀或纯编码字符串。
- **文件上传接口**：接收 `multipart/form-data`，图片字段类型为 `UploadFile`，支持常见图片格式（JPEG、PNG、BMP 等）。

所有图片在服务端通过 OpenCV 解码为 **BGR 3 通道 Mat** 后送入 TensorRT 引擎。

### 1.5 推理模式总览

| 模式 | 说明 | 适用接口 |
|------|------|----------|
| `multi-class` | 一个或多个文本类别提示，批量推理 | `/predict/file` |
| `box` | 纯几何框提示（正/负样本框） | `/predict/file`、JSON 接口 |
| `mixed` | 单个文本 + 几何框混合提示 | `/predict/file`、JSON 接口 |
| `from-image` | 跨图几何提示，将参考图 Box 迁移到目标图 | `/predict/file` |
| `obj-refine` | 先预检测裁剪，再在裁剪区域精细识别 | `/predict/file`、`/predict-obj-refine` |

---

## 2. 数据模型

### 2.1 `BoxInput`

几何框输入，用于 `prompts[i].boxes`。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `label` | `string` | 否 | `"pos"` | 框类型，`"pos"` 表示正样本，`"neg"` 表示负样本 |
| `bbox` | `List[float]` | 是 | - | 左上角 / 右下角坐标 `[x1, y1, x2, y2]`，像素坐标 |

```json
{
  "label": "pos",
  "bbox": [100.0, 200.0, 300.0, 400.0]
}
```

### 2.2 `PromptUnit`

一个提示单元，可包含文本和若干几何框。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | `string` | 否 | `""` | 文本提示，例如 `"person"`、`"helmet"` |
| `boxes` | `List[BoxInput]` | 否 | `[]` | 该提示关联的几何框列表 |

```json
{
  "text": "person",
  "boxes": [
    {"label": "pos", "bbox": [100, 200, 300, 400]}
  ]
}
```

### 2.3 `CropConfig`

`obj-refine` 模式下的 OmniCrop 高级配置。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `max_size` | `int` | 否 | `640` | 裁剪块最大边长 |
| `padding` | `int` | 否 | `20` | 裁剪块边缘留白像素 |
| `w_diou` | `float` | 否 | `30.0` | 距离惩罚权重 |
| `w_expansion` | `float` | 否 | `5.0` | 扩展惩罚权重 |
| `count_penalty` | `float` | 否 | `120.0` | 裁剪块数量惩罚 |
| `nms_threshold` | `float` | 否 | `0.2` | 裁剪框之间 NMS 阈值 |
| `enable_ar_fix` | `bool` | 否 | `true` | 是否启用宽高比修正 |
| `target_ar` | `float` | 否 | `1.0` | 目标宽高比 |

### 2.4 `InferenceRequest`

Base64 JSON 推理请求体。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `image_base64` | `string` | 是 | `""` | Base64 编码的目标图片 |
| `confidence_threshold` | `float` | 否 | `0.5` | 置信度阈值，低于该值的检测结果会被过滤 |
| `pre_detect_labels` | `List[string]` | 否 | `["person"]` | 预检测标签，用于 `obj-refine` 定位裁剪区域 |
| `prompts` | `List[PromptUnit]` | 是 | - | 提示单元列表 |
| `return_mask` | `bool` | 否 | `false` | 是否返回分割 Mask |
| `merge_results` | `bool` | 否 | `true` | 是否在 `obj-refine` 中合并原图与裁剪区域结果 |
| `crop_config` | `CropConfig` | 否 | `null` | OmniCrop 详细配置 |

### 2.5 `DetectionResult`

单个检测结果。

| 字段 | 类型 | 说明 |
|------|------|------|
| `label` | `string` | 类别标签 |
| `score` | `float` | 置信度分数 |
| `box` | `List[float]` | 检测框 `[left, top, right, bottom]` |
| `mask` | `List[int]\|null` | RLE 编码的分割 Mask，见第 4 节 |
| `mask_width` | `int\|null` | Mask 宽度 |
| `mask_height` | `int\|null` | Mask 高度 |

### 2.6 `InferenceResponse`

统一响应体。

| 字段 | 类型 | 说明 |
|------|------|------|
| `results` | `List[DetectionResult]` | 检测结果列表 |

---

## 3. 接口列表

### 3.1 `GET /`

#### 功能

返回前端 `frontend/index.html` 页面。

#### 请求

无参数。

#### 响应

- `200 OK`：HTML 页面内容。
- 静态资源通过 `/static/*` 提供。

#### 说明

该接口主要用于浏览器直接访问 Web UI。API 调用一般不需要请求此接口。

---

### 3.2 `POST /predict`

#### 功能

Base64 图片 + 结构化 Prompt 的统一推理接口。支持纯文本、纯 Box、文本+Box 混合提示。

#### Content-Type

```
application/json
```

#### 请求体：`InferenceRequest`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `image_base64` | `string` | 是 | `""` | Base64 图片 |
| `confidence_threshold` | `float` | 否 | `0.5` | 置信度阈值 |
| `pre_detect_labels` | `List[string]` | 否 | `["person"]` | 当前接口逻辑中不会被使用 |
| `prompts` | `List[PromptUnit]` | 是 | - | 提示单元 |
| `return_mask` | `bool` | 否 | `false` | 是否返回 Mask |
| `merge_results` | `bool` | 否 | `true` | 当前接口逻辑中不会被使用 |
| `crop_config` | `CropConfig` | 否 | `null` | 当前接口逻辑中不会被使用 |

#### 提示处理规则

后端会按以下规则拆分 `prompts`：

1. **纯文本 Prompt**：`text` 非空且 `boxes` 为空，聚合为一个 `multi-class` 批量推理。
2. **纯 Box Prompt**：`text` 为空且 `boxes` 非空，单独调用 `box` 推理。
3. **混合 Prompt**：`text` 和 `boxes` 均非空，单独调用 `mixed` 推理。

每个 PromptUnit 独立推理，最终结果合并返回。

#### 响应：`InferenceResponse`

```json
{
  "results": [
    {
      "label": "person",
      "score": 0.92,
      "box": [120.0, 80.0, 340.0, 560.0],
      "mask": [1, 5, 100, 20, ...],
      "mask_width": 640,
      "mask_height": 480
    }
  ]
}
```

#### 错误响应

- `400 Bad Request`：JSON 解析失败、Base64 解码失败或图片解码失败。
- `500 Internal Server Error`：TensorRT 推理异常。

#### 调用示例

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64-string>",
    "confidence_threshold": 0.3,
    "return_mask": true,
    "prompts": [
      {"text": "person"},
      {"text": "car"}
    ]
  }'
```

---

### 3.3 `POST /predict/file`

#### 功能

文件上传版统一推理接口，前端 Web UI 主要使用此接口。支持所有 5 种推理模式。

#### Content-Type

```
multipart/form-data
```

#### 请求参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `mode` | `string` | 否 | `"multi-class"` | 推理模式，见下表 |
| `image` | `UploadFile` | 是 | - | 目标图片文件 |
| `class_names` | `List[string]` | 模式依赖 | `[]` | 文本类别标签，multi-class / mixed / obj-refine 使用 |
| `target_boxes` | `string` | 模式依赖 | `null` | JSON 字符串的几何框列表，box / mixed 使用 |
| `prompt_image` | `UploadFile` | 模式依赖 | `null` | 参考图片文件，from-image 使用 |
| `prompt_boxes` | `string` | 模式依赖 | `null` | JSON 字符串的参考图 Box 列表，from-image 使用 |
| `pre_detect_labels` | `string` | 模式依赖 | `"person"` | 逗号分隔的预检测标签，obj-refine 使用 |
| `merge_results` | `bool` | 否 | `true` | 是否合并原图与裁剪结果，obj-refine 使用 |
| `crop_config_json` | `string` | 否 | `null` | JSON 字符串的 `CropConfig`，obj-refine 使用 |
| `confidence` | `float` | 否 | `0.3` | 置信度阈值 |
| `return_mask` | `bool` | 否 | `true` | 是否返回 Mask |

#### `mode` 详细说明

| mode | 必填字段 | 说明 |
|------|----------|------|
| `multi-class` | `image`、`class_names` | 对 `class_names` 中每个类别单独做文本提示推理，结果合并。 |
| `box` | `image`、`target_boxes` | 仅根据几何框做分割/检测，`class_names` 被忽略。 |
| `mixed` | `image`、`class_names`（至少第一项）、`target_boxes` | 第一个 `class_names` 作为文本提示，与 `target_boxes` 同时输入。 |
| `from-image` | `image`、`prompt_image`、`prompt_boxes` | 将参考图上的 Box 几何特征迁移到目标图。 |
| `obj-refine` | `image`、`class_names`、`pre_detect_labels` | 先用 `pre_detect_labels` 在原图做预检测并裁剪，再用 `class_names` 在裁剪块上做精细识别。 |

#### `target_boxes` / `prompt_boxes` JSON 格式

```json
[
  {"type": "pos", "x1": 100, "y1": 200, "x2": 300, "y2": 400},
  {"type": "neg", "x1": 50, "y1": 50, "x2": 150, "y2": 150}
]
```

#### `pre_detect_labels` 格式

逗号分隔字符串，例如：

```
person,car,bicycle
```

#### `crop_config_json` 格式

`CropConfig` 的 JSON 字符串：

```json
{
  "max_size": 640,
  "padding": 20,
  "w_diou": 30.0,
  "w_expansion": 5.0,
  "count_penalty": 120.0,
  "nms_threshold": 0.2,
  "enable_ar_fix": true,
  "target_ar": 1.0
}
```

#### 响应：`InferenceResponse`

与 `/predict` 相同。

#### 错误响应

- `400 Bad Request`：缺少必填字段、图片解码失败、JSON 解析失败、不支持的 mode。
- `500 Internal Server Error`：TensorRT 推理异常或几何提示初始化失败。

#### 调用示例

**multi-class**

```bash
curl -X POST "http://localhost:8000/predict/file" \
  -F "mode=multi-class" \
  -F "image=@target.jpg" \
  -F "class_names=person" \
  -F "class_names=car" \
  -F "confidence=0.3" \
  -F "return_mask=true"
```

**box**

```bash
curl -X POST "http://localhost:8000/predict/file" \
  -F "mode=box" \
  -F "image=@target.jpg" \
  -F 'target_boxes=[{"type":"pos","x1":100,"y1":200,"x2":300,"y2":400}]' \
  -F "confidence=0.3"
```

**mixed**

```bash
curl -X POST "http://localhost:8000/predict/file" \
  -F "mode=mixed" \
  -F "image=@target.jpg" \
  -F "class_names=person" \
  -F 'target_boxes=[{"type":"pos","x1":100,"y1":200,"x2":300,"y2":400}]' \
  -F "confidence=0.3"
```

**from-image**

```bash
curl -X POST "http://localhost:8000/predict/file" \
  -F "mode=from-image" \
  -F "image=@target.jpg" \
  -F "prompt_image=@reference.jpg" \
  -F 'prompt_boxes=[{"type":"pos","x1":50,"y1":50,"x2":200,"y2":250}]' \
  -F "confidence=0.3"
```

**obj-refine**

```bash
curl -X POST "http://localhost:8000/predict/file" \
  -F "mode=obj-refine" \
  -F "image=@target.jpg" \
  -F "class_names=helmet" \
  -F "pre_detect_labels=person" \
  -F "merge_results=true" \
  -F "confidence=0.3" \
  -F 'crop_config_json={"max_size":640,"padding":20}'
```

---

### 3.4 `POST /predict-obj-refine`

#### 功能

专门针对 `obj-refine` 场景的 Base64 JSON 接口。与 `/predict/file` 的 `obj-refine` 模式等价，但请求体为 JSON。

#### Content-Type

```
application/json
```

#### 请求体：`InferenceRequest`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `image_base64` | `string` | 是 | `""` | Base64 图片 |
| `confidence_threshold` | `float` | 否 | `0.5` | 置信度阈值 |
| `pre_detect_labels` | `List[string]` | 否 | `["person"]` | 预检测标签 |
| `prompts` | `List[PromptUnit]` | 是 | - | 精细检测文本提示，每个 `PromptUnit` 的 `text` 字段会被提取为 `refine_texts`；与 `pre_detect_labels` 重复的标签会被自动过滤 |
| `return_mask` | `bool` | 否 | `false` | 是否返回 Mask |
| `merge_results` | `bool` | 否 | `true` | 是否合并原图与裁剪结果 |
| `crop_config` | `CropConfig` | 否 | `null` | OmniCrop 配置 |

#### 标签过滤规则

后端会从 `prompts` 中提取所有 `text`，并过滤掉与 `pre_detect_labels` 大小写不敏感相同的标签。

例如：

- `pre_detect_labels = ["person"]`
- `prompts = [{"text": "person"}, {"text": "helmet"}]`

最终 `refine_texts = ["helmet"]`，避免重复检测预检测标签自身。

#### 响应：`InferenceResponse`

与 `/predict` 相同。

#### 错误响应

- `400 Bad Request`：Base64 解码失败或图片解码失败。
- `500 Internal Server Error`：TensorRT 推理异常。

#### 调用示例

```bash
curl -X POST "http://localhost:8000/predict-obj-refine" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64-string>",
    "confidence_threshold": 0.3,
    "pre_detect_labels": ["person"],
    "prompts": [
      {"text": "helmet"},
      {"text": "vest"}
    ],
    "return_mask": true,
    "merge_results": true,
    "crop_config": {
      "max_size": 640,
      "padding": 20
    }
  }'
```

---

## 4. Mask RLE 格式

当 `return_mask=true` 时，返回的 `mask` 字段为扁平 RLE（Run-Length Encoding）数组：

```
[start1, len1, start2, len2, ...]
```

- 像素按 **行主序（row-major）** 排列，索引从 0 开始。
- `start` 为 **1-based** 起始像素索引。
- `len` 为连续前景像素长度。
- 掩码尺寸由 `mask_width` 和 `mask_height` 给出。

前端解码示例（JavaScript）：

```javascript
function decodeRle(rle, width, height) {
    const total = width * height;
    const mask = new Uint8Array(total);
    for (let i = 0; i < rle.length; i += 2) {
        const start = rle[i] - 1;      // 转 0-based
        const len = rle[i + 1];
        for (let j = 0; j < len; j++) {
            mask[start + j] = 255;
        }
    }
    return mask; // 长度为 width * height
}
```

---

## 5. 错误码

| HTTP 状态码 | 含义 | 常见原因 |
|------------|------|---------|
| 200 | 成功 | 请求正常完成 |
| 400 | 请求参数错误 | 缺少必填字段、图片/Base64 解码失败、JSON 格式错误、不支持的 mode |
| 500 | 服务端内部错误 | TensorRT 推理失败、几何提示初始化失败、模型未正确加载 |

错误响应体示例：

```json
{
  "detail": "class_names is required for multi-class mode"
}
```

---

## 6. 调用示例

### 6.1 Python 请求示例（Base64 + `/predict`）

```python
import base64
import requests

with open("target.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {
    "image_base64": b64,
    "confidence_threshold": 0.3,
    "return_mask": True,
    "prompts": [
        {"text": "person"},
        {"text": "car"}
    ]
}

resp = requests.post("http://localhost:8000/predict", json=payload)
print(resp.json())
```

### 6.2 Python 请求示例（文件上传 + `/predict/file`）

```python
import requests

with open("target.jpg", "rb") as f:
    files = {"image": ("target.jpg", f, "image/jpeg")}
    data = {
        "mode": "multi-class",
        "class_names": ["person", "car"],
        "confidence": 0.3,
        "return_mask": "true"
    }
    resp = requests.post("http://localhost:8000/predict/file", files=files, data=data)
    print(resp.json())
```

### 6.3 结果可视化示例（Python + OpenCV）

```python
import cv2
import numpy as np
import requests

# 假设已拿到 results
results = resp.json()["results"]
image = cv2.imread("target.jpg")

for r in results:
    left, top, right, bottom = map(int, r["box"])
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image, f"{r['label']} {r['score']:.2f}",
                (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if r.get("mask") and r.get("mask_width") and r.get("mask_height"):
        rle = r["mask"]
        w, h = r["mask_width"], r["mask_height"]
        mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(0, len(rle), 2):
            start = rle[i] - 1
            length = rle[i + 1]
            mask.flat[start:start + length] = 255
        # 如需原图尺寸 mask，需根据预处理仿射矩阵做反向映射

cv2.imwrite("output.jpg", image)
```

---

## 附录：OpenAPI 自动文档

服务启动后，可通过以下地址查看由 FastAPI 自动生成的交互式文档：

- Swagger UI：`http://localhost:8000/docs`
- ReDoc：`http://localhost:8000/redoc`
- OpenAPI JSON：`http://localhost:8000/openapi.json`
