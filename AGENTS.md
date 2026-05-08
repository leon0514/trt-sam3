# AGENTS.md — TensorRT SAM3 (C++ Inference)

> This file is intended for AI coding agents. It assumes zero prior knowledge of the project.

---

## Project Overview

`trt-sam3` is a high-performance C++/CUDA inference engine for **SAM3 (Segment Anything Model 3)**. It runs four separate TensorRT engines (Vision Encoder, Text Encoder, Geometry Encoder, Decoder) end-to-end on NVIDIA GPUs. The project supports:

- **Text prompts** — e.g. `"person"`, `"helmet"`
- **Box prompts** — geometric bounding boxes with positive/negative labels
- **Mixed prompts** — text + boxes combined
- **Cross-image prompting** — draw boxes on image A and run recognition on image B by caching geometry encoder features under a string label
- **Multi-class batching** — reuse vision features across multiple text prompts for the same image to improve throughput

The repository contains:
- A **C++ core library** (`libtrtsam_core.a`) with CUDA kernels
- A **pybind11 Python module** (`trtsam3`) for Python integration
- A **C++ executable** (`pro`) for standalone testing
- A **FastAPI web service** (`workspace/app/`) with a frontend UI
- **ONNX export scripts** (`workspace/script/export.py`) derived from the Hugging Face `transformers` SAM3 implementation

Primary documentation languages: English (`README.md`) and Chinese (`README_ZH.md`). Inline source code comments are predominantly **Chinese**.

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | C++17, CUDA 17, Python 3 |
| Build System | CMake >= 3.18 |
| Inference Backend | NVIDIA TensorRT |
| GPU Compute | CUDA Toolkit (cudart, cublas, cudnn) |
| Image Processing | OpenCV 4 |
| Python Binding | pybind11 (installed via pip) |
| Web Framework | FastAPI + Uvicorn |
| Text Tokenization | `tokenizers` (HuggingFace) |
| Font Rendering | FreeType |
| Parallelism | OpenMP |
| Base Docker Image | `nvcr.io/nvidia/tensorrt:25.10-py3` |

Tested environment: Ubuntu 24.04, NVIDIA GeForce RTX 4090, CUDA architectures **80, 86, 89**.

---

## Project Structure

```
.
├── CMakeLists.txt          # Main build configuration (Chinese comments)
├── Dockerfile              # Minimal image based on nvcr tensorrt
├── README.md / README_ZH.md
├── src/
│   ├── common/             # Core utilities and wrappers
│   │   ├── tensorrt.cpp/hpp     # TensorRT Engine abstraction
│   │   ├── object.cpp/hpp       # DetectionBox, Box, Segmentation, etc.
│   │   ├── memory.cu/hpp        # GPU memory helpers (tensor::Memory)
│   │   ├── image.cpp/hpp        # Minimal image helpers
│   │   ├── norm.cpp/hpp         # Normalization parameters for preprocessing
│   │   ├── affine.hpp           # Affine transformation matrices
│   │   ├── check.hpp            # CUDA runtime check macros
│   │   ├── timer.hpp            # GPU event timer
│   │   ├── device.hpp           # CUDA device helpers
│   │   ├── cpm.hpp              # (unclear from header, likely concurrency)
│   │   └── createObject.cpp/hpp # Object creation helpers
│   ├── infer/              # Inference orchestration
│   │   ├── infer.cpp/hpp        # Abstract base class InferBase
│   │   ├── sam3infer.cpp/hpp    # Main SAM3 inference engine
│   │   └── sam3type.hpp         # Input types (Sam3Input, Sam3PromptUnit, BoxPrompt)
│   ├── kernels/            # CUDA kernels
│   │   ├── preprocess.cu/cuh    # warpAffine + normalize kernels
│   │   ├── postprocess.cu/cuh   # Post-processing kernels
│   │   └── process_kernel_warp.cu/hpp  # Kernel launch wrappers
│   ├── osd/                # On-Screen Display (drawing)
│   │   ├── osd.cpp/hpp          # Main osd() function
│   │   ├── cvx_text.cpp/hpp     # FreeType-based text rendering
│   │   └── labelLayout.hpp      # Label layout calculations
│   ├── interface.cpp       # pybind11 Python module definition
│   └── main.cpp            # C++ demo executable
├── workspace/
│   ├── app/                # FastAPI web service
│   │   ├── server.py            # Uvicorn entrypoint, lifespan mgmt
│   │   ├── inference.py         # ModelManager singleton, core inference fns
│   │   ├── router_predict.py    # /predict API endpoints
│   │   ├── router_refine.py     # Refinement endpoints
│   │   ├── router_ui.py         # UI serving routes
│   │   ├── models.py            # Pydantic request/response models
│   │   ├── utils.py             # Post-processing utilities (NMS, merging)
│   │   ├── client.py            # HTTP client example
│   │   └── frontend/            # Static HTML/JS frontend
│   ├── script/             # ONNX export and reference scripts
│   │   ├── export.py            # PyTorch -> ONNX exporter for all 4 modules
│   │   ├── ort.py               # ONNXRuntime reference inference
│   │   └── tokenizer.py         # Tokenizer snippet
│   ├── demo.py             # Python demo script (uses trtsam3 module)
│   ├── images/             # Sample input images
│   ├── output/             # Sample output directory
│   └── engine-models/      # Expected location for .engine files
├── build/                  # CMake out-of-source build directory (created manually)
└── objs/                   # Object file output directory
```

---

## Build System

### Configuration

The project uses a single `CMakeLists.txt` at the repository root. It defines **three targets**:

1. **`trtsam_core`** — STATIC library containing all shared C++/CUDA source.
2. **`trtsam3`** — SHARED Python module built with `pybind11_add_module()`, linking `trtsam_core`.
3. **`pro`** — EXECUTABLE built from `src/main.cpp`, linking `trtsam_core`.

Key CMake settings:
- `CMAKE_CXX_STANDARD 17`, `CMAKE_CUDA_STANDARD 17`
- `CMAKE_CUDA_ARCHITECTURES 80 86 89`
- Output directories are set to `${CMAKE_BINARY_DIR}/workspace` so the built binaries sit next to `demo.py` and `app/`.
- Compile flags for the core library: `-O2 -w -fPIC`

### External Dependencies

CMake searches for the following (all required):
- **pybind11** — queried via `python3 -m pybind11 --cmakedir` first, then fallback.
- **TensorRT** — headers (`NvInfer.h`) and three libraries: `nvinfer`, `nvinfer_plugin`, `nvonnxparser`. Searches under `TENSORRT_ROOT`, `/usr/include/x86_64-linux-gnu`, `/usr/lib/x86_64-linux-gnu/`.
- **OpenCV 4** — components: `core imgproc videoio imgcodecs`
- **CUDAToolkit**
- **Python3** — `Development` component
- **Freetype**
- **OpenMP**

### Build Commands

```bash
# From repository root
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH="$(python3 -m pybind11 --cmakedir)"
make -j$(nproc)
```

### Run Targets

```bash
# Run Python demo (depends on trtsam3 module)
make run

# Run C++ executable (depends on pro)
make runpro
```

---

## Runtime Architecture

### Inference Pipeline

The `Sam3Infer` class (`src/infer/sam3infer.hpp`) orchestrates a 5-stage pipeline:

1. **Preprocess** — `cv::warpAffine` equivalent CUDA kernel resizes input to `1008x1008`, normalizes with `alpha=1/127.5, beta=-1`, and optionally swaps R/B channels.
2. **Vision Encode** — TensorRT engine `vision-encoder.engine` extracts FPN features (`fpn_feat_0/1/2`, `fpn_pos_2`).
3. **Prompt Encode**
   - *Text*: runs `text-encoder.engine` with pre-registered `input_ids` + `attention_mask` (length fixed at 32).
   - *Geometry*: runs `geometry-encoder.engine` with box coordinates (cxcywh normalized) and labels; results are cached by a string label.
4. **Decode** — `decoder.engine` consumes vision features + prompt features and outputs `pred_masks`, `pred_boxes`, `pred_logits`, `presence_logits`.
5. **Postprocess** — CUDA kernels filter by confidence threshold, apply sigmoid, perform NMS-like selection, resize masks back to original image size, and map boxes via inverse affine matrices.

### Batching & Memory Limits

Hard-coded capacity limits inside `Sam3Infer` (tuned for RTX 4090):
- `max_image_batch_ = 2` — max images fed to Vision Encoder simultaneously.
- `max_prompt_batch_ = 4` — max prompts fed to Decoder simultaneously.
- `max_boxes_per_prompt_ = 20` — max geometric boxes per prompt.
- `num_queries_ = 200` — decoder object queries.

All GPU buffers are pre-allocated once in `allocate_memory_once()` and reused across inference calls.

### Prompt Types

| Prompt | C++ Type | Description |
|--------|----------|-------------|
| Text | `Sam3PromptUnit(text="person", boxes=[])` | Class name string matched against pre-registered token IDs. |
| Box | `Sam3PromptUnit(text="", boxes=[("pos", [x1,y1,x2,y2])])` | Geometric bounding boxes on the target image. |
| Mixed | `Sam3PromptUnit(text="tie", boxes=[...])` | Both text and boxes active simultaneously. |
| Cached Geometry | `geom_label` string passed to `forwards()` | Reuses geometry features computed from a previous image. |

---

## Code Organization & Module Divisions

### `src/common/` — Infrastructure
- **TensorRT wrapper** (`tensorrt.cpp/hpp`) — abstracts `IExecutionContext`, dynamic shape setting, binding lookups.
- **Objects** (`object.cpp/hpp`) — Strongly typed detection structures: `Box`, `Pose`, `Obb`, `Segmentation`, `Depth`, `Track`, `DetectionBox`. Uses `std::optional` for optional fields.
- **Memory** (`memory.cu/hpp`) — RAII GPU memory container `tensor::Memory<T>`.
- **Norm** (`norm.cpp/hpp`) — Preprocessing normalization constants.
- **Check** (`check.hpp`) — CUDA error checking macros (`checkRuntime`, `checkKernel`).

### `src/infer/` — Business Logic
- **`InferBase`** — Abstract interface for all inference engines in this family.
- **`Sam3Infer`** — Concrete implementation. Contains the full pipeline, memory buffers, engine handles, and caches.
- **`sam3type.hpp`** — Input payload types used by both C++ and Python.

### `src/kernels/` — GPU Kernels
- `preprocess.cu` — `warp_affine_bilinear_and_normalize_plane_kernel` and single-channel variants.
- `postprocess.cu` — Sigmoid, filtering, mask resizing.
- `process_kernel_warp.cu` — Higher-level wrappers that launch the above kernels with grid/block dimensions.

### `src/osd/` — Visualization
- `osd.cpp/hpp` — Draws bounding boxes, segmentation masks, labels, pose skeletons, OBBs, tracks.
- `cvx_text.cpp/hpp` — Renders Chinese/English text using FreeType into OpenCV `cv::Mat`.
- `labelLayout.hpp` — Computes label background sizes to avoid text overlap.

### `src/interface.cpp` — Python Bindings
Exposes:
- `trtsam3.Sam3Infer` (factory `create_instance()`)
- `trtsam3.Sam3Input`, `trtsam3.Sam3PromptUnit`
- `trtsam3.DetectionBox`, `trtsam3.Box`, `trtsam3.Segmentation`
- `trtsam3.osd()` — in-place drawing function that accepts a numpy array

Numpy `<->` `cv::Mat` conversions share memory when possible; clones are used where in-place mutation is unsafe.

---

## Development Conventions

### Naming
- **Files**: mostly `snake_case.cpp` or `camelCase.cpp` (mixed legacy).
- **Classes**: `PascalCase` (e.g. `Sam3Infer`, `TensorRT::Engine`).
- **Methods**: `snake_case` for public APIs (`setup_text_inputs`, `forwards`).
- **Members**: trailing underscore for private members (`gpu_id_`, `fpn_feat_0_`).
- **Namespaces**: `object`, `TensorRT`, `tensor`, `cuda`, `norm_image`, `nv`.

### Headers
- Use `#pragma once` at the top of every header.
- Traditional include guards also present in some headers (e.g. `__TENSORRT_HPP__`).

### Comments
- Inline comments are written in **Chinese**.
- When modifying code, follow the existing bilingual style: keep Chinese comments for local logic and use English for public API docstrings if adding pybind11 bindings.

### Error Handling
- CUDA calls are wrapped with `checkRuntime(...)` macros that assert on failure.
- Python bindings validate array dimensions explicitly (e.g. `input_ids.size() != 32` throws `std::runtime_error`).

---

## Testing Strategy

**There is no formal unit test framework** (no GoogleTest, Catch2, etc.). Validation is performed via:

1. **`workspace/demo.py`** — Integration test script covering:
   - Multi-class text prompt inference
   - Pure box prompt inference
   - Mixed text + box prompts
   - Cross-image geometry prompt caching
2. **`src/main.cpp`** — C++ hard-coded test scenarios (`test_text_prompt`, `test_box_prompt`, `speed_test`).
3. **Visual inspection** — outputs are saved to `workspace/output/` and compared manually.

### How to Validate Changes

```bash
cd build/workspace
python3 ../demo.py          # Python path bindings test
./pro                       # C++ executable test
```

If modifying CUDA kernels, visually verify mask alignment and box coordinates against reference images in `workspace/assert/`.

---

## Deployment

### Docker

```bash
docker build -t trt-sam3 .
docker run --gpus all -p 8000:8000 -v $(pwd)/workspace:/workspace trt-sam3
```

The Dockerfile installs system deps (`libopencv-dev`, `libfreetype6-dev`) and Python deps (`opencv-python-headless`, `fastapi`, `uvicorn`, `tokenizers`, `pybind11`, etc.). The default `CMD` starts `python3 server.py` on port 8000.

### Web Service

The FastAPI app (`workspace/app/server.py`) initializes a `ModelManager` singleton on startup. It exposes:
- `/` — Serves `frontend/index.html`
- `/predict` — Object detection with text/box prompts
- `/refine` — Mask refinement endpoints
- Static files under `/static`

**Security note**: CORS is configured with `allow_origins=["*"]`. The server does not implement authentication. Do not expose directly to the public internet without a reverse proxy and auth layer.

### Model Assets

Before running inference, you must obtain or generate four TensorRT engine files and place them in `workspace/engine-models/`:

| Engine File | Source |
|-------------|--------|
| `vision-encoder.engine` | Export ONNX via `workspace/script/export.py`, then `trtexec` or TensorRT API build. |
| `text-encoder.engine` | Same as above. |
| `geometry-encoder.engine` | Same as above. |
| `decoder.engine` | Same as above. |
| `tokenizer.json` | HuggingFace `tokenizers` CLIP tokenizer export. |

Pre-exported ONNX models are available at: [HuggingFace — tangliyang/onnx_model_store](https://huggingface.co/tangliyang/onnx_model_store)

---

## Security Considerations

1. **No input sanitization on image paths** in `src/main.cpp`. It directly reads hard-coded relative paths. Do not run `pro` with elevated privileges in directories containing untrusted files.
2. **CORS wide open** in the FastAPI app. Add `allow_origins` restrictions before production deployment.
3. **Base64 image decoding** in the API could be abused for large payloads. Add size limits in `router_predict.py` if exposing externally.
4. **No model file integrity checks**. Ensure `.engine` files are loaded from a trusted path to avoid deserialization attacks on TensorRT engines.
5. **Python GIL handling**: `pybind11` bindings release the GIL during `forwards()` and `setup_geometry_input()` to allow multi-threaded Python callers. Ensure thread safety when modifying shared caches (e.g. `geom_features_cache_`).

---

## Quick Reference for Agents

| Task | File to Edit |
|------|--------------|
| Add a new CUDA kernel | `src/kernels/*.cu` + `src/kernels/*.cuh` |
| Expose new API to Python | `src/interface.cpp` |
| Change inference pipeline logic | `src/infer/sam3infer.cpp` / `sam3infer.hpp` |
| Change object data structures | `src/common/object.cpp` / `object.hpp` |
| Change drawing / visualization | `src/osd/osd.cpp` / `osd.hpp` |
| Change web API routes | `workspace/app/router_*.py` |
| Change build flags or targets | `CMakeLists.txt` |
| Add a new test scenario | `src/main.cpp` or `workspace/demo.py` |
