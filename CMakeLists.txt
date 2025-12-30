# =============================================================================
# 1. 项目基本配置
# =============================================================================
cmake_minimum_required(VERSION 3.18)
project(TRTSam LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/workspace)
set(CMAKE_RUNTIME_OUTPUT_PATH ${CMAKE_BINARY_DIR}/workspace)

# =============================================================================
# 2. 查找依赖项
# =============================================================================
# --- 查找通过pip安装的pybind11 ---
find_package(pybind11 REQUIRED)
message(STATUS "Found pybind11: ${pybind11_VERSION} (installed via package manager)")

# --- 查找其他外部依赖 (保持不变) ---
message(STATUS "Finding TensorRT components manually (for Docker-like environments)...")
find_path(TENSORRT_INCLUDE_DIRS NvInfer.h HINTS /usr/include/x86_64-linux-gnu /usr/local/cuda/include /usr/include REQUIRED)
find_library(NVINFER_LIBRARY nvinfer REQUIRED)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin REQUIRED)
find_library(NVONNXPARSER_LIBRARY nvonnxparser REQUIRED)
set(TENSORRT_LIBRARIES ${NVINFER_LIBRARY} ${NVINFER_PLUGIN_LIBRARY} ${NVONNXPARSER_LIBRARY})

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc videoio imgcodecs)
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)
find_package(Freetype REQUIRED)
find_package(OpenMP REQUIRED)

# =============================================================================
# 3. 收集源文件
# =============================================================================
# 收集所有核心逻辑的源文件 (不包含绑定文件 interface.cpp)
file(GLOB_RECURSE CORE_SOURCES "src/*.cpp" "src/*.cu")
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*/interface\\.cpp$") # 核心库不包含绑定文件

# =============================================================================
# 4. 定义构建目标
# =============================================================================
# A. 定义核心静态库 (包含除了interface.cpp之外的所有代码)
add_library(trtsam_core STATIC ${CORE_SOURCES})

# B. 使用 pybind11_add_module 创建Python模块 (编译 interface.cpp)
# 这个函数由 find_package(pybind11) 提供，会自动处理命名和链接
pybind11_add_module(trtsam3 SHARED src/interface.cpp)

# C. 定义纯C++可执行文件 (由核心源文件构成)
add_executable(pro ${CORE_SOURCES})

target_include_directories(trtsam_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${FREETYPE_INCLUDE_DIRS}
)

# 使用PUBLIC，这样链接 trtsam_core 的目标会自动继承这些依赖
target_link_libraries(trtsam_core PUBLIC
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARIES}
    cudart cublas cudnn
    Freetype::Freetype
    OpenMP::OpenMP_CXX
    "dl"
)
target_compile_options(trtsam_core PUBLIC -O2 -w -fPIC)

# --- B. 为最终目标链接核心库 ---
# Python模块 和 C++可执行文件 都依赖于核心库
target_link_libraries(trtsam3 PRIVATE trtsam_core)
target_link_libraries(pro PRIVATE trtsam_core)


# --- C. 为最终产物设置RPATH ---
set_target_properties(trtsam3 pro PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)
# =============================================================================
# 6. 添加自定义命令
# =============================================================================
add_custom_target(run COMMAND python3 demo.py WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_PATH} DEPENDS trtsam3)
add_custom_target(runpro COMMAND ./pro WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_PATH} DEPENDS pro)