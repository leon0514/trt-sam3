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
# 智能查找通过pip安装的pybind11
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_FOUND_RESULT
)
if(pybind11_FOUND_RESULT EQUAL 0)
    find_package(pybind11 REQUIRED HINTS ${pybind11_DIR})
    message(STATUS "Found pybind11: ${pybind11_VERSION} (at ${pybind11_DIR})")
else()
    message(WARNING "Could not auto-find pybind11. Falling back to standard search.")
    find_package(pybind11 REQUIRED)
endif()

# 手动查找TensorRT组件
message(STATUS "Finding TensorRT components manually...")
find_path(TENSORRT_INCLUDE_DIRS NvInfer.h HINTS /usr/include/x86_64-linux-gnu /usr/include REQUIRED)
find_library(NVINFER_LIBRARY nvinfer REQUIRED)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin REQUIRED)
find_library(NVONNXPARSER_LIBRARY nvonnxparser REQUIRED)
set(TENSORRT_LIBRARIES ${NVINFER_LIBRARY} ${NVINFER_PLUGIN_LIBRARY} ${NVONNXPARSER_LIBRARY})

# 查找其他依赖
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc videoio imgcodecs)
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)
find_package(Freetype REQUIRED)
find_package(OpenMP REQUIRED)

# =============================================================================
# 3. 收集源文件 (最佳实践：明确列出所有文件)
# =============================================================================
# A. 核心库的源文件
set(CORE_SOURCES
    src/common/createObject.cpp
    src/common/image.cpp
    src/common/memory.cu
    src/common/norm.cpp
    src/common/object.cpp
    src/common/tensorrt.cpp
    src/infer/infer.cpp
    src/infer/sam3infer.cpp
    src/kernels/postprocess.cu
    src/kernels/preprocess.cu
    src/kernels/process_kernel_warp.cu
    src/osd/cvx_text.cpp
    src/osd/osd.cpp
)

# B. Python绑定的源文件
set(INTERFACE_SOURCES
    src/interface.cpp
)

# C. C++可执行文件的源文件
set(EXECUTABLE_SOURCES
    src/main.cpp
)

# =============================================================================
# 4. 定义构建目标
# =============================================================================
# A. 定义核心静态库 (包含所有共享逻辑)
add_library(trtsam_core STATIC ${CORE_SOURCES})

# B. 定义Python模块 (由interface.cpp创建)
pybind11_add_module(trtsam3 SHARED ${INTERFACE_SOURCES})

# C. 定义纯C++可执行文件 (由main.cpp创建)
add_executable(pro ${EXECUTABLE_SOURCES})

# =============================================================================
# 5. 设置目标属性 (链接)
# =============================================================================
# --- A. 为核心库 trtsam_core 配置所有外部依赖 ---
# 使用PUBLIC，这样链接它的目标会自动继承这些属性
target_include_directories(trtsam_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${FREETYPE_INCLUDE_DIRS}
)

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