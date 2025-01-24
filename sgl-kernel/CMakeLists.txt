cmake_minimum_required(VERSION 3.18)
project(sgl_kernel LANGUAGES CXX CUDA)

# find python
find_package(Python REQUIRED COMPONENTS Development)
# find Torch
execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

# find torch_python
execute_process(
    COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_INSTALL_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
link_directories(${TORCH_INSTALL_PREFIX}/lib)
include_directories(${TORCH_INSTALL_PREFIX}/include)

# 设置 CUDA 架构
set(CUDA_ARCHS "75;80;89;90")
if (ENABLE_SM90A)
    list(APPEND CUDA_ARCHS "90a")
endif()

# 设置 CUDA 编译标志
set(NVCC_FLAGS
    "-DNDEBUG"
    "-O3"
    "-Xcompiler"
    "-fPIC"
    "-std=c++17"
    "-use_fast_math"
    "-DFLASHINFER_ENABLE_F16"
)

# 根据 CUDA 版本和设备架构添加额外的编译标志
if (TORCH_CUDA_AVAILABLE)
    if (CUDA_VERSION_MAJOR GREATER_EQUAL 12 AND SM_VERSION GREATER_EQUAL 90)
        list(APPEND NVCC_FLAGS "-gencode=arch=compute_90a,code=sm_90a")
    endif()
    if (SM_VERSION GREATER_EQUAL 90)
        list(APPEND NVCC_FLAGS
            "-DFLASHINFER_ENABLE_FP8"
            "-DFLASHINFER_ENABLE_FP8_E4M3"
            "-DFLASHINFER_ENABLE_FP8_E5M2"
        )
    endif()
    if (SM_VERSION GREATER_EQUAL 80)
        list(APPEND NVCC_FLAGS "-DFLASHINFER_ENABLE_BF16")
    endif()
else()
    if (ENABLE_SM90A)
        list(APPEND NVCC_FLAGS "-gencode=arch=compute_90a,code=sm_90a")
    endif()
    if (ENABLE_FP8)
        list(APPEND NVCC_FLAGS
            "-DFLASHINFER_ENABLE_FP8"
            "-DFLASHINFER_ENABLE_FP8_E4M3"
            "-DFLASHINFER_ENABLE_FP8_E5M2"
        )
    endif()
    if (ENABLE_BF16)
        list(APPEND NVCC_FLAGS "-DFLASHINFER_ENABLE_BF16")
    endif()
endif()

# 移除不需要的 CUDA 标志
foreach(flag IN ITEMS
    "-D__CUDA_NO_HALF_OPERATORS__"
    "-D__CUDA_NO_HALF_CONVERSIONS__"
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__"
    "-D__CUDA_NO_HALF2_OPERATORS__"
)
    string(REPLACE "${flag}" "" NVCC_FLAGS "${NVCC_FLAGS}")
endforeach()

# 设置 C++ 编译标志
set(CXX_FLAGS "-O3")

# 设置链接库
set(LIBRARIES c10 torch torch_python cuda)

# 设置额外的链接参数
set(EXTRA_LINK_ARGS "-Wl,-rpath,$ORIGIN/../../torch/lib" "-L/usr/lib/x86_64-linux-gnu")

# 添加包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/3rdparty/cutlass/include
    ${CMAKE_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
    ${CMAKE_SOURCE_DIR}/src/sgl-kernel/csrc
    ${CMAKE_SOURCE_DIR}/3rdparty/flashinfer/include
    ${CMAKE_SOURCE_DIR}/3rdparty/flashinfer/include/gemm
    ${CMAKE_SOURCE_DIR}/3rdparty/flashinfer/csrc
)

# 添加 CUDA 源文件
set(CUDA_SOURCES
    src/sgl-kernel/csrc/trt_reduce_internal.cu
    src/sgl-kernel/csrc/trt_reduce_kernel.cu
    src/sgl-kernel/csrc/moe_align_kernel.cu
    src/sgl-kernel/csrc/int8_gemm_kernel.cu
    src/sgl-kernel/csrc/sampling_scaling_penalties.cu
    src/sgl-kernel/csrc/lightning_attention_decode_kernel.cu
    src/sgl-kernel/csrc/sgl_kernel_ops.cu
    src/sgl-kernel/csrc/rotary_embedding.cu
    3rdparty/flashinfer/csrc/activation.cu
    3rdparty/flashinfer/csrc/bmm_fp8.cu
    3rdparty/flashinfer/csrc/group_gemm.cu
    3rdparty/flashinfer/csrc/group_gemm_sm90.cu
    3rdparty/flashinfer/csrc/norm.cu
    3rdparty/flashinfer/csrc/sampling.cu
    3rdparty/flashinfer/csrc/renorm.cu
)

set(GPU_USE_SABI, ON)


Python_add_library(_kernels MODULE 
    USE_SABI ${GPU_USE_SABI}
    WITH_SOABI
    ${CUDA_SOURCES}
)

# 设置目标属性
set_property(TARGET _kernels PROPERTY CXX_STANDARD 17)

# 设置编译选项
target_compile_options(_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>
    ${CXX_FLAGS}
)

# 设置编译定义
target_compile_definitions(_kernels PRIVATE
    "-DTORCH_EXTENSION_NAME=_kernels"
)

# 设置包含目录
target_include_directories(_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/3rdparty/cutlass/include
    ${CMAKE_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
    ${CMAKE_SOURCE_DIR}/src/sgl-kernel/csrc
    ${CMAKE_SOURCE_DIR}/3rdparty/flashinfer/include
    ${CMAKE_SOURCE_DIR}/3rdparty/flashinfer/include/gemm
    ${CMAKE_SOURCE_DIR}/3rdparty/flashinfer/csrc
)

# 设置链接库
target_link_libraries(_kernels PRIVATE
    torch
    ${LIBRARIES}
    CUDA::cudart
    CUDA::cuda_driver
)

# 安装目标
install(TARGETS _kernels
    LIBRARY DESTINATION sgl_kernel/ops
    COMPONENT _kernels
)
