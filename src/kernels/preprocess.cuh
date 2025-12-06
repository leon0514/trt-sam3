#ifndef PREPROCESS_CUH__
#define PREPROCESS_CUH__

#include "common/norm.hpp"
#include <cuda_runtime.h>
#include <memory>

namespace cuda
{

// 核函数 三通道图片warpaffine操作 并且做归一化
__global__ void warp_affine_bilinear_and_normalize_plane_kernel(uint8_t *src,
                                                                int src_line_size,
                                                                int src_width,
                                                                int src_height,
                                                                float *dst,
                                                                int dst_width,
                                                                int dst_height,
                                                                uint8_t const_value_st,
                                                                float *warp_affine_matrix_2_3,
                                                                norm_image::Norm norm);

// 核函数 单通道图片warpaffine操作
__global__ void warp_affine_bilinear_single_channel_kernel(float *src,
                                                           int src_line_size,
                                                           int src_width,
                                                           int src_height,
                                                           float *dst,
                                                           int dst_width,
                                                           int dst_height,
                                                           float const_value_st,
                                                           float *warp_affine_matrix_2_3);

__global__ void warp_affine_bilinear_single_channel_mask_kernel(float *src,
                                                                int src_line_size,
                                                                int src_width,
                                                                int src_height,
                                                                uint8_t *dst,
                                                                int dst_width,
                                                                int dst_height,
                                                                float const_value_st,
                                                                float *warp_affine_matrix_2_3);

} // namespace cuda

#endif