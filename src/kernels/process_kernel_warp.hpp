#ifndef KERNELWARP_HPP__
#define KERNELWARP_HPP__

#include "common/norm.hpp"
#include "kernels/preprocess.cuh"
#include "kernels/postprocess.cuh"

void warp_affine_bilinear_and_normalize_plane(uint8_t *src,
                                              int src_line_size,
                                              int src_width,
                                              int src_height,
                                              float *dst,
                                              int dst_width,
                                              int dst_height,
                                              float *matrix_2_3,
                                              uint8_t const_value,
                                              const norm_image::Norm &norm,
                                              cudaStream_t stream);

void warp_affine_bilinear_single_channel_plane(float *src,
                                               int src_line_size,
                                               int src_width,
                                               int src_height,
                                               float *dst,
                                               int dst_width,
                                               int dst_height,
                                               float *matrix_2_3,
                                               float const_value,
                                               cudaStream_t stream);

void warp_affine_bilinear_single_channel_mask_plane(float *src,
                                                    int src_line_size,
                                                    int src_width,
                                                    int src_height,
                                                    uint8_t *dst,
                                                    int dst_width,
                                                    int dst_height,
                                                    float *matrix_2_3,
                                                    uint8_t const_value,
                                                    cudaStream_t stream);


void sam3_postprocess_plane(const float* pred_masks_gpu,
    const float* pred_boxes_gpu,
    const float* pred_logits_gpu,
    const float* presence_logits_gpu,
    float* filtered_boxes_gpu,
    int* filtered_indices_gpu,
    float* filtered_scores_gpu,
    int* filtered_count_gpu,
    int num_queries,
    int mask_height, int mask_width,
    int original_image_width, int original_image_height,
    float confidence_threshold,
    cudaStream_t stream);
#endif