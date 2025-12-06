#include "common/check.hpp"
#include "kernels/postprocess.cuh"
#include "kernels/process_kernel_warp.hpp"

#define GPU_BLOCK_THREADS 512

static dim3 grid_dims(int numJobs)
{
    int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

static dim3 block_dims(int numJobs) { return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS; }

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
                                              cudaStream_t stream)
{
    dim3 grid((dst_width + 31) / 32, (dst_height + 31) / 32);
    dim3 block(32, 32);

    checkKernel(cuda::warp_affine_bilinear_and_normalize_plane_kernel<<<grid, block, 0, stream>>>(src,
                                                                                                  src_line_size,
                                                                                                  src_width,
                                                                                                  src_height,
                                                                                                  dst,
                                                                                                  dst_width,
                                                                                                  dst_height,
                                                                                                  const_value,
                                                                                                  matrix_2_3,
                                                                                                  norm));
}

void warp_affine_bilinear_single_channel_plane(float *src,
                                               int src_line_size,
                                               int src_width,
                                               int src_height,
                                               float *dst,
                                               int dst_width,
                                               int dst_height,
                                               float *matrix_2_3,
                                               float const_value,
                                               cudaStream_t stream)
{
    dim3 grid((dst_width + 31) / 32, (dst_height + 31) / 32);
    dim3 block(32, 32);

    checkKernel(cuda::warp_affine_bilinear_single_channel_kernel<<<grid, block, 0, stream>>>(src,
                                                                                             src_line_size,
                                                                                             src_width,
                                                                                             src_height,
                                                                                             dst,
                                                                                             dst_width,
                                                                                             dst_height,
                                                                                             const_value,
                                                                                             matrix_2_3));
}

void warp_affine_bilinear_single_channel_mask_plane(float *src,
                                                    int src_line_size,
                                                    int src_width,
                                                    int src_height,
                                                    uint8_t *dst,
                                                    int dst_width,
                                                    int dst_height,
                                                    float *matrix_2_3,
                                                    uint8_t const_value,
                                                    cudaStream_t stream)
{
    dim3 grid((dst_width + 31) / 32, (dst_height + 31) / 32);
    dim3 block(32, 32);

    checkKernel(cuda::warp_affine_bilinear_single_channel_mask_kernel<<<grid, block, 0, stream>>>(src,
                                                                                                  src_line_size,
                                                                                                  src_width,
                                                                                                  src_height,
                                                                                                  dst,
                                                                                                  dst_width,
                                                                                                  dst_height,
                                                                                                  const_value,
                                                                                                  matrix_2_3));
}


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
    cudaStream_t stream)
{
    dim3 block_size_1d(256);
    dim3 grid_size_1d((num_queries + block_size_1d.x - 1) / block_size_1d.x);

    cuda::sam3_postprocess_kernel<<<grid_size_1d, block_size_1d, 0, stream>>>(
        pred_masks_gpu,
        pred_boxes_gpu,
        pred_logits_gpu,
        presence_logits_gpu,
        filtered_boxes_gpu,
        filtered_indices_gpu,
        filtered_scores_gpu,
        filtered_count_gpu,
        num_queries, mask_height, mask_width,
        original_image_width, original_image_height,
        confidence_threshold);
}
