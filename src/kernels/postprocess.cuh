#ifndef POSTPROCESS_CUH__
#define POSTPROCESS_CUH__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm> // for std::max/min on host
#include <cmath>     // for fmaxf/fminf on device

namespace cuda
{

__device__ inline float sigmoid_gpu(float x) 
{
    return 1.0f / (1.0f + expf(-x));
}

__global__ void sam3_postprocess_kernel(
    const float* pred_masks_gpu,
    const float* pred_boxes_gpu,
    const float* pred_logits_gpu,
    const float* presence_logits_gpu,
    float* filtered_boxes_gpu,
    int* filtered_indices_gpu,
    float* filtered_scores_gpu,
    int* d_filtered_count,
    int num_queries,
    int MASK_H, int MASK_W,
    int orig_w, int orig_h,
    float conf_threshold);

__global__ void threshold_and_convert_kernel(const float* src, unsigned char* dst,
                                             int num_elements, float threshold);

}


#endif