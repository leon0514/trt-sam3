#include "kernels/postprocess.cuh"

namespace cuda
{

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
    float conf_threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_queries) {
        return;
    }

    float presence_score = sigmoid_gpu(presence_logits_gpu[0]);
    float score = sigmoid_gpu(pred_logits_gpu[idx]) * presence_score;

    if (score > conf_threshold) {
        int current_idx = atomicAdd(d_filtered_count, 1);
        filtered_scores_gpu[current_idx] = score;
        filtered_indices_gpu[current_idx] = idx;

        const float* box_ptr = pred_boxes_gpu + (idx * 4);
        float x1 = box_ptr[0] * orig_w;
        float y1 = box_ptr[1] * orig_h;
        float x2 = box_ptr[2] * orig_w;
        float y2 = box_ptr[3] * orig_h;

        filtered_boxes_gpu[current_idx * 4 + 0] = fmaxf(0.0f, fminf(x1, (float)orig_w));
        filtered_boxes_gpu[current_idx * 4 + 1] = fmaxf(0.0f, fminf(y1, (float)orig_h));
        filtered_boxes_gpu[current_idx * 4 + 2] = fmaxf(0.0f, fminf(x2, (float)orig_w));
        filtered_boxes_gpu[current_idx * 4 + 3] = fmaxf(0.0f, fminf(y2, (float)orig_h));
    }
}


/**
 * @brief CUDA Kernel - 对浮点数掩码进行阈值处理，并将结果转为 unsigned char。
 * @param src           (输入) 源浮点数掩码 (在GPU上)
 * @param dst           (输出) 目标 uchar 掩码 (在GPU上)
 * @param num_elements  掩码中的总像素数
 * @param threshold     用于二值化的阈值
 */
__global__ void threshold_and_convert_kernel(const float* src, unsigned char* dst,
                                             int num_elements, float threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    dst[idx] = (src[idx] > threshold) ? 255 : 0;
}

}