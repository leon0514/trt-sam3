#include "infer/sam3infer.hpp"
#include "common/affine.hpp"
#include "common/image.hpp"
#include "kernels/process_kernel_warp.hpp"
#include "kernels/postprocess.cuh"
#include "common/createObject.hpp"

Sam3Infer::Sam3Infer(
    const std::string &vision_encoder_path,
    const std::string &text_encoder_path,
    const std::string &decoder_path,
    int gpu_id,
    float confidence_threshold) : InferBase()
{
    // 初始化分配内存/显存

    affine_matrix_tensor_.cpu(2 * 3);
    affine_matrix_tensor_.gpu(2 * 3);
    mask_affine_matrix_tensor_.cpu(2 * 3);
    mask_affine_matrix_tensor_.gpu(2 * 3);
    // Initialize tensors for preprocess
    preprocess_image_tensor_.cpu(1 * 3 * input_image_width_ * input_image_height_);
    preprocess_image_tensor_.gpu(1 * 3 * input_image_width_ * input_image_height_);
    // Initialize Image Encoder tensors
    encode_image_input_tensor_.cpu(1 * 3 * input_image_width_ * input_image_height_);
    encode_image_input_tensor_.gpu(1 * 3 * input_image_width_ * input_image_height_);
    // Initialize Image Encoder output tensors
    encode_image_output_fpn_feat_0_tensor_.cpu(1 * 256 * 288 * 288);
    encode_image_output_fpn_feat_0_tensor_.gpu(1 * 256 * 288 * 288);
    encode_image_output_fpn_feat_1_tensor_.cpu(1 * 256 * 144 * 144);
    encode_image_output_fpn_feat_1_tensor_.gpu(1 * 256 * 144 * 144);
    encode_image_output_fpn_feat_2_tensor_.cpu(1 * 256 * 72 * 72);
    encode_image_output_fpn_feat_2_tensor_.gpu(1 * 256 * 72 * 72);
    encode_image_output_fpn_pos_2_tensor_.cpu(1 * 256 * 72 * 72);
    encode_image_output_fpn_pos_2_tensor_.gpu(1 * 256 * 72 * 72);

    // Initialize Text Encoder tensors
    encode_text_input_ids_tensor_.cpu(1 * 32);
    encode_text_input_ids_tensor_.gpu(1 * 32);
    encode_text_input_attention_mask_tensor_.cpu(1 * 32);
    encode_text_input_attention_mask_tensor_.gpu(1 * 32);

    encode_text_output_text_features_tensor_.cpu(1 * 32 * 256);
    encode_text_output_text_features_tensor_.gpu(1 * 32 * 256);
    encode_text_output_text_mask_tensor_.cpu(1 * 32);
    encode_text_output_text_mask_tensor_.gpu(1 * 32);

    // decode output tensors
    /**
     *  pred_masks : {-1 x 200 x 288 x 288} [float32]
        pred_boxes : {-1 x 200 x 4} [float32]
        pred_logits : {-1 x 200} [float32]
        presence_logits : {-1 x 1} [float32]
     */
    decode_output_pred_masks_tensor_.cpu(1 * 200 * 288 * 288);
    decode_output_pred_masks_tensor_.gpu(1 * 200 * 288 * 288);
    decode_output_pred_boxes_tensor_.cpu(1 * 200 * 4);
    decode_output_pred_boxes_tensor_.gpu(1 * 200 * 4);
    decode_output_pred_logits_tensor_.cpu(1 * 200);
    decode_output_pred_logits_tensor_.gpu(1 * 200);
    decode_output_presence_logits_tensor_.cpu(1 * 1);
    decode_output_presence_logits_tensor_.gpu(1 * 1);

    // 存储中间过滤结果的tensor
    filter_boxes_tensor_.cpu(num_queries_ * 4);
    filter_boxes_tensor_.gpu(num_queries_ * 4);
    filter_scores_tensor_.cpu(num_queries_);
    filter_scores_tensor_.gpu(num_queries_);
    filter_indices_tensor_.cpu(num_queries_);
    filter_indices_tensor_.gpu(num_queries_);

    box_count_.cpu(1);
    box_count_.gpu(1);

    confidence_threshold_ = confidence_threshold;
    vision_encoder_path_ = vision_encoder_path;
    text_encoder_path_ = text_encoder_path;
    decoder_path_ = decoder_path;
    gpu_id_ = gpu_id;
}

bool Sam3Infer::load_engines()
{
    // Load TensorRT engines for vision encoder, text encoder, and decoder
    // Implementation details would go here
    vision_encoder_trt_ = TensorRT::load(vision_encoder_path_);
    if (!vision_encoder_trt_)
    {
        std::cerr << "Failed to load vision encoder TensorRT engine from " << vision_encoder_path_ << std::endl;
        return false;
    }
    vision_encoder_trt_->print();
    text_encoder_trt_ = TensorRT::load(text_encoder_path_);
    if (!text_encoder_trt_)
    {
        std::cerr << "Failed to load text encoder TensorRT engine from " << text_encoder_path_ << std::endl;
        return false;
    }
    text_encoder_trt_->print();
    decoder_trt_ = TensorRT::load(decoder_path_);
    if (!decoder_trt_)
    {
        std::cerr << "Failed to load decoder TensorRT engine from " << decoder_path_ << std::endl;
        return false;
    }
    decoder_trt_->print();
    return true;
}

void Sam3Infer::setup_text_inputs(const std::string &input_text, const std::array<int64_t, 32> &input_ids, const std::array<int64_t, 32> &attention_mask)
{
    text_input_map_[input_text] = std::make_pair(input_ids, attention_mask);
}

void Sam3Infer::preprocess(const cv::Mat &input_image, void *stream)
{
    tensor::Image img_tensor = tensor::cvimg(input_image);
    original_image_height_ = img_tensor.height;
    original_image_width_ = img_tensor.width;
    affine::ResizeMatrix matrix;
    matrix.compute(
        std::make_tuple(original_image_width_, original_image_height_),
        std::make_tuple(input_image_width_, input_image_height_));

    size_t size_image = original_image_width_ * original_image_height_ * 3;
    uint8_t *original_image_host = original_image_tensor_.cpu(size_image);
    uint8_t *original_image_device = original_image_tensor_.gpu(size_image);

    float *input_device = preprocess_image_tensor_.gpu();

    float *affine_matrix_host = affine_matrix_tensor_.cpu();
    float *affine_matrix_device = affine_matrix_tensor_.gpu();

    cudaStream_t stream_ = (cudaStream_t)stream;
    // memory copy
    memcpy(original_image_host, img_tensor.bgrptr, size_image);
    memcpy(affine_matrix_host, matrix.d2i, sizeof(matrix.d2i));

    // cuda memcpy to device
    cudaMemcpyAsync(original_image_device, original_image_host, size_image, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(matrix.d2i), cudaMemcpyHostToDevice, stream_);

    warp_affine_bilinear_and_normalize_plane(original_image_device,
                                             img_tensor.width * 3,
                                             img_tensor.width,
                                             img_tensor.height,
                                             input_device,
                                             input_image_width_,
                                             input_image_height_,
                                             affine_matrix_device,
                                             114,
                                             preprocess_norm_,
                                             stream_);
}

bool Sam3Infer::encode_image(void *stream)
{
    auto input_dims = vision_encoder_trt_->static_dims(0);
    input_dims[0] = 1;
    if (!vision_encoder_trt_->set_run_dims(0, input_dims))
    {
        printf("Fail to set run dims\n");
        return false;
    }
    cudaStream_t stream_ = (cudaStream_t)stream;
    std::unordered_map<std::string, const void *> bindings = {
        {"images", preprocess_image_tensor_.gpu()},
        {"fpn_feat_0", encode_image_output_fpn_feat_0_tensor_.gpu()},
        {"fpn_feat_1", encode_image_output_fpn_feat_1_tensor_.gpu()},
        {"fpn_feat_2", encode_image_output_fpn_feat_2_tensor_.gpu()},
        {"fpn_pos_2", encode_image_output_fpn_pos_2_tensor_.gpu()}};
    if (!vision_encoder_trt_->forward(bindings, stream_))
    {
        printf("[ENCODE IMAGE] Failed to tensorRT forward.\n");
        return false;
    }
    return true;
}

bool Sam3Infer::encode_text(const std::string &input_text, void *stream)
{
    // text_input_map_
    if (text_input_map_.find(input_text) == text_input_map_.end())
    {
        printf("[ENCODE TEXT] Input text not found in text input map.\n");
        return false;
    }

    auto input_dims_1 = text_encoder_trt_->static_dims(0);
    input_dims_1[0] = 1;
    if (!text_encoder_trt_->set_run_dims(0, input_dims_1))
    {
        printf("Fail to set run dims\n");
        return false;
    }
    auto input_dims_2 = text_encoder_trt_->static_dims(1);
    input_dims_2[0] = 1;
    if (!text_encoder_trt_->set_run_dims(1, input_dims_2))
    {
        printf("Fail to set run dims\n");
        return false;
    }
    const auto &text_pair = text_input_map_[input_text];
    // 设置输入tensor
    memcpy(encode_text_input_ids_tensor_.cpu(), text_pair.first.data(), sizeof(int64_t) * text_pair.first.size());
    memcpy(encode_text_input_attention_mask_tensor_.cpu(), text_pair.second.data(), sizeof(int64_t) * text_pair.second.size());

    cudaStream_t stream_ = (cudaStream_t)stream;
    cudaMemcpyAsync(encode_text_input_ids_tensor_.gpu(),
                    encode_text_input_ids_tensor_.cpu(),
                    sizeof(int64_t) * text_pair.first.size(),
                    cudaMemcpyHostToDevice,
                    stream_);
    cudaMemcpyAsync(encode_text_input_attention_mask_tensor_.gpu(),
                    encode_text_input_attention_mask_tensor_.cpu(),
                    sizeof(int64_t) * text_pair.second.size(),
                    cudaMemcpyHostToDevice,
                    stream_);
    // Implementation for encoding text would go here
    std::unordered_map<std::string, const void *> bindings = {
        {"input_ids", encode_text_input_ids_tensor_.gpu()},
        {"attention_mask", encode_text_input_attention_mask_tensor_.gpu()},
        {"text_features", encode_text_output_text_features_tensor_.gpu()},
        {"text_mask", encode_text_output_text_mask_tensor_.gpu()}};
    if (!text_encoder_trt_->forward(bindings, stream_))
    {
        printf("[ENCODE TEXT] Failed to tensorRT forward.\n");
        return false;
    }
    return true;
}

bool Sam3Infer::decode(void *stream)
{
    for (int i = 0; i < 6; ++i)
    {
        auto input_dims = decoder_trt_->static_dims(i);
        input_dims[0] = 1;
        if (i == 4 || i == 5)
        {
            input_dims[1] = 32;
        }
        if (!decoder_trt_->set_run_dims(i, input_dims))
        {
            printf("Fail to set run dims\n");
            return false;
        }
    }

    cudaStream_t stream_ = (cudaStream_t)stream;
    std::unordered_map<std::string, const void *> bindings = {
        {"fpn_feat_0", encode_image_output_fpn_feat_0_tensor_.gpu()},
        {"fpn_feat_1", encode_image_output_fpn_feat_1_tensor_.gpu()},
        {"fpn_feat_2", encode_image_output_fpn_feat_2_tensor_.gpu()},
        {"fpn_pos_2", encode_image_output_fpn_pos_2_tensor_.gpu()},
        {"prompt_features", encode_text_output_text_features_tensor_.gpu()},
        {"prompt_mask", encode_text_output_text_mask_tensor_.gpu()},
        {"pred_masks", decode_output_pred_masks_tensor_.gpu()},
        {"pred_boxes", decode_output_pred_boxes_tensor_.gpu()},
        {"pred_logits", decode_output_pred_logits_tensor_.gpu()},
        {"presence_logits", decode_output_presence_logits_tensor_.gpu()}};
    if (!decoder_trt_->forward(bindings, stream_))
    {
        printf("[DECODE] Failed to tensorRT forward.\n");
        return false;
    }
    return true;
}

// void Sam3Infer::postprocess(InferResult &result, void *stream)
// {
//     cudaStream_t stream_ = (cudaStream_t)stream;

//     float *pred_masks_host = decode_output_pred_masks_tensor_.cpu();
//     float *pred_boxes_host = decode_output_pred_boxes_tensor_.cpu();
//     float *pred_logits_host = decode_output_pred_logits_tensor_.cpu();
//     float *presence_logits_host = decode_output_presence_logits_tensor_.cpu();

//     cudaMemcpyAsync(pred_masks_host,
//                     decode_output_pred_masks_tensor_.gpu(),
//                     decode_output_pred_masks_tensor_.cpu_bytes(),
//                     cudaMemcpyDeviceToHost,
//                     stream_);
//     cudaMemcpyAsync(pred_boxes_host,
//                     decode_output_pred_boxes_tensor_.gpu(),
//                     decode_output_pred_boxes_tensor_.cpu_bytes(),
//                     cudaMemcpyDeviceToHost,
//                     stream_);
//     cudaMemcpyAsync(pred_logits_host,
//                     decode_output_pred_logits_tensor_.gpu(),
//                     decode_output_pred_logits_tensor_.cpu_bytes(),
//                     cudaMemcpyDeviceToHost,
//                     stream_);
//     cudaMemcpyAsync(presence_logits_host,
//                     decode_output_presence_logits_tensor_.gpu(),
//                     decode_output_presence_logits_tensor_.cpu_bytes(),
//                     cudaMemcpyDeviceToHost,
//                     stream_);

//     cudaStreamSynchronize(stream_);

//     const int MASK_H = mask_height_;
//     const int MASK_W = mask_width_;
//     const float conf_threshold = confidence_threshold_;

//     int orig_w = original_image_width_;
//     int orig_h = original_image_height_;

//     // 获取 Query 数量 (假设 pred_logits 的 shape 是 [Batch=1, Num_Queries])
//     // 你需要根据 Tensor 类的 API 获取维度，这里假设是 result.num_queries 或者固定值
//     int num_queries = num_queries_;

//     // Helper: Sigmoid 函数
//     auto sigmoid = [](float x)
//     { return 1.0f / (1.0f + std::exp(-x)); };

//     // A. 计算 Presence Score (全局是否存在物体的概率)
//     float presence_score = sigmoid(presence_logits_host[0]);

//     // B. 遍历所有预测结果 (Iterate Loop)
//     for (int i = 0; i < num_queries; ++i)
//     {
//         // Python: scores = (1 / (1 + np.exp(-pred_logits))) * presence_score
//         float raw_logit = pred_logits_host[i];
//         float score = sigmoid(raw_logit) * presence_score;
//         if (score > conf_threshold)
//         {
//             float *box_ptr = pred_boxes_host + (i * 4);

//             float x1 = box_ptr[0] * orig_w;
//             float y1 = box_ptr[1] * orig_h;
//             float x2 = box_ptr[2] * orig_w;
//             float y2 = box_ptr[3] * orig_h;

//             x1 = std::max(0.0f, std::min(x1, (float)orig_w));
//             y1 = std::max(0.0f, std::min(y1, (float)orig_h));
//             x2 = std::max(0.0f, std::min(x2, (float)orig_w));
//             y2 = std::max(0.0f, std::min(y2, (float)orig_h));

//             // --- 处理 Mask ---
//             // Python: mask_resized = cv2.resize(m, (w, h)) ... > 0

//             // 找到当前 mask 的数据起始位置
//             float *mask_data_ptr = pred_masks_host + (i * MASK_H * MASK_W);

//             // 创建 cv::Mat 包装数据 (不拷贝，直接用 float 指针)
//             cv::Mat raw_mask(MASK_H, MASK_W, CV_32F, mask_data_ptr);

//             // 缩放 Mask
//             cv::Mat resized_mask;
//             // INTER_LINEAR 对应 python 的 cv2.INTER_LINEAR
//             cv::resize(raw_mask, resized_mask, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);

//             // 二值化 (Python: > 0)
//             cv::Mat binary_mask;
//             // 大于 0 的置为 255，否则为 0
//             cv::threshold(resized_mask, binary_mask, 0.0, 255, cv::THRESH_BINARY);

//             // 转换为 8位单通道 (CV_8U) 方便存储和显示
//             binary_mask.convertTo(binary_mask, CV_8U);

//             auto box = object::createSegmentationBox(x1, y1, x2, y2, binary_mask, score, -1, "person");
//             // auto box = object::createBox(x1, y1, x2, y2, score, -1, "person");
//             result.push_back(box);
//         }
//     }
// }

void Sam3Infer::postprocess(InferResult &result, const std::string &label, void *stream)
{
    cudaStream_t stream_ = (cudaStream_t)stream;
    cudaMemsetAsync(box_count_.gpu(), 0, box_count_.cpu_bytes(), stream_);
    sam3_postprocess_plane(
        decode_output_pred_masks_tensor_.gpu(),
        decode_output_pred_boxes_tensor_.gpu(),
        decode_output_pred_logits_tensor_.gpu(),
        decode_output_presence_logits_tensor_.gpu(),
        filter_boxes_tensor_.gpu(),
        filter_indices_tensor_.gpu(),
        filter_scores_tensor_.gpu(),
        box_count_.gpu(),
        num_queries_,
        mask_height_,
        mask_width_,
        original_image_width_,
        original_image_height_,
        confidence_threshold_,
        stream_);

    cudaMemcpyAsync(
        filter_boxes_tensor_.cpu(),
        filter_boxes_tensor_.gpu(),
        filter_boxes_tensor_.cpu_bytes(),
        cudaMemcpyDeviceToHost,
        stream_);
    cudaMemcpyAsync(
        filter_indices_tensor_.cpu(),
        filter_indices_tensor_.gpu(),
        filter_indices_tensor_.cpu_bytes(),
        cudaMemcpyDeviceToHost,
        stream_);
    cudaMemcpyAsync(
        filter_scores_tensor_.cpu(),
        filter_scores_tensor_.gpu(),
        filter_scores_tensor_.cpu_bytes(),
        cudaMemcpyDeviceToHost,
        stream_);
    cudaMemcpyAsync(
        box_count_.cpu(),
        box_count_.gpu(),
        box_count_.cpu_bytes(),
        cudaMemcpyDeviceToHost,
        stream_);
    cudaStreamSynchronize(stream_);
    mask_buffer_.gpu(*box_count_.cpu() * original_image_height_ * original_image_width_);
    mask_buffer_.cpu(*box_count_.cpu() * original_image_height_ * original_image_width_);

    affine::ResizeMatrix matrix;
    matrix.compute(
        std::make_tuple(mask_width_, mask_height_),
        std::make_tuple(original_image_width_, original_image_height_));
    // 计算 mask 的仿射矩阵
    float *mask_affine_matrix_host = mask_affine_matrix_tensor_.cpu();
    float *mask_affine_matrix_device = mask_affine_matrix_tensor_.gpu();
    memcpy(mask_affine_matrix_host, matrix.d2i, sizeof(matrix.d2i));
    cudaMemcpyAsync(mask_affine_matrix_device, mask_affine_matrix_host, sizeof(matrix.d2i), cudaMemcpyHostToDevice, stream_);

    for (int i = 0; i < *box_count_.cpu(); ++i)
    {
        int idx = filter_indices_tensor_.cpu()[i];

        // 处理 Mask
        float *mask_data_ptr = decode_output_pred_masks_tensor_.gpu() + (idx * mask_height_ * mask_width_);

        // 使用 CUDA 内核进行缩放和二值化
        warp_affine_bilinear_single_channel_mask_plane(
            mask_data_ptr,
            mask_width_,
            mask_width_,
            mask_height_,
            mask_buffer_.gpu() + i * original_image_height_ * original_image_width_,
            original_image_width_,
            original_image_height_,
            mask_affine_matrix_device,
            0,
            stream_);
    }
    cudaMemcpyAsync(
        mask_buffer_.cpu(),
        mask_buffer_.gpu(),
        mask_buffer_.cpu_bytes(),
        cudaMemcpyDeviceToHost,
        stream_);
    cudaStreamSynchronize(stream_);
    for (int i = 0; i < *box_count_.cpu(); ++i)
    {
        float score = filter_scores_tensor_.cpu()[i];
        float *box_ptr = filter_boxes_tensor_.cpu() + i * 4;

        float x1 = box_ptr[0];
        float y1 = box_ptr[1];
        float x2 = box_ptr[2];
        float y2 = box_ptr[3];
        cv::Mat binary_mask(original_image_height_, original_image_width_, CV_8U, mask_buffer_.cpu() + i * original_image_height_ * original_image_width_);
        auto box = object::createSegmentationBox(x1, y1, x2, y2, binary_mask, score, -1, label);
        result.push_back(box);
    }
}

InferResult Sam3Infer::forward(const cv::Mat &input_image, const std::string &input_text, void *stream)
{
    // Preprocess the input image
    preprocess(input_image);

    // Encode the image
    if (!encode_image(stream))
    {
        printf("[FORWARD] Image encoding failed.\n");
        return {};
    }

    // Encode the text
    if (!encode_text(input_text, stream))
    {
        printf("[FORWARD] Text encoding failed.\n");
        return {};
    }

    // Decode to get the final output
    if (!decode(stream))
    {
        printf("[FORWARD] Decoding failed.\n");
        return {};
    }

    // Postprocess and prepare the InferResult
    InferResult result;
    postprocess(result, input_text, stream);
    return result;
}