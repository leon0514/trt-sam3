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
    AutoDevice device_guard(gpu_id);
    // 初始化分配内存/显存
    confidence_threshold_ = confidence_threshold;
    vision_encoder_path_ = vision_encoder_path;
    text_encoder_path_ = text_encoder_path;
    decoder_path_ = decoder_path;
    gpu_id_ = gpu_id;
}

Sam3Infer::Sam3Infer(
    const std::string &vision_encoder_path,
    const std::string &text_encoder_path,
    const std::string &geometry_encoder_path,
    const std::string &decoder_path,
    int gpu_id,
    float confidence_threshold)
    : InferBase()
{
    AutoDevice device_guard(gpu_id);
    // 初始化分配内存/显存
    confidence_threshold_ = confidence_threshold;
    vision_encoder_path_ = vision_encoder_path;
    text_encoder_path_ = text_encoder_path;
    decoder_path_ = decoder_path;
    geometry_encoder_path_ = geometry_encoder_path;
    gpu_id_ = gpu_id;
}

bool Sam3Infer::load_engines()
{
    AutoDevice device_guard(gpu_id_);
    // Load TensorRT engines for vision encoder, text encoder, and decoder
    // Implementation details would go here

    if (!geometry_encoder_path_.empty())
    {
        geometry_encoder_trt_ = TensorRT::load(geometry_encoder_path_);
        if (!geometry_encoder_trt_)
        {
            std::cerr << "Failed to load geometry encoder TensorRT engine from " << geometry_encoder_path_ << std::endl;
            return false;
        }
        isdynamic_model_ = geometry_encoder_trt_->has_dynamic_dim() && isdynamic_model_;
        geometry_encoder_trt_->print(geometry_encoder_path_.c_str());

        geometry_encoder_input_input_boxes_shape_ = geometry_encoder_trt_->static_dims(0);
        geometry_encoder_input_input_boxes_labels_shape_ = geometry_encoder_trt_->static_dims(1);
        geometry_encoder_input_fpn_feat_2_shape_ = geometry_encoder_trt_->static_dims(2);
        geometry_encoder_input_fpn_pos_2_shape_ = geometry_encoder_trt_->static_dims(3);
        geometry_encoder_output_geometry_features_shape_ = geometry_encoder_trt_->static_dims(4);
        geometry_encoder_output_geometry_mask_shape_ = geometry_encoder_trt_->static_dims(5);
    }

    vision_encoder_trt_ = TensorRT::load(vision_encoder_path_);
    if (!vision_encoder_trt_)
    {
        std::cerr << "Failed to load vision encoder TensorRT engine from " << vision_encoder_path_ << std::endl;
        return false;
    }
    isdynamic_model_ = vision_encoder_trt_->has_dynamic_dim() && isdynamic_model_;
    vision_encoder_trt_->print(vision_encoder_path_.c_str());

    vision_encoder_input_images_shape_ = vision_encoder_trt_->static_dims(0);
    vision_encoder_output_fpn_feat_0_shape_ = vision_encoder_trt_->static_dims(1);
    vision_encoder_output_fpn_feat_1_shape_ = vision_encoder_trt_->static_dims(2);
    vision_encoder_output_fpn_feat_2_shape_ = vision_encoder_trt_->static_dims(3);
    vision_encoder_output_fpn_pos_2_shape_ = vision_encoder_trt_->static_dims(4);

    input_image_height_ = vision_encoder_input_images_shape_[2];
    input_image_width_ = vision_encoder_input_images_shape_[3];

    text_encoder_trt_ = TensorRT::load(text_encoder_path_);
    if (!text_encoder_trt_)
    {
        std::cerr << "Failed to load text encoder TensorRT engine from " << text_encoder_path_ << std::endl;
        return false;
    }
    isdynamic_model_ = text_encoder_trt_->has_dynamic_dim() && isdynamic_model_;
    text_encoder_trt_->print(text_encoder_path_.c_str());

    text_encoder_input_input_ids_shape_ = text_encoder_trt_->static_dims(0);
    text_encoder_input_attention_mask_shape_ = text_encoder_trt_->static_dims(1);
    text_encoder_output_text_features_shape_ = text_encoder_trt_->static_dims(2);
    text_encoder_output_text_mask_shape_ = text_encoder_trt_->static_dims(3);

    decoder_trt_ = TensorRT::load(decoder_path_);
    if (!decoder_trt_)
    {
        std::cerr << "Failed to load decoder TensorRT engine from " << decoder_path_ << std::endl;
        return false;
    }
    isdynamic_model_ = decoder_trt_->has_dynamic_dim() && isdynamic_model_;
    decoder_trt_->print(decoder_path_.c_str());

    decoder_input_fpn_feat_0_shape_ = decoder_trt_->static_dims(0);
    decoder_input_fpn_feat_1_shape_ = decoder_trt_->static_dims(1);
    decoder_input_fpn_feat_2_shape_ = decoder_trt_->static_dims(2);
    decoder_input_fpn_pos_2_shape_ = decoder_trt_->static_dims(3);
    decoder_input_prompt_features_shape_ = decoder_trt_->static_dims(4);
    decoder_input_prompt_mask_shape_ = decoder_trt_->static_dims(5);
    decoder_output_pred_masks_shape_ = decoder_trt_->static_dims(6);
    decoder_output_pred_boxes_shape_ = decoder_trt_->static_dims(7);
    decoder_output_pred_logits_shape_ = decoder_trt_->static_dims(8);
    decoder_output_presence_logits_shape_ = decoder_trt_->static_dims(9);

    num_queries_ = decoder_output_pred_boxes_shape_[1];
    mask_width_ = decoder_output_pred_masks_shape_[2];
    mask_height_ = decoder_output_pred_masks_shape_[3];
    return true;
}

void Sam3Infer::adjust_memory(int batch_size)
{
    AutoDevice device_guard(gpu_id_);

    affine_matrix_tensor_.cpu(batch_size * 2 * 3);
    affine_matrix_tensor_.gpu(batch_size * 2 * 3);
    mask_affine_matrix_tensor_.cpu(batch_size * 2 * 3);
    mask_affine_matrix_tensor_.gpu(batch_size * 2 * 3);

    if ((int)original_images_tensor_.size() < batch_size)
    {
        for (int i = original_images_tensor_.size(); i < batch_size; ++i)
        {
            original_images_tensor_.emplace_back(std::make_shared<tensor::Memory<uint8_t>>());
        }
    }
    preprocessed_images_tensor_.cpu(batch_size * 3 * input_image_height_ * input_image_width_);
    preprocessed_images_tensor_.gpu(batch_size * 3 * input_image_height_ * input_image_width_);

    // initialize vision encoder tensors
    size_t vision_encoder_input_images_size = batch_size * 3 * input_image_height_ * input_image_width_;
    vision_encoder_input_images_tensor_.cpu(vision_encoder_input_images_size);
    vision_encoder_input_images_tensor_.gpu(vision_encoder_input_images_size);

    size_t vision_encoder_output_fpn_feat_0_size = batch_size *
                                                   vision_encoder_output_fpn_feat_0_shape_[1] *
                                                   vision_encoder_output_fpn_feat_0_shape_[2] *
                                                   vision_encoder_output_fpn_feat_0_shape_[3];
    vision_encoder_output_fpn_feat_0_tensor_.cpu(vision_encoder_output_fpn_feat_0_size);
    vision_encoder_output_fpn_feat_0_tensor_.gpu(vision_encoder_output_fpn_feat_0_size);

    size_t vision_encoder_output_fpn_feat_1_size = batch_size *
                                                   vision_encoder_output_fpn_feat_1_shape_[1] *
                                                   vision_encoder_output_fpn_feat_1_shape_[2] *
                                                   vision_encoder_output_fpn_feat_1_shape_[3];
    vision_encoder_output_fpn_feat_1_tensor_.cpu(vision_encoder_output_fpn_feat_1_size);
    vision_encoder_output_fpn_feat_1_tensor_.gpu(vision_encoder_output_fpn_feat_1_size);

    size_t vision_encoder_output_fpn_feat_2_size = batch_size *
                                                   vision_encoder_output_fpn_feat_2_shape_[1] *
                                                   vision_encoder_output_fpn_feat_2_shape_[2] *
                                                   vision_encoder_output_fpn_feat_2_shape_[3];
    vision_encoder_output_fpn_feat_2_tensor_.cpu(vision_encoder_output_fpn_feat_2_size);
    vision_encoder_output_fpn_feat_2_tensor_.gpu(vision_encoder_output_fpn_feat_2_size);

    size_t vision_encoder_output_fpn_pos_2_size = batch_size *
                                                  vision_encoder_output_fpn_pos_2_shape_[1] *
                                                  vision_encoder_output_fpn_pos_2_shape_[2] *
                                                  vision_encoder_output_fpn_pos_2_shape_[3];
    vision_encoder_output_fpn_pos_2_tensor_.cpu(vision_encoder_output_fpn_pos_2_size);
    vision_encoder_output_fpn_pos_2_tensor_.gpu(vision_encoder_output_fpn_pos_2_size);

    // initialize text encoder tensors
    size_t text_encoder_input_input_ids_size = batch_size * text_encoder_input_input_ids_shape_[1];
    text_encoder_input_input_ids_tensor_.cpu(text_encoder_input_input_ids_size);
    text_encoder_input_input_ids_tensor_.gpu(text_encoder_input_input_ids_size);

    size_t text_encoder_input_attention_mask_size = batch_size * text_encoder_input_attention_mask_shape_[1];
    text_encoder_input_attention_mask_tensor_.cpu(text_encoder_input_attention_mask_size);
    text_encoder_input_attention_mask_tensor_.gpu(text_encoder_input_attention_mask_size);

    size_t text_encoder_output_text_features_size = batch_size *
                                                    text_encoder_output_text_features_shape_[1] *
                                                    text_encoder_output_text_features_shape_[2];
    text_encoder_output_text_features_tensor_.cpu(text_encoder_output_text_features_size);
    text_encoder_output_text_features_tensor_.gpu(text_encoder_output_text_features_size);

    size_t text_encoder_output_text_mask_size = batch_size * text_encoder_output_text_mask_shape_[1];
    text_encoder_output_text_mask_tensor_.cpu(text_encoder_output_text_mask_size);
    text_encoder_output_text_mask_tensor_.gpu(text_encoder_output_text_mask_size);

    size_t decoder_output_pred_masks_size = batch_size *
                                            decoder_output_pred_masks_shape_[1] *
                                            decoder_output_pred_masks_shape_[2] *
                                            decoder_output_pred_masks_shape_[3];
    decoder_output_pred_masks_tensor_.cpu(decoder_output_pred_masks_size);
    decoder_output_pred_masks_tensor_.gpu(decoder_output_pred_masks_size);

    size_t decoder_output_pred_boxes_size = batch_size *
                                            decoder_output_pred_boxes_shape_[1] *
                                            decoder_output_pred_boxes_shape_[2];
    decoder_output_pred_boxes_tensor_.cpu(decoder_output_pred_boxes_size);
    decoder_output_pred_boxes_tensor_.gpu(decoder_output_pred_boxes_size);

    size_t decoder_output_pred_logits_size = batch_size *
                                             decoder_output_pred_logits_shape_[1];
    decoder_output_pred_logits_tensor_.cpu(decoder_output_pred_logits_size);
    decoder_output_pred_logits_tensor_.gpu(decoder_output_pred_logits_size);

    size_t decoder_output_presence_logits_size = batch_size *
                                                 decoder_output_presence_logits_shape_[1];
    decoder_output_presence_logits_tensor_.cpu(decoder_output_presence_logits_size);
    decoder_output_presence_logits_tensor_.gpu(decoder_output_presence_logits_size);

    bool allocate_geometry_encoder_memory = geometry_encoder_input_input_boxes_shape_[1] > 0 && geometry_encoder_input_input_boxes_labels_shape_[1] > 0;

    if (allocate_geometry_encoder_memory)
    {
        size_t geometry_encoder_input_input_boxes_size = batch_size * geometry_encoder_input_input_boxes_shape_[1] * 4;
        geometry_encoder_input_input_boxes_tensor_.cpu(geometry_encoder_input_input_boxes_size);
        geometry_encoder_input_input_boxes_tensor_.gpu(geometry_encoder_input_input_boxes_size);

        size_t geometry_encoder_input_input_boxes_labels_size = batch_size * geometry_encoder_input_input_boxes_labels_shape_[1];
        geometry_encoder_input_input_boxes_labels_tensor_.cpu(geometry_encoder_input_input_boxes_labels_size);
        geometry_encoder_input_input_boxes_labels_tensor_.gpu(geometry_encoder_input_input_boxes_labels_size);

        // size_t geometry_encoder_input_fpn_feat_2_size = batch_size *
        //                                                 geometry_encoder_input_fpn_feat_2_shape_[1] *
        //                                                 geometry_encoder_input_fpn_feat_2_shape_[2] *
        //                                                 geometry_encoder_input_fpn_feat_2_shape_[3];
        // geometry_encoder_input_fpn_feat_2_tensor_.cpu(geometry_encoder_input_fpn_feat_2_size);
        // geometry_encoder_input_fpn_feat_2_tensor_.gpu(geometry_encoder_input_fpn_feat_2_size);

        // size_t geometry_encoder_input_fpn_pos_2_size = batch_size *
        //                                             geometry_encoder_input_fpn_pos_2_shape_[1] *
        //                                             geometry_encoder_input_fpn_pos_2_shape_[2] *
        //                                             geometry_encoder_input_fpn_pos_2_shape_[3];
        // geometry_encoder_input_fpn_pos_2_tensor_.cpu(geometry_encoder_input_fpn_pos_2_size);
        // geometry_encoder_input_fpn_pos_2_tensor_.gpu(geometry_encoder_input_fpn_pos_2_size);
        geometry_encoder_input_fpn_feat_2_tensor_ = vision_encoder_output_fpn_feat_2_tensor_;
        geometry_encoder_input_fpn_pos_2_tensor_ = vision_encoder_output_fpn_pos_2_tensor_;

        size_t geometry_encoder_output_geometry_features_size = batch_size *
                                                                 geometry_encoder_output_geometry_features_shape_[1] *
                                                                 geometry_encoder_output_geometry_features_shape_[2];
        geometry_encoder_output_geometry_features_tensor_.cpu(geometry_encoder_output_geometry_features_size);
        geometry_encoder_output_geometry_features_tensor_.gpu(geometry_encoder_output_geometry_features_size);

        size_t geometry_encoder_output_geometry_mask_size = batch_size *
                                                             geometry_encoder_output_geometry_mask_shape_[1];
        geometry_encoder_output_geometry_mask_tensor_.cpu(geometry_encoder_output_geometry_mask_size);
        geometry_encoder_output_geometry_mask_tensor_.gpu(geometry_encoder_output_geometry_mask_size);
    }

    // initialize decoder tensors

    decoder_input_fpn_feat_0_tensor_ = vision_encoder_output_fpn_feat_0_tensor_;
    decoder_input_fpn_feat_1_tensor_ = vision_encoder_output_fpn_feat_1_tensor_;
    decoder_input_fpn_feat_2_tensor_ = vision_encoder_output_fpn_feat_2_tensor_;
    decoder_input_fpn_pos_2_tensor_ = vision_encoder_output_fpn_pos_2_tensor_;
    // 后续加入Geometry时需要修改
    if (allocate_geometry_encoder_memory)
    {
        size_t decoder_input_prompt_features_tensor_size = batch_size *
                                                    (text_encoder_input_input_ids_shape_[1] + geometry_encoder_output_geometry_features_shape_[1]) *
                                                    decoder_input_prompt_features_shape_[2];
        decoder_input_prompt_features_tensor_.cpu(decoder_input_prompt_features_tensor_size);
        decoder_input_prompt_features_tensor_.gpu(decoder_input_prompt_features_tensor_size);

        size_t decoder_input_prompt_mask_tensor_size = batch_size *
                                                (text_encoder_input_input_ids_shape_[1] + geometry_encoder_output_geometry_mask_shape_[1]);
        decoder_input_prompt_mask_tensor_.cpu(decoder_input_prompt_mask_tensor_size);
        decoder_input_prompt_mask_tensor_.gpu(decoder_input_prompt_mask_tensor_size);
    }
    else
    {
        decoder_input_prompt_features_tensor_ = text_encoder_output_text_features_tensor_;
        decoder_input_prompt_mask_tensor_ = text_encoder_output_text_mask_tensor_;
    }
    

    // init postprocess intermediate tensors
    filter_boxes_tensor_.cpu(1 * num_queries_ * 4);
    filter_boxes_tensor_.gpu(1 * num_queries_ * 4);
    filter_scores_tensor_.cpu(1 * num_queries_);
    filter_scores_tensor_.gpu(1 * num_queries_);
    filter_indices_tensor_.cpu(1 * num_queries_);
    filter_indices_tensor_.gpu(1 * num_queries_);
    box_count_.cpu(1);
    box_count_.gpu(1);
}


void Sam3Infer::setup_text_inputs(const std::string &input_text, const std::array<int64_t, 32> &input_ids, const std::array<int64_t, 32> &attention_mask)
{
    text_input_map_[input_text] = std::make_pair(input_ids, attention_mask);
}

void Sam3Infer::preprocess(const cv::Mat &input_image, int ibatch, void *stream)
{
    tensor::Image img_tensor = tensor::cvimg(input_image);
    original_image_sizes_.emplace_back(img_tensor.width, img_tensor.height);
    affine::ResizeMatrix matrix;
    matrix.compute(
        std::make_tuple(original_image_sizes_[ibatch].first, original_image_sizes_[ibatch].second),
        std::make_tuple(input_image_width_, input_image_height_));

    size_t size_image = original_image_sizes_[ibatch].first * original_image_sizes_[ibatch].second * 3;
    uint8_t *original_image_host = original_images_tensor_[ibatch]->cpu(size_image);
    uint8_t *original_image_device = original_images_tensor_[ibatch]->gpu(size_image);

    float *input_device = preprocessed_images_tensor_.gpu() + ibatch * 3 * input_image_height_ * input_image_width_;

    float *affine_matrix_host = affine_matrix_tensor_.cpu() + ibatch * 6;
    float *affine_matrix_device = affine_matrix_tensor_.gpu() + ibatch * 6;

    cudaStream_t stream_ = (cudaStream_t)stream;
    if (input_image.isContinuous())
    {
        memcpy(original_image_host, input_image.data, size_image);
    }
    else
    {
        // 如果有 padding，必须逐行拷贝
        int width_bytes = original_image_sizes_[ibatch].first * 3;
        for (int h = 0; h < original_image_sizes_[ibatch].second; ++h)
        {
            const uint8_t *src_ptr = input_image.ptr<uint8_t>(h);
            uint8_t *dst_ptr = original_image_host + h * width_bytes;
            memcpy(dst_ptr, src_ptr, width_bytes);
        }
    }
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
    cudaStream_t stream_ = (cudaStream_t)stream;
    std::unordered_map<std::string, const void *> bindings = {
        {"images", preprocessed_images_tensor_.gpu()},
        {"fpn_feat_0", vision_encoder_output_fpn_feat_0_tensor_.gpu()},
        {"fpn_feat_1", vision_encoder_output_fpn_feat_1_tensor_.gpu()},
        {"fpn_feat_2", vision_encoder_output_fpn_feat_2_tensor_.gpu()},
        {"fpn_pos_2", vision_encoder_output_fpn_pos_2_tensor_.gpu()}};
    if (!vision_encoder_trt_->forward(bindings, stream_))
    {
        printf("[ENCODE IMAGE] Failed to tensorRT forward.\n");
        return false;
    }
    return true;
}

bool Sam3Infer::encode_boxes(const std::vector<BoxPrompt> &boxes, void *stream)
{
    int batch_size = geometry_encoder_input_input_boxes_tensor_.cpu_size() / geometry_encoder_input_input_boxes_shape_[1];

    float *host_boxes = geometry_encoder_input_input_boxes_tensor_.cpu();
    int64_t *host_labels = geometry_encoder_input_input_boxes_labels_tensor_.cpu();

    for (int ibatch = 0; ibatch < batch_size; ++ibatch)
    {
        int original_width = original_image_sizes_[ibatch].first;
        int original_height = original_image_sizes_[ibatch].second;
        std::vector<int64_t> labels;
        std::vector<std::array<float, 4>> box_values;
        for (const auto &pair : boxes)
        {
            if (pair.first == "pos")
            {
                labels.push_back(1);
            }
            else if (pair.first == "neg")
            {
                labels.push_back(0);
            }
            else
            {
                labels.push_back(1); // unknown
            }
            for (int i = 0; i < 4; ++i)
            {
                float x1 = pair.second[0];
                float y1 = pair.second[1];
                float x2 = pair.second[2];
                float y2 = pair.second[3];
                float cx = (x1 + x2) / 2.0f / original_width;
                float cy = (y1 + y2) / 2.0f / original_height;
                float w = (x2 - x1) / original_width;;
                float h = (y2 - y1) / original_height;
                box_values.push_back({cx, cy, w, h});
            }
        }
        memcpy(host_labels + ibatch * box_values.size(), labels.data(), sizeof(float) * labels.size());
        memcpy(host_boxes + ibatch * box_values.size() * 4, box_values.data(), sizeof(int64_t) * box_values.size() * 4);
    }

    cudaStream_t stream_ = (cudaStream_t)stream;

    // 拷贝整个 batch 的数据
    cudaMemcpyAsync(geometry_encoder_input_input_boxes_tensor_.gpu(),
                    geometry_encoder_input_input_boxes_tensor_.cpu(),
                    geometry_encoder_input_input_boxes_tensor_.gpu_bytes(), // 乘以 batch_size
                    cudaMemcpyHostToDevice,
                    stream_);

    cudaMemcpyAsync(geometry_encoder_input_input_boxes_labels_tensor_.gpu(),
                    geometry_encoder_input_input_boxes_labels_tensor_.cpu(),
                    geometry_encoder_input_input_boxes_labels_tensor_.gpu_bytes(), // 乘以 batch_size
                    cudaMemcpyHostToDevice,
                    stream_);
    std::unordered_map<std::string, const void *> bindings = {
        {"input_boxes", geometry_encoder_input_input_boxes_tensor_.gpu()},
        {"input_boxes_labels", geometry_encoder_input_input_boxes_labels_tensor_.gpu()},
        {"fpn_feat_2", geometry_encoder_input_fpn_feat_2_tensor_.gpu()},
        {"fpn_pos_2", geometry_encoder_input_fpn_pos_2_tensor_.gpu()},
        {"geometry_features", geometry_encoder_output_geometry_features_tensor_.gpu()},
        {"geometry_mask", geometry_encoder_output_geometry_mask_tensor_.gpu()}};
    if (!geometry_encoder_trt_->forward(bindings, stream_))
    {
        printf("[ENCODE BOXES] Failed to tensorRT forward.\n");
        return false;
    }
    return true;
}

bool Sam3Infer::encode_text(const std::string &input_text, void *stream)
{
    int batch_size = text_encoder_input_input_ids_tensor_.cpu_size() / text_encoder_input_input_ids_shape_[1];
    // text_input_map_
    std::pair<std::array<int64_t, 32>, std::array<int64_t, 32>> text_pair;
    if (input_text.empty())
    {
        // 1. 处理 input_ids：全部填充为 49407
        std::array<int64_t, 32> ids;
        ids.fill(49407); // 这样 ids 变成 [49407, 49407, ..., 49407]
        // 2. 处理 attention_mask：第一个为 1，其余为 0
        std::array<int64_t, 32> mask = {0}; // 先全部初始化为 0
        mask[0] = 1;                        // 将第一个设为 1
        text_pair = std::make_pair(ids, mask);
    }
    else if (text_input_map_.find(input_text) == text_input_map_.end())
    {
        printf("[ENCODE TEXT] Input text not found in text input map.\n");
        return false;
    }
    else
    {
        text_pair = text_input_map_[input_text];
    }

    // 获取单条数据的长度
    size_t ids_len = text_pair.first.size();
    size_t mask_len = text_pair.second.size();

    // 循环复制数据，填满整个 batch
    int64_t *host_ids = (int64_t *)text_encoder_input_input_ids_tensor_.cpu();
    int64_t *host_mask = (int64_t *)text_encoder_input_attention_mask_tensor_.cpu();

    for (int i = 0; i < batch_size; ++i)
    {
        memcpy(host_ids + i * ids_len, text_pair.first.data(), sizeof(int64_t) * ids_len);
        memcpy(host_mask + i * mask_len, text_pair.second.data(), sizeof(int64_t) * mask_len);
    }

    cudaStream_t stream_ = (cudaStream_t)stream;

    // 拷贝整个 batch 的数据
    cudaMemcpyAsync(text_encoder_input_input_ids_tensor_.gpu(),
                    text_encoder_input_input_ids_tensor_.cpu(),
                    text_encoder_input_input_ids_tensor_.gpu_bytes(), // 乘以 batch_size
                    cudaMemcpyHostToDevice,
                    stream_);

    cudaMemcpyAsync(text_encoder_input_attention_mask_tensor_.gpu(),
                    text_encoder_input_attention_mask_tensor_.cpu(),
                    text_encoder_input_attention_mask_tensor_.gpu_bytes(), // 乘以 batch_size
                    cudaMemcpyHostToDevice,
                    stream_);
    // Implementation for encoding text would go here
    std::unordered_map<std::string, const void *> bindings = {
        {"input_ids", text_encoder_input_input_ids_tensor_.gpu()},
        {"attention_mask", text_encoder_input_attention_mask_tensor_.gpu()},
        {"text_features", text_encoder_output_text_features_tensor_.gpu()},
        {"text_mask", text_encoder_output_text_mask_tensor_.gpu()}};
    if (!text_encoder_trt_->forward(bindings, stream_))
    {
        printf("[ENCODE TEXT] Failed to tensorRT forward.\n");
        return false;
    }
    return true;
}

bool Sam3Infer::decode(void *stream)
{
    cudaStream_t stream_ = (cudaStream_t)stream;
    std::unordered_map<std::string, const void *> bindings = {
        {"fpn_feat_0", decoder_input_fpn_feat_0_tensor_.gpu()},
        {"fpn_feat_1", decoder_input_fpn_feat_1_tensor_.gpu()},
        {"fpn_feat_2", decoder_input_fpn_feat_2_tensor_.gpu()},
        {"fpn_pos_2", decoder_input_fpn_pos_2_tensor_.gpu()},
        {"prompt_features", decoder_input_prompt_features_tensor_.gpu()},
        {"prompt_mask", decoder_input_prompt_mask_tensor_.gpu()},
        {"pred_masks", decoder_output_pred_masks_tensor_.gpu()},
        {"pred_boxes", decoder_output_pred_boxes_tensor_.gpu()},
        {"pred_logits", decoder_output_pred_logits_tensor_.gpu()},
        {"presence_logits", decoder_output_presence_logits_tensor_.gpu()}};
    if (!decoder_trt_->forward(bindings, stream_))
    {
        printf("[DECODE] Failed to tensorRT forward.\n");
        return false;
    }
    return true;
}

void Sam3Infer::postprocess(InferResult &result, int ibatch, const std::string &label, void *stream)
{
    cudaStream_t stream_ = (cudaStream_t)stream;
    cudaMemsetAsync(box_count_.gpu(), 0, box_count_.cpu_bytes(), stream_);
    sam3_postprocess_plane(
        decoder_output_pred_masks_tensor_.gpu() + ibatch * num_queries_ * mask_height_ * mask_width_,
        decoder_output_pred_boxes_tensor_.gpu() + ibatch * num_queries_ * 4,
        decoder_output_pred_logits_tensor_.gpu() + ibatch * num_queries_,
        decoder_output_presence_logits_tensor_.gpu() + ibatch,
        filter_boxes_tensor_.gpu(),
        filter_indices_tensor_.gpu(),
        filter_scores_tensor_.gpu(),
        box_count_.gpu(),
        num_queries_,
        mask_height_,
        mask_width_,
        original_image_sizes_[ibatch].first,
        original_image_sizes_[ibatch].second,
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
    printf("Batch %d: %d boxes after filtering.\n", ibatch, *box_count_.cpu());
    mask_buffer_.gpu(*box_count_.cpu() * original_image_sizes_[ibatch].second * original_image_sizes_[ibatch].first);
    mask_buffer_.cpu(*box_count_.cpu() * original_image_sizes_[ibatch].second * original_image_sizes_[ibatch].first);

    affine::ResizeMatrix matrix;
    matrix.compute(
        std::make_tuple(mask_width_, mask_height_),
        std::make_tuple(original_image_sizes_[ibatch].first, original_image_sizes_[ibatch].second));
    // 计算 mask 的仿射矩阵
    float *mask_affine_matrix_host = mask_affine_matrix_tensor_.cpu();
    float *mask_affine_matrix_device = mask_affine_matrix_tensor_.gpu();
    memcpy(mask_affine_matrix_host, matrix.d2i, sizeof(matrix.d2i));
    cudaMemcpyAsync(mask_affine_matrix_device, mask_affine_matrix_host, sizeof(matrix.d2i), cudaMemcpyHostToDevice, stream_);

    for (int i = 0; i < *box_count_.cpu(); ++i)
    {
        int idx = filter_indices_tensor_.cpu()[i];

        // 处理 Mask
        float *mask_data_ptr = decoder_output_pred_masks_tensor_.gpu() + (ibatch * num_queries_ * mask_height_ * mask_width_) + (idx * mask_height_ * mask_width_);

        // 使用 CUDA 内核进行缩放和二值化
        warp_affine_bilinear_single_channel_mask_plane(
            mask_data_ptr,
            mask_width_,
            mask_width_,
            mask_height_,
            mask_buffer_.gpu() + i * original_image_sizes_[ibatch].second * original_image_sizes_[ibatch].first,
            original_image_sizes_[ibatch].first,
            original_image_sizes_[ibatch].second,
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
        cv::Mat binary_mask(original_image_sizes_[ibatch].second, original_image_sizes_[ibatch].first, CV_8U, mask_buffer_.cpu() + i * original_image_sizes_[ibatch].second * original_image_sizes_[ibatch].first);
        auto box = object::createSegmentationBox(x1, y1, x2, y2, binary_mask, score, -1, label);
        result.push_back(box);
    }
}

InferResult Sam3Infer::forward(const cv::Mat &input_image, const std::string &input_text, void *stream)
{
    return forwards({input_image}, input_text, stream)[0];
}

InferResultArray Sam3Infer::forwards(const std::vector<cv::Mat> &input_images, const std::string &input_text, void *stream)
{
    original_image_sizes_.clear();
    int batch_size = static_cast<int>(input_images.size());
    int infer_bacth_size = vision_encoder_input_images_shape_[0];
    if (infer_bacth_size != batch_size)
    {
        if (isdynamic_model_)
        {
            // dynamic model 支持不同 batch size 推理
            vision_encoder_trt_->set_run_dims(0, {batch_size,
                                                  vision_encoder_input_images_shape_[1],
                                                  vision_encoder_input_images_shape_[2],
                                                  vision_encoder_input_images_shape_[3]});
            text_encoder_trt_->set_run_dims(0, {batch_size,
                                                text_encoder_input_input_ids_shape_[1]});
            text_encoder_trt_->set_run_dims(1, {batch_size,
                                                text_encoder_input_attention_mask_shape_[1]});
            decoder_trt_->set_run_dims(0, {batch_size,
                                           decoder_input_fpn_feat_0_shape_[1],
                                           decoder_input_fpn_feat_0_shape_[2],
                                           decoder_input_fpn_feat_0_shape_[3]});
            decoder_trt_->set_run_dims(1, {batch_size,
                                           decoder_input_fpn_feat_1_shape_[1],
                                           decoder_input_fpn_feat_1_shape_[2],
                                           decoder_input_fpn_feat_1_shape_[3]});
            decoder_trt_->set_run_dims(2, {batch_size,
                                           decoder_input_fpn_feat_2_shape_[1],
                                           decoder_input_fpn_feat_2_shape_[2],
                                           decoder_input_fpn_feat_2_shape_[3]});
            decoder_trt_->set_run_dims(3, {batch_size,
                                           decoder_input_fpn_pos_2_shape_[1],
                                           decoder_input_fpn_pos_2_shape_[2],
                                           decoder_input_fpn_pos_2_shape_[3]});
            // 后续加入Geometry时需要修改
            decoder_trt_->set_run_dims(4, {batch_size,
                                           text_encoder_input_input_ids_shape_[1],
                                           decoder_input_prompt_features_shape_[2]});
            decoder_trt_->set_run_dims(5, {batch_size,
                                           text_encoder_input_input_ids_shape_[1]});
        }
        else
        {
            printf("[FORWARDS] Static model batch size %d does not match input batch size %d.\n", infer_bacth_size, batch_size);
            return {};
        }
    }

    AutoDevice device_guard(gpu_id_);
    adjust_memory(batch_size);
    for (int ibatch = 0; ibatch < (int)input_images.size(); ++ibatch)
    {
        preprocess(input_images[ibatch], ibatch);
    }
    if (!encode_text(input_text, stream))
    {
        printf("[FORWARDS] Text encoding failed.\n");
    }
    if (!encode_image(stream))
    {
        printf("[FORWARDS] Image encoding failed.\n");
    }
    if (!decode(stream))
    {
        printf("[FORWARDS] Decoding failed.\n");
    }
    InferResultArray results;
    for (int ibatch = 0; ibatch < (int)input_images.size(); ++ibatch)
    {
        InferResult result;
        postprocess(result, ibatch, input_text, stream);
        results.push_back(result);
    }
    return results;
}

InferResult Sam3Infer::forward(const cv::Mat &input_image, const std::string &input_text, const std::vector<BoxPrompt> &boxes, void *stream)
{
    if (boxes.empty())
    {
        return forward(input_image, input_text, stream);
    }
    return forwards({input_image}, input_text, boxes, stream)[0];
}


InferResultArray Sam3Infer::forwards(const std::vector<cv::Mat> &input_images, const std::string &input_text, const std::vector<BoxPrompt> &boxes, void *stream)
{
    if (boxes.empty())
    {
        return forwards(input_images, input_text, stream);
    }
    /**
     * 后续加入 Geometry Encoder 支持
     */

    original_image_sizes_.clear();
    int batch_size = static_cast<int>(input_images.size());
    int infer_bacth_size = vision_encoder_input_images_shape_[0];

    int num_boxes = boxes.size();
    geometry_encoder_input_input_boxes_shape_[1] = num_boxes;
    geometry_encoder_input_input_boxes_labels_shape_[1] = num_boxes;
    geometry_encoder_output_geometry_features_shape_[1] = num_boxes+1;
    geometry_encoder_output_geometry_mask_shape_[1] = num_boxes+1;

    if (infer_bacth_size != batch_size)
    {
        if (isdynamic_model_)
        {
            // dynamic model 支持不同 batch size 推理
            vision_encoder_trt_->set_run_dims(0, {batch_size,
                                                  vision_encoder_input_images_shape_[1],
                                                  vision_encoder_input_images_shape_[2],
                                                  vision_encoder_input_images_shape_[3]});

            text_encoder_trt_->set_run_dims(0, {batch_size,
                                                text_encoder_input_input_ids_shape_[1]});
            text_encoder_trt_->set_run_dims(1, {batch_size,
                                                text_encoder_input_attention_mask_shape_[1]});

            decoder_trt_->set_run_dims(0, {batch_size,
                                           decoder_input_fpn_feat_0_shape_[1],
                                           decoder_input_fpn_feat_0_shape_[2],
                                           decoder_input_fpn_feat_0_shape_[3]});
            decoder_trt_->set_run_dims(1, {batch_size,
                                           decoder_input_fpn_feat_1_shape_[1],
                                           decoder_input_fpn_feat_1_shape_[2],
                                           decoder_input_fpn_feat_1_shape_[3]});
            decoder_trt_->set_run_dims(2, {batch_size,
                                           decoder_input_fpn_feat_2_shape_[1],
                                           decoder_input_fpn_feat_2_shape_[2],
                                           decoder_input_fpn_feat_2_shape_[3]});
            decoder_trt_->set_run_dims(3, {batch_size,
                                           decoder_input_fpn_pos_2_shape_[1],
                                           decoder_input_fpn_pos_2_shape_[2],
                                           decoder_input_fpn_pos_2_shape_[3]});

            geometry_encoder_trt_->set_run_dims(0, {batch_size,
                                                  geometry_encoder_input_input_boxes_shape_[1],
                                                  geometry_encoder_input_input_boxes_shape_[2]});
            geometry_encoder_trt_->set_run_dims(1, {batch_size,
                                                  geometry_encoder_input_input_boxes_labels_shape_[1]});
            geometry_encoder_trt_->set_run_dims(2, {batch_size,
                                                  geometry_encoder_input_fpn_feat_2_shape_[1],
                                                  geometry_encoder_input_fpn_feat_2_shape_[2],
                                                  geometry_encoder_input_fpn_feat_2_shape_[3]});    
            geometry_encoder_trt_->set_run_dims(3, {batch_size,
                                                  geometry_encoder_input_fpn_pos_2_shape_[1],
                                                  geometry_encoder_input_fpn_pos_2_shape_[2],
                                                  geometry_encoder_input_fpn_pos_2_shape_[3]});
            
            int prompt_len = text_encoder_input_input_ids_shape_[1] + geometry_encoder_output_geometry_features_shape_[1];
            decoder_trt_->set_run_dims(4, {batch_size,
                                           prompt_len,
                                           decoder_input_prompt_features_shape_[2]});
            decoder_trt_->set_run_dims(5, {batch_size,
                                           prompt_len});
        }
        else
        {
            printf("[FORWARDS] Static model batch size %d does not match input batch size %d.\n", infer_bacth_size, batch_size);
            return {};
        }
    }

    AutoDevice device_guard(gpu_id_);
    adjust_memory(batch_size);
    for (int ibatch = 0; ibatch < (int)input_images.size(); ++ibatch)
    {
        preprocess(input_images[ibatch], ibatch);
    }
    if (!encode_image(stream))
    {
        printf("[FORWARDS] Image encoding failed.\n");
    }
    if (!encode_boxes(boxes, stream))
    {
        printf("[FORWARDS] Box encoding failed.\n");
    }
    if (!encode_text(input_text, stream))
    {
        printf("[FORWARDS] Text encoding failed.\n");
    }


    int text_seq_len = text_encoder_input_input_ids_shape_[1];
    int geom_seq_len = geometry_encoder_output_geometry_features_shape_[1];
    int total_seq_len = text_seq_len + geom_seq_len;
    int feature_dim = decoder_input_prompt_features_shape_[2];

    // 计算单个 Batch 的步长 (以 float/element 数量或字节为单位，这里计算字节偏移)
    size_t text_feat_batch_bytes = text_seq_len * feature_dim * sizeof(float);
    size_t geom_feat_batch_bytes = geom_seq_len * feature_dim * sizeof(float);
    size_t decoder_feat_batch_bytes = total_seq_len * feature_dim * sizeof(float);

    size_t text_mask_batch_bytes = text_seq_len * sizeof(bool); // mask 是 bool
    size_t geom_mask_batch_bytes = geom_seq_len * sizeof(bool);
    size_t decoder_mask_batch_bytes = total_seq_len * sizeof(bool);

    // 获取 GPU 指针 (强转为 char* 以便进行字节偏移计算)
    char* d_decoder_feat = (char*)decoder_input_prompt_features_tensor_.gpu();
    char* d_text_feat    = (char*)text_encoder_output_text_features_tensor_.gpu();
    char* d_geom_feat    = (char*)geometry_encoder_output_geometry_features_tensor_.gpu();

    char* d_decoder_mask = (char*)decoder_input_prompt_mask_tensor_.gpu();
    char* d_text_mask    = (char*)text_encoder_output_text_mask_tensor_.gpu();
    char* d_geom_mask    = (char*)geometry_encoder_output_geometry_mask_tensor_.gpu();

    cudaStream_t stream_ = (cudaStream_t)stream;

    // 逐个 Batch 进行拼接拷贝
    for (int i = 0; i < batch_size; ++i)
    {
        // 1. 拼接 Features: [Text_Feat_i | Geom_Feat_i]
        
        // 目标地址：当前 Batch 的起始位置
        char* dst_feat_ptr = d_decoder_feat + i * decoder_feat_batch_bytes;
        
        // 源地址：Text 的第 i 个 Batch
        cudaMemcpyAsync(dst_feat_ptr, 
                        d_text_feat + i * text_feat_batch_bytes, 
                        text_feat_batch_bytes, 
                        cudaMemcpyDeviceToDevice, stream_);
        
        // 源地址：Geom 的第 i 个 Batch (拷贝到 Text 后面)
        cudaMemcpyAsync(dst_feat_ptr + text_feat_batch_bytes, 
                        d_geom_feat + i * geom_feat_batch_bytes, 
                        geom_feat_batch_bytes, 
                        cudaMemcpyDeviceToDevice, stream_);

        // 2. 拼接 Masks: [Text_Mask_i | Geom_Mask_i]

        // 目标地址
        char* dst_mask_ptr = d_decoder_mask + i * decoder_mask_batch_bytes;

        // 源地址：Text Mask
        cudaMemcpyAsync(dst_mask_ptr, 
                        d_text_mask + i * text_mask_batch_bytes, 
                        text_mask_batch_bytes, 
                        cudaMemcpyDeviceToDevice, stream_);
        
        // 源地址：Geom Mask
        cudaMemcpyAsync(dst_mask_ptr + text_mask_batch_bytes, 
                        d_geom_mask + i * geom_mask_batch_bytes, 
                        geom_mask_batch_bytes, 
                        cudaMemcpyDeviceToDevice, stream_);
    }


    if (!decode(stream))
    {
        printf("[FORWARDS] Decoding failed.\n");
    }
    InferResultArray results;
    for (int ibatch = 0; ibatch < (int)input_images.size(); ++ibatch)
    {
        InferResult result;
        postprocess(result, ibatch, input_text, stream);
        results.push_back(result);
    }
    return results;
}