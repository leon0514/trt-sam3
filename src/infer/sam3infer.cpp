#include "infer/sam3infer.hpp"
#include "common/affine.hpp"
#include "common/image.hpp"
#include "kernels/process_kernel_warp.hpp"
#include "kernels/postprocess.cuh"
#include "common/createObject.hpp"
#include <algorithm>

// 全局 load 函数
std::shared_ptr<InferBase> load(
    const std::string &vision_encoder_path,
    const std::string &text_encoder_path,
    const std::string &decoder_path,
    int gpu_id)
{
    return Sam3Infer::create_instance(vision_encoder_path, text_encoder_path, decoder_path, gpu_id);
}

// 静态工厂函数
std::shared_ptr<Sam3Infer> Sam3Infer::create_instance(
    const std::string &vision_encoder_path,
    const std::string &text_encoder_path,
    const std::string &decoder_path,
    int gpu_id)
{
    std::string geom_path = "";
    auto instance = std::make_shared<Sam3Infer>(
        vision_encoder_path, text_encoder_path, geom_path, decoder_path, gpu_id);

    if (!instance->load_engines()) {
        std::cerr << "Failed to load Sam3Infer engines!" << std::endl;
        return nullptr;
    }
    return instance;
}

std::shared_ptr<Sam3Infer> Sam3Infer::create_instance(
    const std::string &vision_encoder_path,
    const std::string &text_encoder_path,
    const std::string &geometry_encoder_path,
    const std::string &decoder_path,
    int gpu_id)
{
    auto instance = std::make_shared<Sam3Infer>(
        vision_encoder_path, text_encoder_path, geometry_encoder_path, decoder_path, gpu_id);

    if (!instance->load_engines()) {
        std::cerr << "Failed to load Sam3Infer engines!" << std::endl;
        return nullptr;
    }
    return instance;
}

Sam3Infer::Sam3Infer(
    const std::string &vision_encoder_path,
    const std::string &text_encoder_path,
    const std::string &geometry_encoder_path,
    const std::string &decoder_path,
    int gpu_id)
    : InferBase(),
      vision_encoder_path_(vision_encoder_path),
      text_encoder_path_(text_encoder_path),
      geometry_encoder_path_(geometry_encoder_path),
      decoder_path_(decoder_path),
      gpu_id_(gpu_id)
{
    // 初始化预留 Image Buffer
    original_images_buf_.resize(max_image_batch_);
    for(auto& buf : original_images_buf_) {
        buf = std::make_shared<tensor::Memory<uint8_t>>();
    }
    // 预留 size 记录
    original_image_sizes_.resize(max_image_batch_);
}

bool Sam3Infer::load_engines()
{
    AutoDevice device_guard(gpu_id_);
    auto load_engine = [&](const std::string &path, std::shared_ptr<TensorRT::Engine> &engine, const char *name)
    {
        if (path.empty()) return true;
        engine = TensorRT::load(path);
        if (!engine) {
            std::cerr << "Failed to load " << name << " from " << path << std::endl;
            return false;
        }
        if (isdynamic_model_) isdynamic_model_ = engine->has_dynamic_dim();
        return true;
    };

    if (!load_engine(vision_encoder_path_, vision_encoder_trt_, "Vision")) return false;
    vision_input_shape_ = vision_encoder_trt_->static_dims(0);
    fpn_feat_0_shape_ = vision_encoder_trt_->static_dims(1);
    input_image_height_ = vision_input_shape_[2];
    input_image_width_ = vision_input_shape_[3];

    if (!load_engine(text_encoder_path_, text_encoder_trt_, "Text")) return false;
    text_ids_shape_ = text_encoder_trt_->static_dims(0);

    if (!geometry_encoder_path_.empty()) {
        if (!load_engine(geometry_encoder_path_, geometry_encoder_trt_, "Geometry")) return false;
        geom_box_shape_ = geometry_encoder_trt_->static_dims(0);
    }

    if (!load_engine(decoder_path_, decoder_trt_, "Decoder")) return false;
    auto pred_masks_shape = decoder_trt_->static_dims(6);
    auto pred_boxes_shape = decoder_trt_->static_dims(7);
    num_queries_ = pred_boxes_shape[1];
    mask_width_ = pred_masks_shape[2];
    mask_height_ = pred_masks_shape[3];

    // 初始化固定显存
    allocate_memory_once();

    return true;
}

void Sam3Infer::setup_text_inputs(const std::string &input_text, const std::array<int64_t, 32> &input_ids, const std::array<int64_t, 32> &attention_mask)
{
    text_input_map_[input_text] = std::make_pair(input_ids, attention_mask);
}

void Sam3Infer::allocate_memory_once()
{
    // 1. Image Batch 相关 (按 max_image_batch_ 分配)
    affine_matrix_.cpu(max_image_batch_ * 6);
    affine_matrix_.gpu(max_image_batch_ * 6);
    
    mask_affine_matrix_.cpu(max_image_batch_ * 6);
    mask_affine_matrix_.gpu(max_image_batch_ * 6);

    preprocessed_images_.gpu(max_image_batch_ * 3 * input_image_height_ * input_image_width_);

    // Vision Encoder Outputs
    size_t feat_0_sz_one = fpn_feat_0_shape_[1] * fpn_feat_0_shape_[2] * fpn_feat_0_shape_[3];
    size_t feat_0_sz_max_img = max_image_batch_ * feat_0_sz_one;
    
    fpn_feat_0_.gpu(feat_0_sz_max_img);
    fpn_feat_1_.gpu(feat_0_sz_max_img / 4);
    fpn_feat_2_.gpu(feat_0_sz_max_img / 16);
    fpn_pos_2_.gpu(feat_0_sz_max_img / 16);

    // 2. Decoder Batch 相关 (按 max_prompt_batch_ 分配)
    size_t feat_0_sz_max_pmt = max_prompt_batch_ * feat_0_sz_one;
    fpn_feat_0_gather_.gpu(feat_0_sz_max_pmt);
    fpn_feat_1_gather_.gpu(feat_0_sz_max_pmt / 4);
    fpn_feat_2_gather_.gpu(feat_0_sz_max_pmt / 16);
    fpn_pos_2_gather_.gpu(feat_0_sz_max_pmt / 16);

    // Text Input
    size_t text_in_sz = max_prompt_batch_ * text_ids_shape_[1];
    text_input_ids_.cpu(text_in_sz);
    text_input_ids_.gpu(text_in_sz);
    text_attention_mask_.cpu(text_in_sz);
    text_attention_mask_.gpu(text_in_sz);
    
    // Text Feats
    text_features_.gpu(text_in_sz * 256);
    text_mask_.gpu(text_in_sz);

    // Geometry (按 max_prompt_batch_ * max_boxes 分配)
    bool use_geom = (!geometry_encoder_path_.empty());
    if (use_geom) {
        size_t box_sz = max_prompt_batch_ * max_boxes_per_prompt_ * 4;
        geom_boxes_.cpu(box_sz);
        geom_boxes_.gpu(box_sz);
        geom_labels_.cpu(max_prompt_batch_ * max_boxes_per_prompt_);
        geom_labels_.gpu(max_prompt_batch_ * max_boxes_per_prompt_);
        
        size_t geom_feat_sz = max_prompt_batch_ * (max_boxes_per_prompt_ + 1) * 256;
        geom_features_.gpu(geom_feat_sz);
        geom_mask_.gpu(max_prompt_batch_ * (max_boxes_per_prompt_ + 1));
    }

    // Decoder Input (Prompt feats)
    // 假设 geometry 不会超过预设
    size_t total_prompt_len = text_ids_shape_[1] + (use_geom ? (max_boxes_per_prompt_ + 1) : 0);
    prompt_features_.gpu(max_prompt_batch_ * total_prompt_len * 256);
    prompt_mask_.gpu(max_prompt_batch_ * total_prompt_len);

    // Decoder Output
    pred_masks_.gpu(max_prompt_batch_ * num_queries_ * mask_height_ * mask_width_);
    pred_boxes_.gpu(max_prompt_batch_ * num_queries_ * 4);
    pred_logits_.gpu(max_prompt_batch_ * num_queries_);
    presence_logits_.gpu(max_prompt_batch_ * 1);

    // Postprocess Buffers
    size_t post_sz = max_prompt_batch_ * num_queries_;
    filter_boxes_.cpu(post_sz * 4);
    filter_boxes_.gpu(post_sz * 4);
    filter_scores_.cpu(post_sz);
    filter_scores_.gpu(post_sz);
    filter_indices_.cpu(post_sz);
    filter_indices_.gpu(post_sz);
    box_count_.cpu(1);
    box_count_.gpu(1);

    // Mask Postprocess (假设每个 prompt 最多出一个有效的 mask 来估计 buffer)
    // 或者直接分配足够大的显存。这里为了安全，按 max_prompt_batch_ * num_queries_ 分配
    // 实际上通常只会有几个 valid box。
    // 分配一个安全值，比如 256MB，或动态检查。这里按最大可能分配。
    // 假设每个 prompt 出 1 个 mask，最大 max_prompt_batch_ 个。
    // 如果出很多 mask，循环处理。
    // 这里的 box_affine_matrices_ 是给 Postprocess kernel 用的
    box_affine_matrices_.cpu(post_sz * 6);
    box_affine_matrices_.gpu(post_sz * 6);
    
    // // Mask Buffer: 预分配一个较大的池子，例如 512MB
    // size_t mask_pool_size = 256 * 1024 * 1024; 
    // mask_buffer_.gpu(mask_pool_size);
    // mask_buffer_.cpu(mask_pool_size);
}

void Sam3Infer::set_binding_dim(std::shared_ptr<TensorRT::Engine> &engine, int idx, const std::vector<int> &dims)
{
    if (engine && isdynamic_model_)
        engine->set_run_dims(idx, dims);
}

void Sam3Infer::preprocess(const Sam3Input &input, int ibatch, void *stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    const cv::Mat &img = input.image;
    tensor::Image img_tensor = tensor::cvimg(img);
    
    // 记录原始尺寸
    original_image_sizes_[ibatch] = {img_tensor.width, img_tensor.height};

    affine::ResizeMatrix matrix;
    matrix.compute(std::make_tuple(img_tensor.width, img_tensor.height),
                   std::make_tuple(input_image_width_, input_image_height_));

    size_t size_image = img_tensor.width * img_tensor.height * 3;
    uint8_t *h_buf = original_images_buf_[ibatch]->cpu(size_image); 

    if (img.isContinuous()) {
        memcpy(h_buf, img.data, size_image);
    } else {
        int w_bytes = img_tensor.width * 3;
        for (int h = 0; h < img_tensor.height; ++h)
            memcpy(h_buf + h * w_bytes, img.ptr<uint8_t>(h), w_bytes);
    }

    float *h_mat = affine_matrix_.cpu() + ibatch * 6;
    memcpy(h_mat, matrix.d2i, sizeof(matrix.d2i));

    cudaMemcpyAsync(original_images_buf_[ibatch]->gpu(size_image), h_buf, size_image, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(affine_matrix_.gpu() + ibatch * 6, h_mat, sizeof(matrix.d2i), cudaMemcpyHostToDevice, s);
    
    // Mask Affine Matrix
    affine::ResizeMatrix mask_m;
    mask_m.compute(std::make_tuple(mask_width_, mask_height_),
                   std::make_tuple(img_tensor.width, img_tensor.height));
    memcpy(mask_affine_matrix_.cpu() + ibatch * 6, mask_m.d2i, sizeof(mask_m.d2i));
    cudaMemcpyAsync(mask_affine_matrix_.gpu() + ibatch * 6, mask_m.d2i, sizeof(mask_m.d2i), cudaMemcpyHostToDevice, s);

    warp_affine_bilinear_and_normalize_plane(
        original_images_buf_[ibatch]->gpu(), img_tensor.width * 3, img_tensor.width, img_tensor.height,
        preprocessed_images_.gpu() + ibatch * 3 * input_image_height_ * input_image_width_,
        input_image_width_, input_image_height_,
        affine_matrix_.gpu() + ibatch * 6, 114, preprocess_norm_, s);
}

bool Sam3Infer::encode_image(int batch_size, void *stream)
{
    // 输入维度设置
    set_binding_dim(vision_encoder_trt_, 0, {batch_size, 3, input_image_height_, input_image_width_});

    return vision_encoder_trt_->forward({{"images", preprocessed_images_.gpu()},
                                         {"fpn_feat_0", fpn_feat_0_.gpu()},
                                         {"fpn_feat_1", fpn_feat_1_.gpu()},
                                         {"fpn_feat_2", fpn_feat_2_.gpu()},
                                         {"fpn_pos_2", fpn_pos_2_.gpu()}},
                                        (cudaStream_t)stream);
}

// 核心优化：Gather 模式
void Sam3Infer::gather_vision_features(const std::vector<PromptMeta>& batch_prompts, int batch_size, void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    
    size_t sz_0 = fpn_feat_0_shape_[1] * fpn_feat_0_shape_[2] * fpn_feat_0_shape_[3];
    size_t sz_1 = sz_0 / 4;
    size_t sz_2 = sz_0 / 16;
    
    // 遍历当前 Batch 的每一个 Prompt
    for(int i = 0; i < batch_size; ++i) {
        int img_idx = batch_prompts[i].image_idx;

        // 源地址：Image 队列中的偏移
        float* src_0 = fpn_feat_0_.gpu() + img_idx * sz_0;
        float* src_1 = fpn_feat_1_.gpu() + img_idx * sz_1;
        float* src_2 = fpn_feat_2_.gpu() + img_idx * sz_2;
        float* src_p = fpn_pos_2_.gpu() + img_idx * sz_2;

        // 目标地址：Prompt 队列中的偏移 (i)
        float* dst_0 = fpn_feat_0_gather_.gpu() + i * sz_0;
        float* dst_1 = fpn_feat_1_gather_.gpu() + i * sz_1;
        float* dst_2 = fpn_feat_2_gather_.gpu() + i * sz_2;
        float* dst_p = fpn_pos_2_gather_.gpu() + i * sz_2;

        // 异步拷贝
        // 这里的拷贝次数等于 batch_size，通常 < 100，开销可控。
        // 极致优化可以写个 Kernel，但在 C++ 逻辑中维护更简单
        cudaMemcpyAsync(dst_0, src_0, sz_0 * sizeof(float), cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(dst_1, src_1, sz_1 * sizeof(float), cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(dst_2, src_2, sz_2 * sizeof(float), cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(dst_p, src_p, sz_2 * sizeof(float), cudaMemcpyDeviceToDevice, s);
    }
}

bool Sam3Infer::encode_text(const std::vector<PromptMeta> &batch_prompts, int batch_size, void *stream)
{
    int seq_len = 32;
    int64_t *h_ids = text_input_ids_.cpu();
    int64_t *h_mask = text_attention_mask_.cpu();

    std::array<int64_t, 32> def_ids; def_ids.fill(49407);
    std::array<int64_t, 32> def_mask = {0}; def_mask[0] = 1;

    for (int i = 0; i < batch_size; ++i)
    {
        const Sam3PromptUnit* prompt = batch_prompts[i].ptr;
        const int64_t *src_ids = def_ids.data();
        const int64_t *src_mask = def_mask.data();

        if (prompt && text_input_map_.count(prompt->text)) 
        {
            src_ids = text_input_map_[prompt->text].first.data();
            src_mask = text_input_map_[prompt->text].second.data();
        } 
        
        memcpy(h_ids + i * seq_len, src_ids, seq_len * sizeof(int64_t));
        memcpy(h_mask + i * seq_len, src_mask, seq_len * sizeof(int64_t));
    }

    cudaStream_t s = (cudaStream_t)stream;
    cudaMemcpyAsync(text_input_ids_.gpu(), h_ids, batch_size * seq_len * sizeof(int64_t), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(text_attention_mask_.gpu(), h_mask, batch_size * seq_len * sizeof(int64_t), cudaMemcpyHostToDevice, s);

    // 设置维度
    set_binding_dim(text_encoder_trt_, 0, {batch_size, seq_len});
    set_binding_dim(text_encoder_trt_, 1, {batch_size, seq_len});

    return text_encoder_trt_->forward({{"input_ids", text_input_ids_.gpu()},
                                       {"attention_mask", text_attention_mask_.gpu()},
                                       {"text_features", text_features_.gpu()},
                                       {"text_mask", text_mask_.gpu()}},
                                      s);
}

bool Sam3Infer::encode_boxes(const std::vector<PromptMeta> &batch_prompts, int batch_size, int max_boxes, void *stream)
{
    if (!geometry_encoder_trt_ || max_boxes == 0) return true;

    float *h_boxes = geom_boxes_.cpu();
    int64_t *h_labels = geom_labels_.cpu();

    // 清零当前 batch 区域
    memset(h_boxes, 0, batch_size * max_boxes * 4 * sizeof(float));
    memset(h_labels, 0, batch_size * max_boxes * sizeof(int64_t));

    for (int i = 0; i < batch_size; ++i)
    {
        int img_idx = batch_prompts[i].image_idx;
        const Sam3PromptUnit* prompt = batch_prompts[i].ptr;
        
        float iw = (float)original_image_sizes_[img_idx].first;
        float ih = (float)original_image_sizes_[img_idx].second;

        if (prompt) {
            const auto& boxes = prompt->boxes;
            for (size_t k = 0; k < boxes.size() && k < (size_t)max_boxes; ++k) 
            {
                const auto &box = boxes[k];
                int64_t label = (box.first == "pos") ? 1 : 0;

                float x1 = box.second[0], y1 = box.second[1];
                float x2 = box.second[2], y2 = box.second[3];

                // Normalize
                float cx = (x1 + x2) * 0.5f / iw;
                float cy = (y1 + y2) * 0.5f / ih;
                float w = (x2 - x1) / iw;
                float h = (y2 - y1) / ih;

                int idx_base = i * max_boxes + k;
                h_labels[idx_base] = label;
                h_boxes[idx_base * 4 + 0] = cx;
                h_boxes[idx_base * 4 + 1] = cy;
                h_boxes[idx_base * 4 + 2] = w;
                h_boxes[idx_base * 4 + 3] = h;
            }
        }
    }

    cudaStream_t s = (cudaStream_t)stream;
    cudaMemcpyAsync(geom_boxes_.gpu(), h_boxes, batch_size * max_boxes * 4 * sizeof(float), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(geom_labels_.gpu(), h_labels, batch_size * max_boxes * sizeof(int64_t), cudaMemcpyHostToDevice, s);

    set_binding_dim(geometry_encoder_trt_, 0, {batch_size, max_boxes, 4});
    set_binding_dim(geometry_encoder_trt_, 1, {batch_size, max_boxes});
    set_binding_dim(geometry_encoder_trt_, 2, {batch_size, 256, 72, 72}); 
    set_binding_dim(geometry_encoder_trt_, 3, {batch_size, 256, 72, 72});

    // 注意：这里使用 Gather 后的 Vision Feature
    return geometry_encoder_trt_->forward({{"input_boxes", geom_boxes_.gpu()},
                                           {"input_boxes_labels", geom_labels_.gpu()},
                                           {"fpn_feat_2", fpn_feat_2_gather_.gpu()}, 
                                           {"fpn_pos_2", fpn_pos_2_gather_.gpu()},
                                           {"geometry_features", geom_features_.gpu()},
                                           {"geometry_mask", geom_mask_.gpu()}},
                                          s);
}

bool Sam3Infer::decode(int batch_size, int prompt_len, void *stream)
{
    int text_len = text_ids_shape_[1];
    int feat_dim = 256;
    size_t feat_sz = feat_dim * sizeof(float);
    size_t mask_sz = sizeof(bool);

    char *d_prompt = (char *)prompt_features_.gpu();
    char *d_prompt_m = (char *)prompt_mask_.gpu();
    char *d_text = (char *)text_features_.gpu();
    char *d_text_m = (char *)text_mask_.gpu();
    char *d_geom = (char *)geom_features_.gpu();
    char *d_geom_m = (char *)geom_mask_.gpu();

    cudaStream_t s = (cudaStream_t)stream;

    // 拼接 Prompt Features
    for (int i = 0; i < batch_size; ++i)
    {
        size_t prompt_off = i * prompt_len * feat_sz;
        size_t prompt_m_off = i * prompt_len * mask_sz;

        cudaMemcpyAsync(d_prompt + prompt_off, d_text + i * text_len * feat_sz, text_len * feat_sz, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(d_prompt_m + prompt_m_off, d_text_m + i * text_len * mask_sz, text_len * mask_sz, cudaMemcpyDeviceToDevice, s);

        if (prompt_len > text_len)
        {
            size_t geom_len = prompt_len - text_len;
            cudaMemcpyAsync(d_prompt + prompt_off + text_len * feat_sz, d_geom + i * geom_len * feat_sz, geom_len * feat_sz, cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(d_prompt_m + prompt_m_off + text_len * mask_sz, d_geom_m + i * geom_len * mask_sz, geom_len * mask_sz, cudaMemcpyDeviceToDevice, s);
        }
    }

    set_binding_dim(decoder_trt_, 0, {batch_size, fpn_feat_0_shape_[1], fpn_feat_0_shape_[2], fpn_feat_0_shape_[3]});
    set_binding_dim(decoder_trt_, 1, {batch_size, fpn_feat_0_shape_[1], fpn_feat_0_shape_[2] / 2, fpn_feat_0_shape_[3] / 2});
    set_binding_dim(decoder_trt_, 2, {batch_size, fpn_feat_0_shape_[1], fpn_feat_0_shape_[2] / 4, fpn_feat_0_shape_[3] / 4});
    set_binding_dim(decoder_trt_, 3, {batch_size, fpn_feat_0_shape_[1], fpn_feat_0_shape_[2] / 4, fpn_feat_0_shape_[3] / 4});
    set_binding_dim(decoder_trt_, 4, {batch_size, prompt_len, 256});
    set_binding_dim(decoder_trt_, 5, {batch_size, prompt_len});

    // 使用 Gather 后的特征
    return decoder_trt_->forward({{"fpn_feat_0", fpn_feat_0_gather_.gpu()},
                                  {"fpn_feat_1", fpn_feat_1_gather_.gpu()},
                                  {"fpn_feat_2", fpn_feat_2_gather_.gpu()},
                                  {"fpn_pos_2", fpn_pos_2_gather_.gpu()},
                                  {"prompt_features", prompt_features_.gpu()},
                                  {"prompt_mask", prompt_mask_.gpu()},
                                  {"pred_masks", pred_masks_.gpu()},
                                  {"pred_boxes", pred_boxes_.gpu()},
                                  {"pred_logits", pred_logits_.gpu()},
                                  {"presence_logits", presence_logits_.gpu()}},
                                 s);
}

void Sam3Infer::postprocess(InferResult &image_result, int batch_idx, int image_idx, const std::string &label, float confidence_threshold, bool return_mask, void *stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    
    // 指针偏移 (基于当前 Batch 内的 index: batch_idx)
    float* d_pred_masks = pred_masks_.gpu() + batch_idx * num_queries_ * mask_height_ * mask_width_;
    float* d_pred_boxes = pred_boxes_.gpu() + batch_idx * num_queries_ * 4;
    float* d_pred_logits = pred_logits_.gpu() + batch_idx * num_queries_;
    float* d_presence = presence_logits_.gpu() + batch_idx;

    float* d_filter_boxes = filter_boxes_.gpu() + batch_idx * num_queries_ * 4;
    float* d_filter_scores = filter_scores_.gpu() + batch_idx * num_queries_;
    int* d_filter_indices = filter_indices_.gpu() + batch_idx * num_queries_;
    
    cudaMemsetAsync(box_count_.gpu(), 0, sizeof(int), s);

    // 筛选
    sam3_postprocess_plane(
        d_pred_masks, d_pred_boxes, d_pred_logits, d_presence,
        d_filter_boxes, d_filter_indices, d_filter_scores, box_count_.gpu(),
        num_queries_, mask_height_, mask_width_,
        original_image_sizes_[image_idx].first, original_image_sizes_[image_idx].second,
        confidence_threshold, s);

    cudaMemcpyAsync(box_count_.cpu(), box_count_.gpu(), sizeof(int), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s); 
    int count = *box_count_.cpu();

    if (count > 0)
    {
        std::vector<float> h_boxes(count * 4);
        std::vector<float> h_scores(count);
        std::vector<int> h_indices(count);
        
        cudaMemcpyAsync(h_boxes.data(), d_filter_boxes, count * 4 * sizeof(float), cudaMemcpyDeviceToHost, s);
        cudaMemcpyAsync(h_scores.data(), d_filter_scores, count * sizeof(float), cudaMemcpyDeviceToHost, s);
        cudaMemcpyAsync(h_indices.data(), d_filter_indices, count * sizeof(int), cudaMemcpyDeviceToHost, s);

        if (!return_mask)
        {
            for (int i = 0; i < count; ++i)
            {
                float *b = h_boxes.data() + i * 4;
                image_result.push_back(object::createBox(b[0], b[1], b[2], b[3], h_scores[i], -1, label));
            }
            return;
        }

        float* h_base_matrix = mask_affine_matrix_.cpu() + image_idx * 6;
        float* h_box_matrices = box_affine_matrices_.cpu(); 

        size_t total_mask_pixels = 0;
        std::vector<size_t> mask_offsets(count);
        std::vector<cv::Size> mask_sizes(count);

        for (int i = 0; i < count; ++i)
        {
            float *b = h_boxes.data() + i * 4;
            int x1 = std::max(0, (int)b[0]);
            int y1 = std::max(0, (int)b[1]);
            int x2 = std::min(original_image_sizes_[image_idx].first, (int)b[2]);
            int y2 = std::min(original_image_sizes_[image_idx].second, (int)b[3]);
            
            int box_w = std::max(1, x2 - x1);
            int box_h = std::max(1, y2 - y1);
            
            mask_sizes[i] = cv::Size(box_w, box_h);
            mask_offsets[i] = total_mask_pixels;
            total_mask_pixels += box_w * box_h;

            float* m_dst = h_box_matrices + i * 6;
            m_dst[0] = h_base_matrix[0]; m_dst[1] = h_base_matrix[1];
            m_dst[3] = h_base_matrix[3]; m_dst[4] = h_base_matrix[4];
            m_dst[2] = h_base_matrix[0] * x1 + h_base_matrix[1] * y1 + h_base_matrix[2];
            m_dst[5] = h_base_matrix[3] * x1 + h_base_matrix[4] * y1 + h_base_matrix[5];
        }

        mask_buffer_.gpu(total_mask_pixels);
        mask_buffer_.cpu(total_mask_pixels);

        cudaMemcpyAsync(box_affine_matrices_.gpu(), box_affine_matrices_.cpu(), count * 6 * sizeof(float), cudaMemcpyHostToDevice, s);

        for (int i = 0; i < count; ++i)
        {
            int idx = h_indices[i];
            float *src = d_pred_masks + idx * mask_height_ * mask_width_;
            uint8_t *dst = mask_buffer_.gpu() + mask_offsets[i];
            float *d_matrix = box_affine_matrices_.gpu() + i * 6;

            warp_affine_bilinear_single_channel_mask_plane(
                src, mask_width_, mask_width_, mask_height_,
                dst, mask_sizes[i].width, mask_sizes[i].height,
                d_matrix, 0, s);
        }

        cudaMemcpyAsync(mask_buffer_.cpu(), mask_buffer_.gpu(), total_mask_pixels, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);

        for (int i = 0; i < count; ++i)
        {
            float *b = h_boxes.data() + i * 4;
            uint8_t *mask_ptr = mask_buffer_.cpu() + mask_offsets[i];
            cv::Mat bin_mask(mask_sizes[i].height, mask_sizes[i].width, CV_8U, mask_ptr);
            image_result.push_back(object::createSegmentationBox(b[0], b[1], b[2], b[3], bin_mask.clone(), h_scores[i], -1, label));
        }
    }
}

InferResultArray Sam3Infer::forwards(const std::vector<Sam3Input> &inputs, bool return_mask, void *stream)
{
    if (inputs.empty()) return {};

    // 1. 检查图片数量是否超限
    if (inputs.size() > (size_t)max_image_batch_) 
    {
        std::cerr << "Input image batch size (" << inputs.size() 
                  << ") exceeds maximum supported (" << max_image_batch_ << "). Returning empty." << std::endl;
        return InferResultArray(inputs.size()); // 返回空结果
    }

    AutoDevice device_guard(gpu_id_);

    std::vector<PromptMeta> all_prompts;
    int max_boxes_input = 0;

    for (size_t i = 0; i < inputs.size(); ++i) 
    {
        if (inputs[i].prompts.empty()) 
        {
            all_prompts.push_back({(int)i, -1, nullptr});
        } 
        else 
        {
            for (size_t j = 0; j < inputs[i].prompts.size(); ++j) 
            {
                all_prompts.push_back({(int)i, (int)j, &inputs[i].prompts[j]});
                if ((int)inputs[i].prompts[j].boxes.size() > max_boxes_input) 
                {
                    max_boxes_input = (int)inputs[i].prompts[j].boxes.size();
                }
            }
        }
    }

    // 3. Vision Encoder (一次性处理所有图片)
    int num_images = inputs.size();
    for (int i = 0; i < num_images; ++i)
        preprocess(inputs[i], i, stream);

    if (!encode_image(num_images, stream)) 
    {
        return InferResultArray(num_images);
    }

    // 4. Decoder 分批循环 (Batch Splitting)
    InferResultArray results(num_images);
    int total_prompts = all_prompts.size();
    bool use_geom = !geometry_encoder_path_.empty() && max_boxes_input > 0;
    
    // 如果实际 Box 数量超过预设显存分配，截断 (防止溢出)
    if (max_boxes_input > max_boxes_per_prompt_) max_boxes_input = max_boxes_per_prompt_;
    
    int prompt_len = text_ids_shape_[1] + (use_geom ? (max_boxes_input + 1) : 0);

    for (int chunk_start = 0; chunk_start < total_prompts; chunk_start += max_prompt_batch_) 
    {
        int chunk_end = std::min(chunk_start + max_prompt_batch_, total_prompts);
        int current_batch_size = chunk_end - chunk_start;

        // 构造当前 Batch 的 Prompt 列表
        std::vector<PromptMeta> batch_prompts(all_prompts.begin() + chunk_start, all_prompts.begin() + chunk_end);

        // a. Gather Vision Features (从 N 张图的特征中 Gather 到当前 batch 的 M 个 Prompt 特征)
        gather_vision_features(batch_prompts, current_batch_size, stream);

        // b. Encode Text
        if (!encode_text(batch_prompts, current_batch_size, stream)) continue;

        // c. Encode Geometry
        if (use_geom) 
        {
            if (!encode_boxes(batch_prompts, current_batch_size, max_boxes_input, stream)) continue;
        }

        // d. Decode
        if (!decode(current_batch_size, prompt_len, stream)) continue;

        // e. Postprocess & Collect Results
        for (int k = 0; k < current_batch_size; ++k) 
        {
            const auto& meta = batch_prompts[k];
            std::string label = "object";
            if (meta.ptr && !meta.ptr->text.empty()) label = meta.ptr->text;
            
            float conf = inputs[meta.image_idx].confidence_threshold;
            
            // 结果写入对应的 image_idx
            postprocess(results[meta.image_idx], k, meta.image_idx, label, conf, return_mask, stream);
        }
    }

    return results;
}