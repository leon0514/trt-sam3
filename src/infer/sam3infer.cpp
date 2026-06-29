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

    if (!instance->load_engines())
    {
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

    if (!instance->load_engines())
    {
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
    for (auto &buf : original_images_buf_)
    {
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
        if (path.empty())
            return true;
        engine = TensorRT::load(path);
        if (!engine)
        {
            std::cerr << "Failed to load " << name << " from " << path << std::endl;
            return false;
        }
        engine->print(path.c_str());
        if (isdynamic_model_)
            isdynamic_model_ = engine->has_dynamic_dim();
        return true;
    };

    if (!load_engine(vision_encoder_path_, vision_encoder_trt_, "Vision"))
        return false;
    vision_input_shape_ = vision_encoder_trt_->static_dims(0);
    fpn_feat_0_shape_ = vision_encoder_trt_->static_dims(1);
    input_image_height_ = vision_input_shape_[2];
    input_image_width_ = vision_input_shape_[3];

    if (!load_engine(text_encoder_path_, text_encoder_trt_, "Text"))
        return false;
    text_ids_shape_ = text_encoder_trt_->static_dims(0);

    if (!geometry_encoder_path_.empty())
    {
        if (!load_engine(geometry_encoder_path_, geometry_encoder_trt_, "Geometry"))
            return false;
        geom_box_shape_ = geometry_encoder_trt_->static_dims(0);
    }

    if (!load_engine(decoder_path_, decoder_trt_, "Decoder"))
        return false;
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

// 只调用 geometry model 将结果存储在geom_features_cache_和geom_mask__cache_中
bool Sam3Infer::setup_geometry_input(const cv::Mat &image,
                                     const std::string &label,
                                     const std::vector<std::pair<std::string, std::array<float, 4>>> &boxes)
{
    if (geometry_encoder_path_.empty())
    {
        return false;
    }

    AutoDevice device_guard(gpu_id_);

    // step 1 : 图片预处理
    int ibatch = 0;
    Sam3Input input = Sam3Input(image);
    preprocess(input, 0, nullptr);
    // step2 : encode image
    if (!encode_image(1, nullptr))
    {
        return false;
    }

    Sam3PromptUnit prompt_unit = Sam3PromptUnit(label, boxes);
    PromptMeta meta = {0, 0, &prompt_unit};
    std::vector<PromptMeta> batch_meta;
    batch_meta.push_back(meta);

    gather_vision_features(batch_meta, 1, nullptr);

    // step3 : encode_boxes
    if (!encode_boxes(batch_meta, 1, boxes.size(), nullptr))
    {
        return false;
    }
    geom_features_cache_[label] = std::make_shared<tensor::Memory<float>>();
    geom_features_cache_[label]->gpu(geom_features_.gpu_bytes());
    geom_mask_cache_[label] = std::make_shared<tensor::Memory<bool>>();
    geom_mask_cache_[label]->gpu(geom_mask_.gpu_bytes());
    cudaStream_t s = (cudaStream_t) nullptr;
    cudaMemcpyAsync(geom_features_cache_[label]->gpu(), geom_features_.gpu(), geom_features_.gpu_bytes(), cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(geom_mask_cache_[label]->gpu(), geom_mask_.gpu(), geom_mask_.gpu_bytes(), cudaMemcpyDeviceToDevice, s);
    // D2D 异步拷贝，后续 forwards() 会在同一 stream 上顺序执行，无需在此处同步
    return true;
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
    if (use_geom)
    {
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

    if (img.isContinuous())
    {
        memcpy(h_buf, img.data, size_image);
    }
    else
    {
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
void Sam3Infer::gather_vision_features(const std::vector<PromptMeta> &batch_prompts, int batch_size, void *stream)
{
    cudaStream_t s = (cudaStream_t)stream;

    size_t sz_0 = fpn_feat_0_shape_[1] * fpn_feat_0_shape_[2] * fpn_feat_0_shape_[3];
    size_t sz_1 = sz_0 / 4;
    size_t sz_2 = sz_0 / 16;

    // 遍历当前 Batch 的每一个 Prompt
    for (int i = 0; i < batch_size; ++i)
    {
        int img_idx = batch_prompts[i].image_idx;

        // 源地址：Image 队列中的偏移
        float *src_0 = fpn_feat_0_.gpu() + img_idx * sz_0;
        float *src_1 = fpn_feat_1_.gpu() + img_idx * sz_1;
        float *src_2 = fpn_feat_2_.gpu() + img_idx * sz_2;
        float *src_p = fpn_pos_2_.gpu() + img_idx * sz_2;

        // 目标地址：Prompt 队列中的偏移 (i)
        float *dst_0 = fpn_feat_0_gather_.gpu() + i * sz_0;
        float *dst_1 = fpn_feat_1_gather_.gpu() + i * sz_1;
        float *dst_2 = fpn_feat_2_gather_.gpu() + i * sz_2;
        float *dst_p = fpn_pos_2_gather_.gpu() + i * sz_2;

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

    std::array<int64_t, 32> def_ids;
    def_ids.fill(49407);
    std::array<int64_t, 32> def_mask = {0};
    def_mask[0] = 1;

    for (int i = 0; i < batch_size; ++i)
    {
        const Sam3PromptUnit *prompt = batch_prompts[i].ptr;
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
    if (!geometry_encoder_trt_ || max_boxes == 0)
        return true;

    float *h_boxes = geom_boxes_.cpu();
    int64_t *h_labels = geom_labels_.cpu();

    // 清零当前 batch 区域
    memset(h_boxes, 0, batch_size * max_boxes * 4 * sizeof(float));
    memset(h_labels, 0, batch_size * max_boxes * sizeof(int64_t));

    for (int i = 0; i < batch_size; ++i)
    {
        int img_idx = batch_prompts[i].image_idx;
        const Sam3PromptUnit *prompt = batch_prompts[i].ptr;

        float iw = (float)original_image_sizes_[img_idx].first;
        float ih = (float)original_image_sizes_[img_idx].second;

        if (prompt)
        {
            const auto &boxes = prompt->boxes;
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
    float *d_pred_masks = pred_masks_.gpu() + batch_idx * num_queries_ * mask_height_ * mask_width_;
    float *d_pred_boxes = pred_boxes_.gpu() + batch_idx * num_queries_ * 4;
    float *d_pred_logits = pred_logits_.gpu() + batch_idx * num_queries_;
    float *d_presence = presence_logits_.gpu() + batch_idx;

    float *d_filter_boxes = filter_boxes_.gpu() + batch_idx * num_queries_ * 4;
    float *d_filter_scores = filter_scores_.gpu() + batch_idx * num_queries_;
    int *d_filter_indices = filter_indices_.gpu() + batch_idx * num_queries_;

    // 筛选：每个 query 固定位置输出，无效条目 score = -1.0f（不再使用 atomicAdd）
    sam3_postprocess_plane(
        d_pred_masks, d_pred_boxes, d_pred_logits, d_presence,
        d_filter_boxes, d_filter_indices, d_filter_scores,
        num_queries_, mask_height_, mask_width_,
        original_image_sizes_[image_idx].first, original_image_sizes_[image_idx].second,
        confidence_threshold, s);

    // 使用预分配的 pinned memory 直接 D2H 全部 num_queries_ 个结果（异步）
    float *h_filter_boxes = filter_boxes_.cpu() + batch_idx * num_queries_ * 4;
    float *h_filter_scores = filter_scores_.cpu() + batch_idx * num_queries_;
    int *h_filter_indices = filter_indices_.cpu() + batch_idx * num_queries_;

    cudaMemcpyAsync(h_filter_boxes, d_filter_boxes, num_queries_ * 4 * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_filter_scores, d_filter_scores, num_queries_ * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_filter_indices, d_filter_indices, num_queries_ * sizeof(int), cudaMemcpyDeviceToHost, s);

    // 仅需一次同步，等待所有筛选结果传回 CPU
    cudaStreamSynchronize(s);

    // 在 CPU 上收集有效结果索引
    std::vector<int> valid_indices;
    valid_indices.reserve(num_queries_);
    for (int i = 0; i < num_queries_; ++i)
    {
        if (h_filter_scores[i] >= 0.0f)
            valid_indices.push_back(i);
    }

    int count = (int)valid_indices.size();
    if (count == 0)
        return;

    if (!return_mask)
    {
        for (int idx : valid_indices)
        {
            float *b = h_filter_boxes + idx * 4;
            image_result.push_back(object::createBox(b[0], b[1], b[2], b[3], h_filter_scores[idx], -1, label));
        }
        return;
    }

    // --- return_mask = true 分支 ---
    float *h_base_matrix = mask_affine_matrix_.cpu() + image_idx * 6;
    float *h_box_matrices = box_affine_matrices_.cpu() + batch_idx * num_queries_ * 6;

    size_t total_mask_pixels = 0;
    std::vector<size_t> mask_offsets(count);
    std::vector<cv::Size> mask_sizes(count);

    for (int k = 0; k < count; ++k)
    {
        int idx = valid_indices[k];
        float *b = h_filter_boxes + idx * 4;
        int x1 = std::max(0, (int)b[0]);
        int y1 = std::max(0, (int)b[1]);
        int x2 = std::min(original_image_sizes_[image_idx].first, (int)b[2]);
        int y2 = std::min(original_image_sizes_[image_idx].second, (int)b[3]);

        int box_w = std::max(1, x2 - x1);
        int box_h = std::max(1, y2 - y1);

        mask_sizes[k] = cv::Size(box_w, box_h);
        mask_offsets[k] = total_mask_pixels;
        total_mask_pixels += box_w * box_h;

        float *m_dst = h_box_matrices + idx * 6;
        m_dst[0] = h_base_matrix[0];
        m_dst[1] = h_base_matrix[1];
        m_dst[3] = h_base_matrix[3];
        m_dst[4] = h_base_matrix[4];
        m_dst[2] = h_base_matrix[0] * x1 + h_base_matrix[1] * y1 + h_base_matrix[2];
        m_dst[5] = h_base_matrix[3] * x1 + h_base_matrix[4] * y1 + h_base_matrix[5];
    }

    mask_buffer_.gpu(total_mask_pixels);
    mask_buffer_.cpu(total_mask_pixels);

    cudaMemcpyAsync(box_affine_matrices_.gpu() + batch_idx * num_queries_ * 6,
                    h_box_matrices,
                    num_queries_ * 6 * sizeof(float), cudaMemcpyHostToDevice, s);

    for (int k = 0; k < count; ++k)
    {
        int idx = h_filter_indices[valid_indices[k]];
        float *src = d_pred_masks + idx * mask_height_ * mask_width_;
        uint8_t *dst = mask_buffer_.gpu() + mask_offsets[k];
        float *d_matrix = box_affine_matrices_.gpu() + batch_idx * num_queries_ * 6 + valid_indices[k] * 6;

        warp_affine_bilinear_single_channel_mask_plane(
            src, mask_width_, mask_width_, mask_height_,
            dst, mask_sizes[k].width, mask_sizes[k].height,
            d_matrix, 0, s);
    }

    cudaMemcpyAsync(mask_buffer_.cpu(), mask_buffer_.gpu(), total_mask_pixels, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    for (int k = 0; k < count; ++k)
    {
        int idx = valid_indices[k];
        float *b = h_filter_boxes + idx * 4;
        uint8_t *mask_ptr = mask_buffer_.cpu() + mask_offsets[k];
        cv::Mat bin_mask(mask_sizes[k].height, mask_sizes[k].width, CV_8U, mask_ptr);
        image_result.push_back(object::createSegmentationBox(b[0], b[1], b[2], b[3], bin_mask.clone(), h_filter_scores[idx], -1, label));
    }
}

InferResultArray Sam3Infer::forwards(const std::vector<Sam3Input> &inputs, const std::string &geom_label, bool return_mask, void *stream)
{
    if (inputs.empty())
        return {};

    // 检查缓存是否存在
    if (geom_mask_cache_.count(geom_label) == 0 || geom_features_cache_.count(geom_label) == 0)
    {
        std::cerr << "Geometry cache not found for label: " << geom_label << std::endl;
        return {};
    }

    if (inputs.size() > (size_t)max_image_batch_)
    {
        std::cerr << "Input image batch size (" << inputs.size()
                  << ") exceeds maximum supported (" << max_image_batch_ << "). Returning empty." << std::endl;
        return InferResultArray(inputs.size());
    }

    AutoDevice device_guard(gpu_id_);
    cudaStream_t s = (cudaStream_t)stream;

    // 1. Vision Encoder 前处理和推理
    int num_images = inputs.size();
    for (int i = 0; i < num_images; ++i)
        preprocess(inputs[i], i, stream);

    if (!encode_image(num_images, stream))
    {
        return InferResultArray(num_images);
    }

    InferResultArray results(num_images);

    int geom_seq_len = (max_boxes_per_prompt_ + 1);
    size_t single_geom_feat_bytes = geom_seq_len * 256 * sizeof(float);
    size_t single_geom_mask_bytes = geom_seq_len * sizeof(bool);

    // 总 Prompt 长度 = Text (32) + Geom
    int total_prompt_len = text_ids_shape_[1] + geom_seq_len;

    // 获取缓存数据的指针 (假设缓存中存储的是 batch=1 时的结果，位于显存起始位置)
    Sam3PromptUnit text_unit(geom_label, {});

    float *cached_feat_mem = geom_features_cache_[geom_label]->gpu();
    bool *cached_mask_mem = geom_mask_cache_[geom_label]->gpu();

    for (int chunk_start = 0; chunk_start < num_images; chunk_start += max_prompt_batch_)
    {
        int chunk_end = std::min(chunk_start + max_prompt_batch_, num_images);
        int current_batch_size = chunk_end - chunk_start;

        std::vector<PromptMeta> batch_meta;
        batch_meta.reserve(current_batch_size);
        for (int i = 0; i < current_batch_size; ++i)
        {
            batch_meta.push_back({chunk_start + i, -1, &text_unit});
        }

        gather_vision_features(batch_meta, current_batch_size, stream);

        if (!encode_text(batch_meta, current_batch_size, stream))
            continue;

        // 覆盖 Geometry 特征 (从缓存读取)
        // 这一步保持不变，依然使用之前缓存好的 Box 特征
        for (int i = 0; i < current_batch_size; ++i)
        {
            float *dst_feat = geom_features_.gpu() + i * geom_seq_len * 256;
            bool *dst_mask = geom_mask_.gpu() + i * geom_seq_len;

            cudaMemcpyAsync(dst_feat, cached_feat_mem, single_geom_feat_bytes, cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(dst_mask, cached_mask_mem, single_geom_mask_bytes, cudaMemcpyDeviceToDevice, s);
        }

        if (!decode(current_batch_size, total_prompt_len, stream))
            continue;

        for (int k = 0; k < current_batch_size; ++k)
        {
            int image_global_idx = chunk_start + k;
            float conf = inputs[image_global_idx].confidence_threshold;

            // 结果存回对应的 input index
            postprocess(results[image_global_idx], k, image_global_idx, geom_label, conf, return_mask, stream);
        }
    }
    return results;
}

// NMS 辅助函数（按类别分别做 NMS）
static float box_iou(const object::Box &a, const object::Box &b)
{
    float xx1 = std::max(a.left, b.left);
    float yy1 = std::max(a.top, b.top);
    float xx2 = std::min(a.right, b.right);
    float yy2 = std::min(a.bottom, b.bottom);
    float inter = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
    float uni = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.0f;
}

static void nms_filter(InferResult &dets, float threshold = 0.5f)
{
    if (dets.size() <= 1)
        return;

    std::unordered_map<std::string, std::vector<size_t>> groups;
    for (size_t i = 0; i < dets.size(); ++i)
        groups[dets[i].class_name].push_back(i);

    std::vector<bool> suppressed(dets.size(), false);
    for (auto &kv : groups)
    {
        auto &indices = kv.second;
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b)
                  { return dets[a].score > dets[b].score; });
        for (size_t i = 0; i < indices.size(); ++i)
        {
            if (suppressed[indices[i]])
                continue;
            for (size_t j = i + 1; j < indices.size(); ++j)
            {
                if (suppressed[indices[j]])
                    continue;
                if (box_iou(dets[indices[i]].box, dets[indices[j]].box) > threshold)
                {
                    suppressed[indices[j]] = true;
                }
            }
        }
    }

    InferResult filtered;
    for (size_t i = 0; i < dets.size(); ++i)
    {
        if (!suppressed[i])
            filtered.push_back(std::move(dets[i]));
    }
    dets = std::move(filtered);
}

InferResult Sam3Infer::process_pre_detect(const Sam3Input &input, bool return_mask, void *stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    AutoDevice device_guard(gpu_id_);

    // ========== Step 1: 预检测原图（Vision Encoder + pre_detect_labels Decoder）==========
    Sam3Input pre_input;
    pre_input.image = input.image.clone();
    pre_input.confidence_threshold = input.confidence_threshold;
    for (const auto &label : input.pre_detect_labels)
        pre_input.prompts.emplace_back(label);
    pre_input.pre_detect_labels.clear();
    pre_input.merge_results = false;

    preprocess(pre_input, 0, stream);
    if (!encode_image(1, stream))
        return InferResult();

    std::vector<PromptMeta> pre_prompts;
    for (size_t j = 0; j < pre_input.prompts.size(); ++j)
        pre_prompts.push_back({0, (int)j, &pre_input.prompts[j]});

    int pre_max_boxes = 0;
    for (const auto &p : pre_input.prompts)
        pre_max_boxes = std::max(pre_max_boxes, (int)p.boxes.size());
    if (pre_max_boxes > max_boxes_per_prompt_)
        pre_max_boxes = max_boxes_per_prompt_;

    bool pre_use_geom = !geometry_encoder_path_.empty() && pre_max_boxes > 0;
    int pre_prompt_len = text_ids_shape_[1] + (pre_use_geom ? (pre_max_boxes + 1) : 0);

    InferResult pre_results;
    int total_pre_prompts = pre_prompts.size();
    for (int chunk_start = 0; chunk_start < total_pre_prompts; chunk_start += max_prompt_batch_)
    {
        int chunk_end = std::min(chunk_start + max_prompt_batch_, total_pre_prompts);
        int current_batch_size = chunk_end - chunk_start;
        std::vector<PromptMeta> batch_prompts(pre_prompts.begin() + chunk_start, pre_prompts.begin() + chunk_end);

        gather_vision_features(batch_prompts, current_batch_size, stream);
        if (!encode_text(batch_prompts, current_batch_size, stream))
            continue;
        if (pre_use_geom && !encode_boxes(batch_prompts, current_batch_size, pre_max_boxes, stream))
            continue;
        if (!decode(current_batch_size, pre_prompt_len, stream))
            continue;

        for (int k = 0; k < current_batch_size; ++k)
        {
            const auto &meta = batch_prompts[k];
            std::string label = meta.ptr && !meta.ptr->text.empty() ? meta.ptr->text : "object";
            postprocess(pre_results, k, 0, label, pre_input.confidence_threshold, return_mask, stream);
        }
    }

    // 预检测没结果
    if (pre_results.empty())
    {
        InferResult merged_result;
        // 即使预检测为空，如果 merge_results 仍要检测大图 prompts
        if (input.merge_results)
        {
            std::vector<PromptMeta> full_prompts;
            for (size_t j = 0; j < input.prompts.size(); ++j)
                full_prompts.push_back({0, (int)j, &input.prompts[j]});

            int full_max_boxes = 0;
            for (const auto &p : input.prompts)
                full_max_boxes = std::max(full_max_boxes, (int)p.boxes.size());
            if (full_max_boxes > max_boxes_per_prompt_)
                full_max_boxes = max_boxes_per_prompt_;

            bool full_use_geom = !geometry_encoder_path_.empty() && full_max_boxes > 0;
            int full_prompt_len = text_ids_shape_[1] + (full_use_geom ? (full_max_boxes + 1) : 0);

            int total_full_prompts = full_prompts.size();
            for (int chunk_start = 0; chunk_start < total_full_prompts; chunk_start += max_prompt_batch_)
            {
                int chunk_end = std::min(chunk_start + max_prompt_batch_, total_full_prompts);
                int current_batch_size = chunk_end - chunk_start;
                std::vector<PromptMeta> batch_prompts(full_prompts.begin() + chunk_start, full_prompts.begin() + chunk_end);

                gather_vision_features(batch_prompts, current_batch_size, stream);
                if (!encode_text(batch_prompts, current_batch_size, stream))
                    continue;
                if (full_use_geom && !encode_boxes(batch_prompts, current_batch_size, full_max_boxes, stream))
                    continue;
                if (!decode(current_batch_size, full_prompt_len, stream))
                    continue;

                for (int k = 0; k < current_batch_size; ++k)
                {
                    const auto &meta = batch_prompts[k];
                    std::string label = meta.ptr && !meta.ptr->text.empty() ? meta.ptr->text : "object";
                    postprocess(merged_result, k, 0, label, input.confidence_threshold, return_mask, stream);
                }
            }
            nms_filter(merged_result, 0.5f);
        }
        return merged_result;
    }

    // 初始化 merged_result 为 pre_results
    InferResult merged_result = pre_results;

    // ========== Step 2: 复用 Vision Features，如果 merge_results 做大图 prompts 检测 ==========
    if (input.merge_results)
    {
        std::vector<PromptMeta> full_prompts;
        for (size_t j = 0; j < input.prompts.size(); ++j)
            full_prompts.push_back({0, (int)j, &input.prompts[j]});

        int full_max_boxes = 0;
        for (const auto &p : input.prompts)
            full_max_boxes = std::max(full_max_boxes, (int)p.boxes.size());
        if (full_max_boxes > max_boxes_per_prompt_)
            full_max_boxes = max_boxes_per_prompt_;

        bool full_use_geom = !geometry_encoder_path_.empty() && full_max_boxes > 0;
        int full_prompt_len = text_ids_shape_[1] + (full_use_geom ? (full_max_boxes + 1) : 0);

        int total_full_prompts = full_prompts.size();
        for (int chunk_start = 0; chunk_start < total_full_prompts; chunk_start += max_prompt_batch_)
        {
            int chunk_end = std::min(chunk_start + max_prompt_batch_, total_full_prompts);
            int current_batch_size = chunk_end - chunk_start;
            std::vector<PromptMeta> batch_prompts(full_prompts.begin() + chunk_start, full_prompts.begin() + chunk_end);

            gather_vision_features(batch_prompts, current_batch_size, stream);
            if (!encode_text(batch_prompts, current_batch_size, stream))
                continue;
            if (full_use_geom && !encode_boxes(batch_prompts, current_batch_size, full_max_boxes, stream))
                continue;
            if (!decode(current_batch_size, full_prompt_len, stream))
                continue;

            for (int k = 0; k < current_batch_size; ++k)
            {
                const auto &meta = batch_prompts[k];
                std::string label = meta.ptr && !meta.ptr->text.empty() ? meta.ptr->text : "object";
                postprocess(merged_result, k, 0, label, input.confidence_threshold, return_mask, stream);
            }
        }
    }

    // ========== Step 3: 用 ominicrop 合并 pre_results 的框（仅基于预检测结果）==========
    std::vector<omnicrop::BBox> boxes;
    for (const auto &det : pre_results)
    {
        boxes.emplace_back(det.box.left, det.box.top, det.box.right, det.box.bottom);
    }

    int img_w = input.image.cols;
    int img_h = input.image.rows;
    // 读取用户传入的 ominicrop 配置，未设置则使用默认值
    int max_crop_size = input.pre_crop_max_size > 0 ? input.pre_crop_max_size : 640;
    int padding = input.pre_crop_padding >= 0 ? input.pre_crop_padding : 20;
    omnicrop::OmniCropEngine crop_engine(max_crop_size, padding);

    omnicrop::Config cfg;
    cfg.w_diou = input.pre_crop_w_diou;
    cfg.w_expansion = input.pre_crop_w_expansion;
    cfg.crop_count_penalty = input.pre_crop_count_penalty;
    cfg.nms_threshold = input.pre_crop_nms_threshold;
    cfg.enable_aspect_ratio_fix = input.pre_crop_enable_ar_fix;
    cfg.target_aspect_ratio = input.pre_crop_target_ar;

    auto crops = crop_engine.cluster_and_crop(boxes, img_w, img_h, cfg);

    // 调试信息：打印 ominicrop 结果
    std::cerr << "[omnicrop] input_boxes=" << boxes.size() << ", output_crops=" << crops.size() << std::endl;
    for (size_t i = 0; i < crops.size(); ++i)
    {
        std::cerr << "[omnicrop] crop[" << i << "]: x1=" << crops[i].x1 << ", y1=" << crops[i].y1
                  << ", x2=" << crops[i].x2 << ", y2=" << crops[i].y2
                  << ", w=" << crops[i].width() << ", h=" << crops[i].height() << std::endl;
    }

    // ========== Step 4: 对每个 crop 区域，复用原图显存做 prompts 检测 ==========
    int original_img_w = img_w;
    int original_img_h = img_h;

    for (const auto &crop : crops)
    {
        int x1 = (int)crop.x1;
        int y1 = (int)crop.y1;
        int cw = (int)(crop.x2 - crop.x1);
        int ch = (int)(crop.y2 - crop.y1);

        x1 = std::max(0, x1);
        y1 = std::max(0, y1);
        cw = std::min(cw, img_w - x1);
        ch = std::min(ch, img_h - y1);

        if (cw <= 0 || ch <= 0)
            continue;

        // 构造 crop resize 的 affine 矩阵
        affine::CropResizeMatrix crop_matrix;
        crop_matrix.compute(std::make_tuple(cw, ch),
                            std::make_tuple(input_image_width_, input_image_height_),
                            std::make_tuple(x1, y1));

        float *h_mat = affine_matrix_.cpu();
        memcpy(h_mat, crop_matrix.d2i, sizeof(crop_matrix.d2i));
        cudaMemcpyAsync(affine_matrix_.gpu(), h_mat, sizeof(crop_matrix.d2i), cudaMemcpyHostToDevice, s);

        // 更新 mask_affine_matrix_
        affine::ResizeMatrix mask_m;
        mask_m.compute(std::make_tuple(mask_width_, mask_height_),
                       std::make_tuple(cw, ch));
        memcpy(mask_affine_matrix_.cpu(), mask_m.d2i, sizeof(mask_m.d2i));
        cudaMemcpyAsync(mask_affine_matrix_.gpu(), mask_m.d2i, sizeof(mask_m.d2i), cudaMemcpyHostToDevice, s);

        original_image_sizes_[0] = {cw, ch};

        // 直接从原图显存 buffer warp 到 preprocessed_images_（不重新拷贝图像数据）
        warp_affine_bilinear_and_normalize_plane(
            original_images_buf_[0]->gpu(), original_img_w * 3, original_img_w, original_img_h,
            preprocessed_images_.gpu(),
            input_image_width_, input_image_height_,
            affine_matrix_.gpu(), 114, preprocess_norm_, s);

        if (!encode_image(1, stream))
            continue;

        // 调整 prompts 中的 box 坐标为相对于 crop 区域
        std::vector<Sam3PromptUnit> adjusted_prompts = input.prompts;
        for (auto &pu : adjusted_prompts)
        {
            for (auto &bp : pu.boxes)
            {
                bp.second[0] -= x1;
                bp.second[1] -= y1;
                bp.second[2] -= x1;
                bp.second[3] -= y1;
            }
        }

        std::vector<PromptMeta> crop_prompts;
        for (size_t j = 0; j < adjusted_prompts.size(); ++j)
            crop_prompts.push_back({0, (int)j, &adjusted_prompts[j]});

        int crop_max_boxes = 0;
        for (const auto &p : adjusted_prompts)
            crop_max_boxes = std::max(crop_max_boxes, (int)p.boxes.size());
        if (crop_max_boxes > max_boxes_per_prompt_)
            crop_max_boxes = max_boxes_per_prompt_;

        bool crop_use_geom = !geometry_encoder_path_.empty() && crop_max_boxes > 0;
        int crop_prompt_len = text_ids_shape_[1] + (crop_use_geom ? (crop_max_boxes + 1) : 0);

        size_t before_count = merged_result.size();
        int total_crop_prompts = crop_prompts.size();
        for (int chunk_start = 0; chunk_start < total_crop_prompts; chunk_start += max_prompt_batch_)
        {
            int chunk_end = std::min(chunk_start + max_prompt_batch_, total_crop_prompts);
            int current_batch_size = chunk_end - chunk_start;
            std::vector<PromptMeta> batch_prompts(crop_prompts.begin() + chunk_start, crop_prompts.begin() + chunk_end);

            gather_vision_features(batch_prompts, current_batch_size, stream);
            if (!encode_text(batch_prompts, current_batch_size, stream))
                continue;
            if (crop_use_geom && !encode_boxes(batch_prompts, current_batch_size, crop_max_boxes, stream))
                continue;
            if (!decode(current_batch_size, crop_prompt_len, stream))
                continue;

            for (int k = 0; k < current_batch_size; ++k)
            {
                const auto &meta = batch_prompts[k];
                std::string label = meta.ptr && !meta.ptr->text.empty() ? meta.ptr->text : "object";
                postprocess(merged_result, k, 0, label, input.confidence_threshold, return_mask, stream);
            }
        }

        // 坐标映射：将本次 crop 添加的结果映射回原图
        for (size_t i = before_count; i < merged_result.size(); ++i)
        {
            merged_result[i].box.left += x1;
            merged_result[i].box.top += y1;
            merged_result[i].box.right += x1;
            merged_result[i].box.bottom += y1;
        }
    }

    // ========== Step 5: NMS 去重（按类别分别做）==========
    nms_filter(merged_result, 0.5f);

    // 将 ominicrop 的裁剪框作为可视化标记附加到结果中（score=-1 避免影响正常展示）
    for (const auto &crop : crops)
    {
        object::DetectionBox crop_box;
        crop_box.type = object::ObjectType::DETECTION;
        crop_box.box.left = crop.x1;
        crop_box.box.top = crop.y1;
        crop_box.box.right = crop.x2;
        crop_box.box.bottom = crop.y2;
        crop_box.score = -1.0f;
        crop_box.class_id = -2;
        crop_box.class_name = "__CROP__";
        merged_result.push_back(std::move(crop_box));
    }

    return merged_result;
}

InferResultArray Sam3Infer::forwards(const std::vector<Sam3Input> &inputs, bool return_mask, void *stream)
{
    if (inputs.empty())
        return {};

    // 检查是否有 input 需要预检测
    bool has_pre_detect = false;
    for (const auto &input : inputs)
    {
        if (!input.pre_detect_labels.empty())
        {
            has_pre_detect = true;
            break;
        }
    }

    if (has_pre_detect)
    {
        // 逐个处理，避免混合 batch 逻辑过于复杂
        InferResultArray results;
        results.reserve(inputs.size());
        for (const auto &input : inputs)
        {
            if (input.pre_detect_labels.empty())
            {
                auto r = forwards({input}, return_mask, stream);
                results.push_back(r.empty() ? InferResult() : r[0]);
            }
            else
            {
                results.push_back(process_pre_detect(input, return_mask, stream));
            }
        }
        return results;
    }

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
    if (max_boxes_input > max_boxes_per_prompt_)
        max_boxes_input = max_boxes_per_prompt_;

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
        if (!encode_text(batch_prompts, current_batch_size, stream))
            continue;

        // c. Encode Geometry
        if (use_geom)
        {
            if (!encode_boxes(batch_prompts, current_batch_size, max_boxes_input, stream))
                continue;
        }

        // d. Decode
        if (!decode(current_batch_size, prompt_len, stream))
            continue;

        // e. Postprocess & Collect Results
        for (int k = 0; k < current_batch_size; ++k)
        {
            const auto &meta = batch_prompts[k];
            std::string label = "object";
            if (meta.ptr && !meta.ptr->text.empty())
                label = meta.ptr->text;

            float conf = inputs[meta.image_idx].confidence_threshold;

            // 结果写入对应的 image_idx
            postprocess(results[meta.image_idx], k, meta.image_idx, label, conf, return_mask, stream);
        }
    }

    return results;
}