#include "infer/sam3infer.hpp"
#include "common/affine.hpp"
#include "common/image.hpp"
#include "kernels/process_kernel_warp.hpp"
#include "kernels/postprocess.cuh"
#include "common/createObject.hpp"

// 全局 load 函数保持不变
std::shared_ptr<InferBase> load(
    const std::string &vision_encoder_path,
    const std::string &text_encoder_path,
    const std::string &decoder_path,
    int gpu_id)
{
    return Sam3Infer::create_instance(vision_encoder_path, text_encoder_path, decoder_path, gpu_id);
}

// 静态工厂函数保持不变
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

    return true;
}

void Sam3Infer::setup_text_inputs(const std::string &input_text, const std::array<int64_t, 32> &input_ids, const std::array<int64_t, 32> &attention_mask)
{
    text_input_map_[input_text] = std::make_pair(input_ids, attention_mask);
}

void Sam3Infer::adjust_memory(int image_batch_size, int total_prompts, int max_boxes)
{
    // 1. Image Batch 相关 (大小 N)
    if (original_images_buf_.size() < image_batch_size) {
        for (int i = original_images_buf_.size(); i < image_batch_size; ++i)
            original_images_buf_.emplace_back(std::make_shared<tensor::Memory<uint8_t>>());
    }
    
    // Resize 矩阵只需要 batch N
    affine_matrix_.cpu(image_batch_size * 6);
    affine_matrix_.gpu(image_batch_size * 6);
    
    // Mask Affine Matrix 需要 batch M (因为每个 Prompt 产生结果后都要做后处理)
    // 或者我们可以在后处理循环中复用 batch N 的矩阵？ 
    // 实际上后处理是按 Query 来的，为了简便，我们为每个 Postprocess 分配足够空间
    // 这里 postprocess 是串行或并行的，但 mask_buffer 需要足够大
    // 修正：Mask恢复只需要对应原图的矩阵，所以 N 份矩阵就够了，但在 postprocess 内核中需要索引。
    // 简单起见，分配 N 份。
    mask_affine_matrix_.cpu(image_batch_size * 6);
    mask_affine_matrix_.gpu(image_batch_size * 6);

    preprocessed_images_.gpu(image_batch_size * 3 * input_image_height_ * input_image_width_);

    // Vision Encoder Outputs (Batch N)
    size_t feat_0_sz_one = fpn_feat_0_shape_[1] * fpn_feat_0_shape_[2] * fpn_feat_0_shape_[3];
    size_t feat_0_sz_n = image_batch_size * feat_0_sz_one;
    
    fpn_feat_0_.gpu(feat_0_sz_n);
    fpn_feat_1_.gpu(feat_0_sz_n / 4);
    fpn_feat_2_.gpu(feat_0_sz_n / 16);
    fpn_pos_2_.gpu(feat_0_sz_n / 16);

    // 2. Prompt/Decoder Batch 相关 (大小 M = Total Prompts)
    size_t feat_0_sz_m = total_prompts * feat_0_sz_one;
    fpn_feat_0_expanded_.gpu(feat_0_sz_m);
    fpn_feat_1_expanded_.gpu(feat_0_sz_m / 4);
    fpn_feat_2_expanded_.gpu(feat_0_sz_m / 16);
    fpn_pos_2_expanded_.gpu(feat_0_sz_m / 16);

    // Text Input
    size_t text_in_sz = total_prompts * text_ids_shape_[1];
    text_input_ids_.cpu(text_in_sz);
    text_input_ids_.gpu(text_in_sz);
    text_attention_mask_.cpu(text_in_sz);
    text_attention_mask_.gpu(text_in_sz);
    
    // Text Feats
    text_features_.gpu(text_in_sz * 256);
    text_mask_.gpu(text_in_sz);

    // Geometry
    bool use_geom = (max_boxes > 0 && geometry_encoder_trt_);
    if (use_geom) {
        size_t box_sz = total_prompts * max_boxes * 4;
        geom_boxes_.cpu(box_sz);
        geom_boxes_.gpu(box_sz);
        geom_labels_.cpu(total_prompts * max_boxes);
        geom_labels_.gpu(total_prompts * max_boxes);
        size_t geom_feat_sz = total_prompts * (max_boxes + 1) * 256;
        geom_features_.gpu(geom_feat_sz);
        geom_mask_.gpu(total_prompts * (max_boxes + 1));
    }

    // Decoder Input (Prompt feats)
    size_t total_prompt_len = text_ids_shape_[1] + (use_geom ? (max_boxes + 1) : 0);
    prompt_features_.gpu(total_prompts * total_prompt_len * 256);
    prompt_mask_.gpu(total_prompts * total_prompt_len);

    // Decoder Output
    pred_masks_.gpu(total_prompts * num_queries_ * mask_height_ * mask_width_);
    pred_boxes_.gpu(total_prompts * num_queries_ * 4);
    pred_logits_.gpu(total_prompts * num_queries_);
    presence_logits_.gpu(total_prompts * 1);

    // Postprocess - 为最大可能的单个批次操作保留空间
    // 我们假设 postprocess 是一次处理一个 prompt 的结果，或者并行处理所有
    // 这里的 buffer 主要用于 cuda kernel 内部的临时存储，通常不需要 total_prompts 那么大，
    // 但为了确保并行安全性，我们按 total_prompts * num_queries 分配，或者循环复用。
    // 为了极致速度，直接并行处理所有 Prompt 的结果。
    size_t post_sz = total_prompts * num_queries_;
    filter_boxes_.cpu(post_sz * 4);
    filter_boxes_.gpu(post_sz * 4);
    filter_scores_.cpu(post_sz);
    filter_scores_.gpu(post_sz);
    filter_indices_.cpu(post_sz);
    filter_indices_.gpu(post_sz);
    box_count_.cpu(1);
    box_count_.gpu(1);
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
    // 这里注意：original_image_sizes_ 必须在 loop 外 clear
    original_image_sizes_.emplace_back(img_tensor.width, img_tensor.height);

    affine::ResizeMatrix matrix;
    matrix.compute(std::make_tuple(img_tensor.width, img_tensor.height),
                   std::make_tuple(input_image_width_, input_image_height_));

    size_t size_image = img_tensor.width * img_tensor.height * 3;
    uint8_t *h_buf = original_images_buf_[ibatch]->cpu(size_image); // Re-use buffer if capacity enough

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
    
    // Mask Affine Matrix 的 CPU 部分也顺便在这里计算，方便后面用
    affine::ResizeMatrix mask_m;
    mask_m.compute(std::make_tuple(mask_width_, mask_height_),
                   std::make_tuple(img_tensor.width, img_tensor.height));
    memcpy(mask_affine_matrix_.cpu() + ibatch * 6, mask_m.d2i, sizeof(mask_m.d2i));
    // GPU拷贝在循环外统一做或者这里做都可以，为了效率统一做
    cudaMemcpyAsync(mask_affine_matrix_.gpu() + ibatch * 6, mask_m.d2i, sizeof(mask_m.d2i), cudaMemcpyHostToDevice, s);

    warp_affine_bilinear_and_normalize_plane(
        original_images_buf_[ibatch]->gpu(), img_tensor.width * 3, img_tensor.width, img_tensor.height,
        preprocessed_images_.gpu() + ibatch * 3 * input_image_height_ * input_image_width_,
        input_image_width_, input_image_height_,
        affine_matrix_.gpu() + ibatch * 6, 114, preprocess_norm_, s);
}

bool Sam3Infer::encode_image(void *stream)
{
    return vision_encoder_trt_->forward({{"images", preprocessed_images_.gpu()},
                                         {"fpn_feat_0", fpn_feat_0_.gpu()},
                                         {"fpn_feat_1", fpn_feat_1_.gpu()},
                                         {"fpn_feat_2", fpn_feat_2_.gpu()},
                                         {"fpn_pos_2", fpn_pos_2_.gpu()}},
                                        (cudaStream_t)stream);
}

// 核心加速：特征扩展
void Sam3Infer::expand_vision_features(const std::vector<int>& prompts_per_image, void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    int num_images = prompts_per_image.size();
    
    // 计算特征图大小 (in float elements)
    size_t sz_0 = fpn_feat_0_shape_[1] * fpn_feat_0_shape_[2] * fpn_feat_0_shape_[3];
    size_t sz_1 = sz_0 / 4;
    size_t sz_2 = sz_0 / 16;
    
    int current_prompt_idx = 0;
    
    // 遍历每一张图
    for(int i = 0; i < num_images; ++i) {
        int count = prompts_per_image[i];
        if (count == 0) continue;

        // 源地址 (Image i 的特征)
        float* src_0 = fpn_feat_0_.gpu() + i * sz_0;
        float* src_1 = fpn_feat_1_.gpu() + i * sz_1;
        float* src_2 = fpn_feat_2_.gpu() + i * sz_2;
        float* src_p = fpn_pos_2_.gpu() + i * sz_2;

        // 目标地址 (Prompt j, j+1... 的特征)
        // 我们可以一次拷贝 count 个块吗？不行，因为内存是不连续的吗？
        // Decoder Input 是 [Total_Prompts, C, H, W]。是连续的。
        // 所以，我们需要把 src 复制 count 次，填充到 dst 中。
        
        for (int k = 0; k < count; ++k) {
            int dst_idx = current_prompt_idx + k;
            
            cudaMemcpyAsync(fpn_feat_0_expanded_.gpu() + dst_idx * sz_0, src_0, sz_0 * sizeof(float), cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(fpn_feat_1_expanded_.gpu() + dst_idx * sz_1, src_1, sz_1 * sizeof(float), cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(fpn_feat_2_expanded_.gpu() + dst_idx * sz_2, src_2, sz_2 * sizeof(float), cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(fpn_pos_2_expanded_.gpu() + dst_idx * sz_2, src_p, sz_2 * sizeof(float), cudaMemcpyDeviceToDevice, s);
        }
        current_prompt_idx += count;
    }
}

bool Sam3Infer::encode_text(const std::vector<Sam3Input> &inputs, int total_prompts, void *stream)
{
    int seq_len = 32;
    int64_t *h_ids = text_input_ids_.cpu();
    int64_t *h_mask = text_attention_mask_.cpu();

    std::array<int64_t, 32> def_ids; def_ids.fill(49407);
    std::array<int64_t, 32> def_mask = {0}; def_mask[0] = 1;

    int global_idx = 0;
    for (const auto& inp : inputs) 
    {
        for (const auto& prompt : inp.prompts) 
        {
            const int64_t *src_ids = def_ids.data();
            const int64_t *src_mask = def_mask.data();

            if (text_input_map_.count(prompt.text)) 
            {
                src_ids = text_input_map_[prompt.text].first.data();
                src_mask = text_input_map_[prompt.text].second.data();
            } 
            else if (!prompt.text.empty()) 
            {
                printf("[Warning] Text prompt '%s' not cached. Using default.\n", prompt.text.c_str());
            }

            memcpy(h_ids + global_idx * seq_len, src_ids, seq_len * sizeof(int64_t));
            memcpy(h_mask + global_idx * seq_len, src_mask, seq_len * sizeof(int64_t));
            global_idx++;
        }
    }

    cudaStream_t s = (cudaStream_t)stream;
    cudaMemcpyAsync(text_input_ids_.gpu(), h_ids, total_prompts * seq_len * sizeof(int64_t), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(text_attention_mask_.gpu(), h_mask, total_prompts * seq_len * sizeof(int64_t), cudaMemcpyHostToDevice, s);

    return text_encoder_trt_->forward({{"input_ids", text_input_ids_.gpu()},
                                       {"attention_mask", text_attention_mask_.gpu()},
                                       {"text_features", text_features_.gpu()},
                                       {"text_mask", text_mask_.gpu()}},
                                      s);
}

bool Sam3Infer::encode_boxes(const std::vector<Sam3Input> &inputs, int total_prompts, int max_boxes, void *stream)
{
    if (!geometry_encoder_trt_ || max_boxes == 0) return true;

    float *h_boxes = geom_boxes_.cpu();
    int64_t *h_labels = geom_labels_.cpu();

    memset(h_boxes, 0, total_prompts * max_boxes * 4 * sizeof(float));
    memset(h_labels, 0, total_prompts * max_boxes * sizeof(int64_t));

    int global_idx = 0;
    int img_idx = 0;
    
    for (const auto& inp : inputs) 
    {
        float iw = (float)original_image_sizes_[img_idx].first;
        float ih = (float)original_image_sizes_[img_idx].second;
        
        for (const auto& prompt : inp.prompts) 
        {
            const auto& boxes = prompt.boxes;
            
            for (size_t k = 0; k < boxes.size(); ++k) 
            {
                const auto &box = boxes[k];
                int64_t label = (box.first == "pos") ? 1 : 0;

                float x1 = box.second[0], y1 = box.second[1];
                float x2 = box.second[2], y2 = box.second[3];

                float cx = (x1 + x2) * 0.5f / iw;
                float cy = (y1 + y2) * 0.5f / ih;
                float w = (x2 - x1) / iw;
                float h = (y2 - y1) / ih;

                int idx_base = global_idx * max_boxes + k;
                h_labels[idx_base] = label;
                h_boxes[idx_base * 4 + 0] = cx;
                h_boxes[idx_base * 4 + 1] = cy;
                h_boxes[idx_base * 4 + 2] = w;
                h_boxes[idx_base * 4 + 3] = h;
            }
            global_idx++;
        }
        img_idx++;
    }

    cudaStream_t s = (cudaStream_t)stream;
    cudaMemcpyAsync(geom_boxes_.gpu(), h_boxes, total_prompts * max_boxes * 4 * sizeof(float), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(geom_labels_.gpu(), h_labels, total_prompts * max_boxes * sizeof(int64_t), cudaMemcpyHostToDevice, s);

    // Geometry Encoder 输入需要的是 expand 后的特征吗？
    // 通常 geometry encoder 会用到 fpn_feat_2 等作为 context，
    // 如果模型结构如此，这里必须使用 fpn_feat_2_expanded_ (大小 M) 而不是原始的 (大小 N)。
    // 假设 geometry encoder 也是对每个 prompt 运行一次。
    
    return geometry_encoder_trt_->forward({{"input_boxes", geom_boxes_.gpu()},
                                           {"input_boxes_labels", geom_labels_.gpu()},
                                           {"fpn_feat_2", fpn_feat_2_expanded_.gpu()}, // 使用 Expanded
                                           {"fpn_pos_2", fpn_pos_2_expanded_.gpu()},   // 使用 Expanded
                                           {"geometry_features", geom_features_.gpu()},
                                           {"geometry_mask", geom_mask_.gpu()}},
                                          s);
}

bool Sam3Infer::decode(int total_prompts, int prompt_len, void *stream)
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

    // 这里的 batch 是 total_prompts
    for (int i = 0; i < total_prompts; ++i)
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

    // 注意：Decoder 输入使用 Expanded Vision Features
    return decoder_trt_->forward({{"fpn_feat_0", fpn_feat_0_expanded_.gpu()},
                                  {"fpn_feat_1", fpn_feat_1_expanded_.gpu()},
                                  {"fpn_feat_2", fpn_feat_2_expanded_.gpu()},
                                  {"fpn_pos_2", fpn_pos_2_expanded_.gpu()},
                                  {"prompt_features", prompt_features_.gpu()},
                                  {"prompt_mask", prompt_mask_.gpu()},
                                  {"pred_masks", pred_masks_.gpu()},
                                  {"pred_boxes", pred_boxes_.gpu()},
                                  {"pred_logits", pred_logits_.gpu()},
                                  {"presence_logits", presence_logits_.gpu()}},
                                 s);
}

void Sam3Infer::postprocess(InferResult &image_result, int global_prompt_idx, int image_idx, const std::string &label, float confidence_threshold, bool return_mask, void *stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    
    // 复用 filter_boxes_ 内存，通过 offset 定位到当前 prompt 的工作区
    // 注意：sam3_postprocess_plane 内部是针对单张图的一组 query 进行过滤。
    // 这里我们传入的是 global_prompt_idx 对应的结果。
    
    // 指针偏移
    float* d_pred_masks = pred_masks_.gpu() + global_prompt_idx * num_queries_ * mask_height_ * mask_width_;
    float* d_pred_boxes = pred_boxes_.gpu() + global_prompt_idx * num_queries_ * 4;
    float* d_pred_logits = pred_logits_.gpu() + global_prompt_idx * num_queries_;
    float* d_presence = presence_logits_.gpu() + global_prompt_idx;

    // 为了防止并行冲突，我们之前分配了 total_prompts * size 的 filter buffer
    float* d_filter_boxes = filter_boxes_.gpu() + global_prompt_idx * num_queries_ * 4;
    float* d_filter_scores = filter_scores_.gpu() + global_prompt_idx * num_queries_;
    int* d_filter_indices = filter_indices_.gpu() + global_prompt_idx * num_queries_;
    
    // Box Count 还是用同一个，每次 Memset 0 即可，因为我们要同步拿结果
    cudaMemsetAsync(box_count_.gpu(), 0, sizeof(int), s);

    sam3_postprocess_plane(
        d_pred_masks, d_pred_boxes, d_pred_logits, d_presence,
        d_filter_boxes, d_filter_indices, d_filter_scores, box_count_.gpu(),
        num_queries_, mask_height_, mask_width_,
        original_image_sizes_[image_idx].first, original_image_sizes_[image_idx].second,
        confidence_threshold, s);

    // 同步获取数量
    cudaMemcpyAsync(box_count_.cpu(), box_count_.gpu(), sizeof(int), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s); // 必须同步才能知道有多少个框
    int count = *box_count_.cpu();

    if (count > 0)
    {
        // 拷贝筛选后的结果到 CPU
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

        box_affine_matrices_.cpu(count * 6); 
        box_affine_matrices_.gpu(count * 6); 
        float* h_base_matrix = mask_affine_matrix_.cpu() + image_idx * 6; // 原图的基础变换矩阵
        float* h_box_matrices = box_affine_matrices_.cpu();

        // 计算所有 Box 的总像素面积，以便分配紧凑的 buffer
        size_t total_mask_pixels = 0;
        std::vector<size_t> mask_offsets(count);
        std::vector<cv::Size> mask_sizes(count);

        for (int i = 0; i < count; ++i)
        {
            float *b = h_boxes.data() + i * 4;
            int x1 = (int)b[0];
            int y1 = (int)b[1];
            int x2 = (int)b[2];
            int y2 = (int)b[3];
            
            // 限制边界防止越界
            int img_w = original_image_sizes_[image_idx].first;
            int img_h = original_image_sizes_[image_idx].second;
            x1 = std::max(0, x1); y1 = std::max(0, y1);
            x2 = std::min(img_w, x2); y2 = std::min(img_h, y2);
            
            int box_w = std::max(1, x2 - x1);
            int box_h = std::max(1, y2 - y1);
            
            mask_sizes[i] = cv::Size(box_w, box_h);
            mask_offsets[i] = total_mask_pixels;
            total_mask_pixels += box_w * box_h;

            // --- 核心数学逻辑 ---
            // 原始矩阵 M 将 (dst_x, dst_y) 映射到 source。
            // 现在的 dst 是 box 内部坐标 (bx, by)。
            // 关系: dst_x_global = bx + box_x1
            //       dst_y_global = by + box_y1
            // Src = M * Dst_Global
            // Src = M * (Dst_Box + Offset)
            // Src = M * Dst_Box + (M * Offset)
            // 所以，新的平移分量 Tx' = M00*x1 + M01*y1 + Tx
            //      新的平移分量 Ty' = M10*x1 + M11*y1 + Ty
            
            float* m_dst = h_box_matrices + i * 6;
            // 复制旋转缩放分量
            m_dst[0] = h_base_matrix[0]; m_dst[1] = h_base_matrix[1];
            m_dst[3] = h_base_matrix[3]; m_dst[4] = h_base_matrix[4];
            
            // 计算新的平移分量
            m_dst[2] = h_base_matrix[0] * x1 + h_base_matrix[1] * y1 + h_base_matrix[2];
            m_dst[5] = h_base_matrix[3] * x1 + h_base_matrix[4] * y1 + h_base_matrix[5];
        }

        // 2. 分配 Mask 显存 (按需分配)
        mask_buffer_.gpu(total_mask_pixels);
        mask_buffer_.cpu(total_mask_pixels);
        
        // 上传矩阵
        cudaMemcpyAsync(box_affine_matrices_.gpu(), box_affine_matrices_.cpu(), count * 6 * sizeof(float), cudaMemcpyHostToDevice, s);

        // 3. 循环发射 Kernel
        // 这里的循环开销很小，因为只是发射 Kernel，不做同步
        for (int i = 0; i < count; ++i)
        {
            int idx = h_indices[i];
            float *src = d_pred_masks + idx * mask_height_ * mask_width_;
            uint8_t *dst = mask_buffer_.gpu() + mask_offsets[i];
            float *d_matrix = box_affine_matrices_.gpu() + i * 6;

            warp_affine_bilinear_single_channel_mask_plane(
                src, mask_width_, mask_width_, mask_height_,
                dst, mask_sizes[i].width, mask_sizes[i].height, // 目标尺寸变为 Box 尺寸
                d_matrix, 0, s);
        }

        // 4. 拷贝回 CPU
        cudaMemcpyAsync(mask_buffer_.cpu(), mask_buffer_.gpu(), total_mask_pixels, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s); // Sync 2: 等待所有 Mask 生成完成

        // 5. 组装结果
        for (int i = 0; i < count; ++i)
        {
            float *b = h_boxes.data() + i * 4;
            // 这里的 mask_ptr 指向的是紧凑排列的 Mask 数据
            uint8_t *mask_ptr = mask_buffer_.cpu() + mask_offsets[i];
            
            // 构造 cv::Mat，注意它现在是 Box 大小，而不是全图大小
            cv::Mat bin_mask(mask_sizes[i].height, mask_sizes[i].width, CV_8U, mask_ptr);
            
            // 结果追加
            // 客户端或可视化代码需要知道 mask 是相对于 box 的 (SegmentationBox 通常隐含了这个逻辑)
            image_result.push_back(object::createSegmentationBox(b[0], b[1], b[2], b[3], bin_mask.clone(), h_scores[i], -1, label));
        }
    }
}

InferResultArray Sam3Infer::forwards(const std::vector<Sam3Input> &inputs, bool return_mask, void *stream)
{
    original_image_sizes_.clear();
    if (inputs.empty()) return {};

    int num_images = inputs.size();
    
    // 1. 统计 Prompt 总数和 Max Boxes
    int total_prompts = 0;
    int max_boxes = 0;
    std::vector<int> prompts_per_image;
    
    for (const auto &inp : inputs) 
    {
        int p_count = inp.prompts.empty() ? 1 : inp.prompts.size(); // 处理空 prompt 情况，虽然理应有
        if (inp.prompts.empty()) 
        {
            p_count = 1;
        }
        prompts_per_image.push_back(p_count);
        total_prompts += p_count;

        for(const auto& p : inp.prompts) {
            if ((int)p.boxes.size() > max_boxes) max_boxes = (int)p.boxes.size();
        }
    }

    bool use_geom = !geometry_encoder_path_.empty() && max_boxes > 0;
    int geom_len = use_geom ? (max_boxes + 1) : 0;
    int prompt_len = text_ids_shape_[1] + geom_len;

    AutoDevice device_guard(gpu_id_);

    // 2. 设置维度 (Decoder 使用 total_prompts 作为 Batch)
    if (isdynamic_model_)
    {
        // Vision Encoder: Batch = Image Count
        set_binding_dim(vision_encoder_trt_, 0, {num_images, 3, input_image_height_, input_image_width_});
        
        // Text Encoder: Batch = Total Prompts
        set_binding_dim(text_encoder_trt_, 0, {total_prompts, text_ids_shape_[1]});
        set_binding_dim(text_encoder_trt_, 1, {total_prompts, text_ids_shape_[1]});

        // Decoder: Batch = Total Prompts (Vision feats 维度也随之改变)
        set_binding_dim(decoder_trt_, 0, {total_prompts, fpn_feat_0_shape_[1], fpn_feat_0_shape_[2], fpn_feat_0_shape_[3]});
        set_binding_dim(decoder_trt_, 1, {total_prompts, fpn_feat_0_shape_[1], fpn_feat_0_shape_[2] / 2, fpn_feat_0_shape_[3] / 2});
        set_binding_dim(decoder_trt_, 2, {total_prompts, fpn_feat_0_shape_[1], fpn_feat_0_shape_[2] / 4, fpn_feat_0_shape_[3] / 4});
        set_binding_dim(decoder_trt_, 3, {total_prompts, fpn_feat_0_shape_[1], fpn_feat_0_shape_[2] / 4, fpn_feat_0_shape_[3] / 4});
        set_binding_dim(decoder_trt_, 4, {total_prompts, prompt_len, 256});
        set_binding_dim(decoder_trt_, 5, {total_prompts, prompt_len});

        if (use_geom) 
        {
            set_binding_dim(geometry_encoder_trt_, 0, {total_prompts, max_boxes, 4});
            set_binding_dim(geometry_encoder_trt_, 1, {total_prompts, max_boxes});
            // Geometry Encoder 的 Vision 输入也必须是 Expanded 后的
            set_binding_dim(geometry_encoder_trt_, 2, {total_prompts, 256, 72, 72}); // fpn_feat_2
            set_binding_dim(geometry_encoder_trt_, 3, {total_prompts, 256, 72, 72}); // fpn_pos_2
        }
    }

    adjust_memory(num_images, total_prompts, use_geom ? max_boxes : 0);

    for (int i = 0; i < num_images; ++i)
        preprocess(inputs[i], i, stream);

    bool ok = true;
    ok &= encode_image(stream);
    
    if (!ok) 
    {
        std::cerr << "Vision Encoding failed" << std::endl;
        return InferResultArray(num_images);
    }

    // Expand Features (Copy N -> M)
    expand_vision_features(prompts_per_image, stream);

    ok &= encode_text(inputs, total_prompts, stream);
    
    if (use_geom)
        ok &= encode_boxes(inputs, total_prompts, max_boxes, stream);
        
    // Decode (Batch = total_prompts)
    if (ok)
        ok &= decode(total_prompts, prompt_len, stream);

    if (!ok) 
    {
        std::cerr << "Forward failed at decode stage" << std::endl;
        return InferResultArray(num_images);
    }

    InferResultArray results(num_images); // 预分配 N 个结果集
    
    int global_idx = 0;
    for (int i = 0; i < num_images; ++i) 
    {
        float confidence_threshold = inputs[i].confidence_threshold;
        int p_count = prompts_per_image[i];
        for (int p = 0; p < p_count; ++p) 
        {
            std::string lbl = "object";
            // 获取对应 Prompt 的 label
            if (!inputs[i].prompts.empty()) 
            {
                lbl = inputs[i].prompts[p].text;
                if(lbl.empty()) lbl = "object";
            }
            postprocess(results[i], global_idx, i, lbl, confidence_threshold, return_mask, stream);
            global_idx++;
        }
    }
    
    return results;
}