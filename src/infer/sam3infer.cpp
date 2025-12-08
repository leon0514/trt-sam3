#include "infer/sam3infer.hpp"
#include "common/affine.hpp"
#include "common/image.hpp"
#include "kernels/process_kernel_warp.hpp"
#include "kernels/postprocess.cuh"
#include "common/createObject.hpp"

// 全局 load 函数实现
std::shared_ptr<InferBase> load(
    const std::string &vision_encoder_path,
    const std::string &text_encoder_path,
    const std::string &decoder_path,
    int gpu_id,
    float confidence_threshold)
{
    return Sam3Infer::create_instance(vision_encoder_path, text_encoder_path, decoder_path, gpu_id, confidence_threshold);
}

// 静态工厂
std::shared_ptr<Sam3Infer> Sam3Infer::create_instance(
    const std::string &vision_encoder_path,
    const std::string &text_encoder_path,
    const std::string &decoder_path,
    int gpu_id,
    float confidence_threshold)
{
    std::string geom_path = "";
    auto instance = std::make_shared<Sam3Infer>(
        vision_encoder_path, text_encoder_path, geom_path, decoder_path, gpu_id, confidence_threshold);

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
    int gpu_id,
    float confidence_threshold)
{
    auto instance = std::make_shared<Sam3Infer>(
        vision_encoder_path, text_encoder_path, geometry_encoder_path, decoder_path, gpu_id, confidence_threshold);

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
    int gpu_id,
    float confidence_threshold)
    : InferBase(),
      vision_encoder_path_(vision_encoder_path),
      text_encoder_path_(text_encoder_path),
      geometry_encoder_path_(geometry_encoder_path),
      decoder_path_(decoder_path),
      gpu_id_(gpu_id),
      confidence_threshold_(confidence_threshold)
{
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
        engine->print(name);
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

    return true;
}

void Sam3Infer::setup_text_inputs(const std::string &input_text, const std::array<int64_t, 32> &input_ids, const std::array<int64_t, 32> &attention_mask)
{
    text_input_map_[input_text] = std::make_pair(input_ids, attention_mask);
}

void Sam3Infer::adjust_memory(int batch_size, int max_boxes)
{
    // 输入部分
    affine_matrix_.cpu(batch_size * 6);
    affine_matrix_.gpu(batch_size * 6);
    mask_affine_matrix_.cpu(batch_size * 6);
    mask_affine_matrix_.gpu(batch_size * 6);

    if ((int)original_images_buf_.size() < batch_size)
    {
        for (int i = original_images_buf_.size(); i < batch_size; ++i)
            original_images_buf_.emplace_back(std::make_shared<tensor::Memory<uint8_t>>());
    }

    preprocessed_images_.gpu(batch_size * 3 * input_image_height_ * input_image_width_);

    // 文本输入
    size_t text_in_sz = batch_size * text_ids_shape_[1];
    text_input_ids_.cpu(text_in_sz);
    text_input_ids_.gpu(text_in_sz);
    text_attention_mask_.cpu(text_in_sz);
    text_attention_mask_.gpu(text_in_sz);

    // 中间特征 (Vision)
    size_t feat_0_sz = batch_size * fpn_feat_0_shape_[1] * fpn_feat_0_shape_[2] * fpn_feat_0_shape_[3];
    fpn_feat_0_.gpu(feat_0_sz);
    fpn_feat_1_.gpu(feat_0_sz / 4);
    fpn_feat_2_.gpu(feat_0_sz / 16);
    fpn_pos_2_.gpu(feat_0_sz / 16);

    // 中间特征 (Text)
    text_features_.gpu(text_in_sz * 256);
    text_mask_.gpu(text_in_sz);

    // Geometry
    bool use_geom = (max_boxes > 0 && geometry_encoder_trt_);
    if (use_geom)
    {
        size_t box_sz = batch_size * max_boxes * 4;
        geom_boxes_.cpu(box_sz);
        geom_boxes_.gpu(box_sz);
        size_t lbl_sz = batch_size * max_boxes;
        geom_labels_.cpu(lbl_sz);
        geom_labels_.gpu(lbl_sz);

        size_t geom_feat_sz = batch_size * (max_boxes + 1) * 256;
        geom_features_.gpu(geom_feat_sz);
        geom_mask_.gpu(batch_size * (max_boxes + 1));
    }

    // Decoder Input
    size_t total_prompt_len = text_ids_shape_[1] + (use_geom ? (max_boxes + 1) : 0);
    prompt_features_.gpu(batch_size * total_prompt_len * 256);
    prompt_mask_.gpu(batch_size * total_prompt_len);

    // Decoder Output
    pred_masks_.gpu(batch_size * num_queries_ * mask_height_ * mask_width_);
    pred_boxes_.gpu(batch_size * num_queries_ * 4);
    pred_logits_.gpu(batch_size * num_queries_);
    presence_logits_.gpu(batch_size * 1);

    // Postprocess
    filter_boxes_.cpu(1 * num_queries_ * 4);
    filter_boxes_.gpu(1 * num_queries_ * 4);
    filter_scores_.cpu(1 * num_queries_);
    filter_scores_.gpu(1 * num_queries_);
    filter_indices_.cpu(1 * num_queries_);
    filter_indices_.gpu(1 * num_queries_);
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
    original_image_sizes_.emplace_back(img_tensor.width, img_tensor.height);

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

bool Sam3Infer::encode_text(const std::vector<Sam3Input> &inputs, void *stream)
{
    int batch_size = inputs.size();
    int seq_len = 32;
    int64_t *h_ids = text_input_ids_.cpu();
    int64_t *h_mask = text_attention_mask_.cpu();

    // 默认空 Token
    std::array<int64_t, 32> def_ids;
    def_ids.fill(49407);
    std::array<int64_t, 32> def_mask = {0};
    def_mask[0] = 1;

    for (int i = 0; i < batch_size; ++i)
    {
        std::string txt = inputs[i].text_prompt;
        const int64_t *src_ids = def_ids.data();
        const int64_t *src_mask = def_mask.data();

        if (text_input_map_.count(txt))
        {
            src_ids = text_input_map_[txt].first.data();
            src_mask = text_input_map_[txt].second.data();
        }
        else if (!txt.empty())
        {
            // 如果没找到且不为空，打印警告，实际项目可能需要在线 Tokenizer
            printf("[Warning] Text prompt '%s' not cached. Using default.\n", txt.c_str());
        }

        memcpy(h_ids + i * seq_len, src_ids, seq_len * sizeof(int64_t));
        memcpy(h_mask + i * seq_len, src_mask, seq_len * sizeof(int64_t));
    }

    cudaStream_t s = (cudaStream_t)stream;
    cudaMemcpyAsync(text_input_ids_.gpu(), h_ids, text_input_ids_.gpu_bytes(), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(text_attention_mask_.gpu(), h_mask, text_attention_mask_.gpu_bytes(), cudaMemcpyHostToDevice, s);

    return text_encoder_trt_->forward({{"input_ids", text_input_ids_.gpu()},
                                       {"attention_mask", text_attention_mask_.gpu()},
                                       {"text_features", text_features_.gpu()},
                                       {"text_mask", text_mask_.gpu()}},
                                      s);
}

bool Sam3Infer::encode_boxes(const std::vector<Sam3Input> &inputs, int max_boxes, void *stream)
{
    if (!geometry_encoder_trt_ || max_boxes == 0)
        return true;

    int batch_size = inputs.size();
    float *h_boxes = geom_boxes_.cpu();
    int64_t *h_labels = geom_labels_.cpu();

    // 清零 (Padding)
    memset(h_boxes, 0, geom_boxes_.cpu_bytes());
    memset(h_labels, 0, geom_labels_.cpu_bytes());

    for (int b = 0; b < batch_size; ++b)
    {
        float iw = (float)original_image_sizes_[b].first;
        float ih = (float)original_image_sizes_[b].second;
        const auto &boxes = inputs[b].box_prompts;

        for (size_t i = 0; i < boxes.size(); ++i)
        {
            const auto &box = boxes[i];
            int64_t label = (box.first == "pos") ? 1 : 0; // 0 for neg/padding

            float x1 = box.second[0], y1 = box.second[1];
            float x2 = box.second[2], y2 = box.second[3];

            float cx = (x1 + x2) * 0.5f / iw;
            float cy = (y1 + y2) * 0.5f / ih;
            float w = (x2 - x1) / iw;
            float h = (y2 - y1) / ih;

            int idx_base = b * max_boxes + i;
            h_labels[idx_base] = label;
            h_boxes[idx_base * 4 + 0] = cx;
            h_boxes[idx_base * 4 + 1] = cy;
            h_boxes[idx_base * 4 + 2] = w;
            h_boxes[idx_base * 4 + 3] = h;
        }
    }

    cudaStream_t s = (cudaStream_t)stream;
    cudaMemcpyAsync(geom_boxes_.gpu(), geom_boxes_.cpu(), geom_boxes_.gpu_bytes(), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(geom_labels_.gpu(), geom_labels_.cpu(), geom_labels_.gpu_bytes(), cudaMemcpyHostToDevice, s);

    return geometry_encoder_trt_->forward({{"input_boxes", geom_boxes_.gpu()},
                                           {"input_boxes_labels", geom_labels_.gpu()},
                                           {"fpn_feat_2", fpn_feat_2_.gpu()},
                                           {"fpn_pos_2", fpn_pos_2_.gpu()},
                                           {"geometry_features", geom_features_.gpu()},
                                           {"geometry_mask", geom_mask_.gpu()}},
                                          s);
}

bool Sam3Infer::decode(int batch_size, int prompt_len, void *stream)
{
    // Feature concatenation handled by CUDA Memcpy
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

    for (int i = 0; i < batch_size; ++i)
    {
        size_t prompt_off = i * prompt_len * feat_sz;
        size_t prompt_m_off = i * prompt_len * mask_sz;

        // Copy Text
        cudaMemcpyAsync(d_prompt + prompt_off, d_text + i * text_len * feat_sz, text_len * feat_sz, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(d_prompt_m + prompt_m_off, d_text_m + i * text_len * mask_sz, text_len * mask_sz, cudaMemcpyDeviceToDevice, s);

        // Copy Geometry
        if (prompt_len > text_len)
        {
            size_t geom_len = prompt_len - text_len;
            cudaMemcpyAsync(d_prompt + prompt_off + text_len * feat_sz, d_geom + i * geom_len * feat_sz, geom_len * feat_sz, cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(d_prompt_m + prompt_m_off + text_len * mask_sz, d_geom_m + i * geom_len * mask_sz, geom_len * mask_sz, cudaMemcpyDeviceToDevice, s);
        }
    }

    return decoder_trt_->forward({{"fpn_feat_0", fpn_feat_0_.gpu()},
                                  {"fpn_feat_1", fpn_feat_1_.gpu()},
                                  {"fpn_feat_2", fpn_feat_2_.gpu()},
                                  {"fpn_pos_2", fpn_pos_2_.gpu()},
                                  {"prompt_features", prompt_features_.gpu()},
                                  {"prompt_mask", prompt_mask_.gpu()},
                                  {"pred_masks", pred_masks_.gpu()},
                                  {"pred_boxes", pred_boxes_.gpu()},
                                  {"pred_logits", pred_logits_.gpu()},
                                  {"presence_logits", presence_logits_.gpu()}},
                                 s);
}

void Sam3Infer::postprocess(InferResult &result, int ibatch, const std::string &label, void *stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    cudaMemsetAsync(box_count_.gpu(), 0, box_count_.cpu_bytes(), s);

    sam3_postprocess_plane(
        pred_masks_.gpu() + ibatch * num_queries_ * mask_height_ * mask_width_,
        pred_boxes_.gpu() + ibatch * num_queries_ * 4,
        pred_logits_.gpu() + ibatch * num_queries_,
        presence_logits_.gpu() + ibatch,
        filter_boxes_.gpu(), filter_indices_.gpu(), filter_scores_.gpu(), box_count_.gpu(),
        num_queries_, mask_height_, mask_width_,
        original_image_sizes_[ibatch].first, original_image_sizes_[ibatch].second,
        confidence_threshold_, s);

    cudaMemcpyAsync(box_count_.cpu(), box_count_.gpu(), sizeof(int), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    int count = *box_count_.cpu();

    if (count > 0)
    {
        cudaMemcpyAsync(filter_boxes_.cpu(), filter_boxes_.gpu(), count * 4 * sizeof(float), cudaMemcpyDeviceToHost, s);
        cudaMemcpyAsync(filter_scores_.cpu(), filter_scores_.gpu(), count * sizeof(float), cudaMemcpyDeviceToHost, s);
        cudaMemcpyAsync(filter_indices_.cpu(), filter_indices_.gpu(), count * sizeof(int), cudaMemcpyDeviceToHost, s);

        size_t mask_buf_sz = count * original_image_sizes_[ibatch].second * original_image_sizes_[ibatch].first;
        mask_buffer_.gpu(mask_buf_sz);
        mask_buffer_.cpu(mask_buf_sz);

        affine::ResizeMatrix matrix;
        matrix.compute(std::make_tuple(mask_width_, mask_height_),
                       std::make_tuple(original_image_sizes_[ibatch].first, original_image_sizes_[ibatch].second));
        memcpy(mask_affine_matrix_.cpu(), matrix.d2i, sizeof(matrix.d2i));
        cudaMemcpyAsync(mask_affine_matrix_.gpu(), mask_affine_matrix_.cpu(), sizeof(matrix.d2i), cudaMemcpyHostToDevice, s);

        for (int i = 0; i < count; ++i)
        {
            int idx = filter_indices_.cpu()[i];
            float *src = pred_masks_.gpu() + (ibatch * num_queries_ + idx) * mask_height_ * mask_width_;
            uint8_t *dst = mask_buffer_.gpu() + i * original_image_sizes_[ibatch].second * original_image_sizes_[ibatch].first;

            warp_affine_bilinear_single_channel_mask_plane(
                src, mask_width_, mask_width_, mask_height_,
                dst, original_image_sizes_[ibatch].first, original_image_sizes_[ibatch].second,
                mask_affine_matrix_.gpu(), 0, s);
        }

        cudaMemcpyAsync(mask_buffer_.cpu(), mask_buffer_.gpu(), mask_buf_sz, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);

        for (int i = 0; i < count; ++i)
        {
            float *b = filter_boxes_.cpu() + i * 4;
            cv::Mat bin_mask(original_image_sizes_[ibatch].second, original_image_sizes_[ibatch].first, CV_8U,
                             mask_buffer_.cpu() + i * original_image_sizes_[ibatch].second * original_image_sizes_[ibatch].first);
            result.push_back(object::createSegmentationBox(b[0], b[1], b[2], b[3], bin_mask, filter_scores_.cpu()[i], -1, label));
        }
    }
}

InferResultArray Sam3Infer::forwards(const std::vector<Sam3Input> &inputs, void *stream)
{
    original_image_sizes_.clear();
    if (inputs.empty())
        return {};

    int batch_size = inputs.size();
    int max_boxes = 0;
    for (const auto &inp : inputs)
    {
        if ((int)inp.box_prompts.size() > max_boxes)
            max_boxes = (int)inp.box_prompts.size();
    }

    bool use_geom = !geometry_encoder_path_.empty() && max_boxes > 0;
    int geom_len = use_geom ? (max_boxes + 1) : 0;
    int prompt_len = text_ids_shape_[1] + geom_len;

    AutoDevice device_guard(gpu_id_);

    if (isdynamic_model_)
    {
        set_binding_dim(vision_encoder_trt_, 0, {batch_size, 3, input_image_height_, input_image_width_});
        set_binding_dim(text_encoder_trt_, 0, {batch_size, text_ids_shape_[1]});
        set_binding_dim(text_encoder_trt_, 1, {batch_size, text_ids_shape_[1]});

        set_binding_dim(decoder_trt_, 0, {batch_size, fpn_feat_0_shape_[1], fpn_feat_0_shape_[2], fpn_feat_0_shape_[3]});
        set_binding_dim(decoder_trt_, 1, {batch_size, fpn_feat_0_shape_[1], fpn_feat_0_shape_[2] / 2, fpn_feat_0_shape_[3] / 2});
        set_binding_dim(decoder_trt_, 2, {batch_size, fpn_feat_0_shape_[1], fpn_feat_0_shape_[2] / 4, fpn_feat_0_shape_[3] / 4});
        set_binding_dim(decoder_trt_, 3, {batch_size, fpn_feat_0_shape_[1], fpn_feat_0_shape_[2] / 4, fpn_feat_0_shape_[3] / 4});
        set_binding_dim(decoder_trt_, 4, {batch_size, prompt_len, 256});
        set_binding_dim(decoder_trt_, 5, {batch_size, prompt_len});

        if (use_geom)
        {
            set_binding_dim(geometry_encoder_trt_, 0, {batch_size, max_boxes, 4});
            set_binding_dim(geometry_encoder_trt_, 1, {batch_size, max_boxes});
            // Geometry output fixed sizes
            set_binding_dim(geometry_encoder_trt_, 2, {batch_size, 256, 72, 72});
            set_binding_dim(geometry_encoder_trt_, 3, {batch_size, 256, 72, 72});
        }
    }

    adjust_memory(batch_size, use_geom ? max_boxes : 0);

    for (int i = 0; i < batch_size; ++i)
        preprocess(inputs[i], i, stream);

    bool ok = true;
    ok &= encode_image(stream);
    ok &= encode_text(inputs, stream);
    if (use_geom)
        ok &= encode_boxes(inputs, max_boxes, stream);
    if (ok)
        ok &= decode(batch_size, prompt_len, stream);

    if (!ok)
    {
        std::cerr << "Forward failed" << std::endl;
        return InferResultArray(batch_size);
    }

    InferResultArray results;
    for (int i = 0; i < batch_size; ++i)
    {
        InferResult res;
        std::string lbl = inputs[i].text_prompt.empty() ? "object" : inputs[i].text_prompt;
        postprocess(res, i, lbl, stream);
        results.push_back(res);
    }
    return results;
}