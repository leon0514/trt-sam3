#ifndef SAM3INFER_HPP__
#define SAM3INFER_HPP__

#include "infer/infer.hpp"
#include "common/memory.hpp"
#include "common/norm.hpp"
#include "common/tensorrt.hpp"
#include "common/device.hpp"
#include <unordered_map>

class Sam3Infer : public InferBase
{
public:
    static std::shared_ptr<Sam3Infer> create_instance(
        const std::string &vision_encoder_path,
        const std::string &text_encoder_path,
        const std::string &decoder_path,
        int gpu_id = 0);

    static std::shared_ptr<Sam3Infer> create_instance(
        const std::string &vision_encoder_path,
        const std::string &text_encoder_path,
        const std::string &geometry_encoder_path,
        const std::string &decoder_path,
        int gpu_id = 0);

    Sam3Infer(
        const std::string &vision_encoder_path,
        const std::string &text_encoder_path,
        const std::string &geometry_encoder_path,
        const std::string &decoder_path,
        int gpu_id = 0);

    virtual ~Sam3Infer() = default;

    bool load_engines();

    void setup_text_inputs(const std::string &input_text,
                           const std::array<int64_t, 32> &input_ids,
                           const std::array<int64_t, 32> &attention_mask) override;

    // 核心实现
    virtual InferResultArray forwards(const std::vector<Sam3Input> &inputs, bool return_mask = false, void *stream = nullptr) override;

private:
    // 内部处理函数
    void preprocess(const Sam3Input &input, int ibatch, void *stream);
    bool encode_image(void *stream);
    
    // 特征扩展：将 Vision Encoder 的结果 (Batch N) 广播到 (Batch M = Total Prompts)
    void expand_vision_features(const std::vector<int>& prompts_per_image, void* stream);

    // 修改后的编码函数，基于总 Prompt 数量
    bool encode_text(const std::vector<Sam3Input> &inputs, int total_prompts, void *stream);
    bool encode_boxes(const std::vector<Sam3Input> &inputs, int total_prompts, int max_boxes, void *stream);
    bool decode(int total_prompts, int prompt_len, void *stream);
    
    // 后处理现在需要知道当前是第几个 global prompt，以及属于哪张原图
    void postprocess(InferResult &image_result, int global_prompt_idx, int image_idx, const std::string &label, float confidence_threshold, bool return_mask, void *stream);

    // 内存与维度管理
    void adjust_memory(int image_batch_size, int total_prompts, int max_boxes);
    void set_binding_dim(std::shared_ptr<TensorRT::Engine> &engine, int binding_index, const std::vector<int> &dims);

private:
    // 配置
    bool isdynamic_model_ = true;
    int input_image_width_ = 1008;
    int input_image_height_ = 1008;
    int gpu_id_ = 0;

    // 状态变量
    std::vector<std::pair<int, int>> original_image_sizes_;
    int num_queries_ = 200;
    int mask_height_ = 288;
    int mask_width_ = 288;

    // 模型路径
    std::string vision_encoder_path_;
    std::string text_encoder_path_;
    std::string geometry_encoder_path_;
    std::string decoder_path_;

    // TRT 引擎
    std::shared_ptr<TensorRT::Engine> vision_encoder_trt_;
    std::shared_ptr<TensorRT::Engine> text_encoder_trt_;
    std::shared_ptr<TensorRT::Engine> decoder_trt_;
    std::shared_ptr<TensorRT::Engine> geometry_encoder_trt_;

    // 数据缓存
    std::unordered_map<std::string, std::pair<std::array<int64_t, 32>, std::array<int64_t, 32>>> text_input_map_;

    // --- 内存管理 ---
    norm_image::Norm preprocess_norm_ = norm_image::Norm::alpha_beta(
        1.0f / 127.5f, -1.0f, norm_image::ChannelType::SwapRB);

    std::vector<int> vision_input_shape_;
    std::vector<int> fpn_feat_0_shape_;
    std::vector<int> text_ids_shape_;
    std::vector<int> geom_box_shape_;

    // Image Batch buffers (Size N)
    tensor::Memory<float> preprocessed_images_;
    std::vector<std::shared_ptr<tensor::Memory<uint8_t>>> original_images_buf_;
    tensor::Memory<float> affine_matrix_;
    tensor::Memory<float> box_affine_matrices_;
    
    tensor::Memory<float> fpn_feat_0_;
    tensor::Memory<float> fpn_feat_1_;
    tensor::Memory<float> fpn_feat_2_;
    tensor::Memory<float> fpn_pos_2_;

    // Expanded Feature buffers for Decoder (Size M = Total Prompts)
    // 我们需要把 vision feature 复制多份给不同的 prompt 使用
    tensor::Memory<float> fpn_feat_0_expanded_;
    tensor::Memory<float> fpn_feat_1_expanded_;
    tensor::Memory<float> fpn_feat_2_expanded_;
    tensor::Memory<float> fpn_pos_2_expanded_;

    // Prompt Batch buffers (Size M)
    tensor::Memory<int64_t> text_input_ids_;
    tensor::Memory<int64_t> text_attention_mask_;

    tensor::Memory<float> geom_boxes_;
    tensor::Memory<int64_t> geom_labels_;

    tensor::Memory<float> text_features_;
    tensor::Memory<bool> text_mask_;

    tensor::Memory<float> geom_features_;
    tensor::Memory<bool> geom_mask_;

    tensor::Memory<float> prompt_features_;
    tensor::Memory<bool> prompt_mask_;

    // Decoder Output (Size M)
    tensor::Memory<float> pred_masks_;
    tensor::Memory<float> pred_boxes_;
    tensor::Memory<float> pred_logits_;
    tensor::Memory<float> presence_logits_;

    // Postprocess (Shared/Reused per single prompt decoding usually, or huge batch)
    // 为了批处理，我们按最大量分配
    tensor::Memory<float> filter_boxes_;
    tensor::Memory<float> filter_scores_;
    tensor::Memory<int> filter_indices_;
    tensor::Memory<int> box_count_;
    tensor::Memory<uint8_t> mask_buffer_;
    tensor::Memory<float> mask_affine_matrix_;
};

#endif // SAM3INFER_HPP__