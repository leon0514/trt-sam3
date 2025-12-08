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
    // 静态工厂方法，方便 CPM 使用
    static std::shared_ptr<Sam3Infer> create_instance(
        const std::string &vision_encoder_path,
        const std::string &text_encoder_path,
        const std::string &decoder_path,
        int gpu_id = 0,
        float confidence_threshold = 0.5f);

    static std::shared_ptr<Sam3Infer> create_instance(
        const std::string &vision_encoder_path,
        const std::string &text_encoder_path,
        const std::string &geometry_encoder_path,
        const std::string &decoder_path,
        int gpu_id = 0,
        float confidence_threshold = 0.5f);

    Sam3Infer(
        const std::string &vision_encoder_path,
        const std::string &text_encoder_path,
        const std::string &geometry_encoder_path,
        const std::string &decoder_path,
        int gpu_id = 0,
        float confidence_threshold = 0.5f);

    virtual ~Sam3Infer() = default;

    bool load_engines();

    void setup_text_inputs(const std::string &input_text,
                           const std::array<int64_t, 32> &input_ids,
                           const std::array<int64_t, 32> &attention_mask) override;

    // 核心实现
    virtual InferResultArray forwards(const std::vector<Sam3Input> &inputs, void *stream = nullptr) override;

private:
    // 内部处理函数
    void preprocess(const Sam3Input &input, int ibatch, void *stream);
    bool encode_image(void *stream);
    bool encode_text(const std::vector<Sam3Input> &inputs, void *stream);
    bool encode_boxes(const std::vector<Sam3Input> &inputs, int max_boxes, void *stream);
    bool decode(int batch_size, int prompt_len, void *stream);
    void postprocess(InferResult &result, int ibatch, const std::string &label, void *stream);

    // 内存与维度管理
    void adjust_memory(int batch_size, int max_boxes);
    void set_binding_dim(std::shared_ptr<TensorRT::Engine> &engine, int binding_index, const std::vector<int> &dims);

private:
    // 配置
    bool isdynamic_model_ = true;
    int input_image_width_ = 1008;
    int input_image_height_ = 1008;
    int gpu_id_ = 0;
    float confidence_threshold_ = 0.5f;

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

    // --- 内存管理 (Host/Device Tensors) ---
    norm_image::Norm preprocess_norm_ = norm_image::Norm::alpha_beta(
        1.0f / 127.5f, -1.0f, norm_image::ChannelType::SwapRB);

    std::vector<int> vision_input_shape_;
    std::vector<int> fpn_feat_0_shape_;
    std::vector<int> text_ids_shape_;
    std::vector<int> geom_box_shape_;

    tensor::Memory<float> preprocessed_images_;
    std::vector<std::shared_ptr<tensor::Memory<uint8_t>>> original_images_buf_;
    tensor::Memory<float> affine_matrix_;
    tensor::Memory<float> mask_affine_matrix_;

    tensor::Memory<int64_t> text_input_ids_;
    tensor::Memory<int64_t> text_attention_mask_;

    tensor::Memory<float> geom_boxes_;
    tensor::Memory<int64_t> geom_labels_;

    tensor::Memory<float> fpn_feat_0_;
    tensor::Memory<float> fpn_feat_1_;
    tensor::Memory<float> fpn_feat_2_;
    tensor::Memory<float> fpn_pos_2_;

    tensor::Memory<float> text_features_;
    tensor::Memory<bool> text_mask_;

    tensor::Memory<float> geom_features_;
    tensor::Memory<bool> geom_mask_;

    tensor::Memory<float> prompt_features_;
    tensor::Memory<bool> prompt_mask_;

    tensor::Memory<float> pred_masks_;
    tensor::Memory<float> pred_boxes_;
    tensor::Memory<float> pred_logits_;
    tensor::Memory<float> presence_logits_;

    tensor::Memory<float> filter_boxes_;
    tensor::Memory<float> filter_scores_;
    tensor::Memory<int> filter_indices_;
    tensor::Memory<int> box_count_;
    tensor::Memory<uint8_t> mask_buffer_;
};

#endif // SAM3INFER_HPP__