#ifndef SAM3INFER_HPP__
#define SAM3INFER_HPP__

#include "infer/infer.hpp"
#include "common/memory.hpp"
#include "common/norm.hpp"
#include "common/tensorrt.hpp"
#include "common/device.hpp"
#include <unordered_map>
#include <vector>

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

    bool setup_geometry_input(const cv::Mat &image,
                              const std::string &label,
                              const std::vector<std::pair<std::string, std::array<float, 4>>> &boxes) override;

    // 核心实现
    virtual InferResultArray forwards(const std::vector<Sam3Input> &inputs, bool return_mask = false, void *stream = nullptr) override;
    virtual InferResultArray forwards(const std::vector<Sam3Input> &inputs, const std::string &geom_label, bool return_mask = false, void *stream = nullptr) override;

private:
    // 定义内部结构用于扁平化 Prompt
    struct PromptMeta
    {
        int image_idx;             // 该 Prompt 属于第几张图
        int original_idx;          // 该 Prompt 在原图 vector 中的索引
        const Sam3PromptUnit *ptr; // 指向原始 Prompt 数据的指针
    };

    // 内部处理函数
    void preprocess(const Sam3Input &input, int ibatch, void *stream);
    bool encode_image(int batch_size, void *stream);

    // 修改：Gather 特征，根据当前 Prompt Batch 对应的图片索引，从 Vision 特征中收集数据
    void gather_vision_features(const std::vector<PromptMeta> &batch_prompts, int batch_size, void *stream);

    // 修改后的编码函数，基于当前分批的 Batch Size
    bool encode_text(const std::vector<PromptMeta> &batch_prompts, int batch_size, void *stream);
    bool encode_boxes(const std::vector<PromptMeta> &batch_prompts, int batch_size, int max_boxes, void *stream);
    bool decode(int batch_size, int prompt_len, void *stream);

    // 后处理
    void postprocess(InferResult &image_result, int batch_idx, int image_idx, const std::string &label, float confidence_threshold, bool return_mask, void *stream);

    // 内存初始化 (只调用一次)
    void allocate_memory_once();
    void set_binding_dim(std::shared_ptr<TensorRT::Engine> &engine, int binding_index, const std::vector<int> &dims);

private:
    // 配置
    bool isdynamic_model_ = true;
    int input_image_width_ = 1008;
    int input_image_height_ = 1008;
    int gpu_id_ = 0;

    // --- 批处理限制配置 ---
    // 可根据显存大小调整
    const int max_image_batch_ = 2;       // 这种 Vision Encoder 比较大，限制同时处理的图片数
    const int max_prompt_batch_ = 4;      // Decoder 较小，但显存有限，限制每次 Decode 的 Prompt 数
    const int max_boxes_per_prompt_ = 20; // 预设支持的最大 Box 数量

    // 状态变量
    std::vector<std::pair<int, int>> original_image_sizes_; // Size: max_image_batch_
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

    // Image Batch buffers (Size: max_image_batch_)
    tensor::Memory<float> preprocessed_images_;
    std::vector<std::shared_ptr<tensor::Memory<uint8_t>>> original_images_buf_;
    tensor::Memory<float> affine_matrix_;
    // Mask 后处理需要对应原图的矩阵 (Size: max_image_batch_)
    tensor::Memory<float> mask_affine_matrix_;

    // Vision Encoder Outputs (Size: max_image_batch_)
    tensor::Memory<float> fpn_feat_0_;
    tensor::Memory<float> fpn_feat_1_;
    tensor::Memory<float> fpn_feat_2_;
    tensor::Memory<float> fpn_pos_2_;

    // Decoder Input Buffers (Size: max_prompt_batch_)
    // 这些是从 Vision Output Gather 过来的
    tensor::Memory<float> fpn_feat_0_gather_;
    tensor::Memory<float> fpn_feat_1_gather_;
    tensor::Memory<float> fpn_feat_2_gather_;
    tensor::Memory<float> fpn_pos_2_gather_;

    // Prompt Inputs (Size: max_prompt_batch_)
    tensor::Memory<int64_t> text_input_ids_;
    tensor::Memory<int64_t> text_attention_mask_;

    tensor::Memory<float> geom_boxes_;
    tensor::Memory<int64_t> geom_labels_;

    tensor::Memory<float> text_features_;
    tensor::Memory<bool> text_mask_;

    tensor::Memory<float> geom_features_;
    tensor::Memory<bool> geom_mask_;

    // 用来存储预先设置好的geometry model的结果
    std::unordered_map<std::string, std::shared_ptr<tensor::Memory<float>>> geom_features_cache_;
    std::unordered_map<std::string, std::shared_ptr<tensor::Memory<bool>>> geom_mask_cache_;

    tensor::Memory<float> prompt_features_;
    tensor::Memory<bool> prompt_mask_;

    // Decoder Output (Size: max_prompt_batch_)
    tensor::Memory<float> pred_masks_;
    tensor::Memory<float> pred_boxes_;
    tensor::Memory<float> pred_logits_;
    tensor::Memory<float> presence_logits_;

    // Postprocess (Size: max_prompt_batch_)
    tensor::Memory<float> filter_boxes_;
    tensor::Memory<float> filter_scores_;
    tensor::Memory<int> filter_indices_;
    tensor::Memory<int> box_count_;
    tensor::Memory<uint8_t> mask_buffer_;
    tensor::Memory<float> box_affine_matrices_; // Mask 恢复时针对每个 Box 的矩阵
};

#endif // SAM3INFER_HPP__