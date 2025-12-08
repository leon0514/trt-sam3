#ifndef SAM3INFER_HPP__
#define SAM3INFER_HPP__

#include "common/object.hpp"
#include "common/memory.hpp"
#include "common/norm.hpp"
#include "common/tensorrt.hpp"
#include "common/device.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <variant>
#include <unordered_map>
#include <array>
#include "infer/infer.hpp"

/**
 * forward step
 * 1. preprocess images : resize + normalize (image / 127.5 - 1.0) + transpose HWC->CHW
 * 2. encode images : vision encoder
 * 3. encode texts : text encoder
 * 4. decode : decoder
 * 5. postprocess
 */

class Sam3Infer : public InferBase
{
public:
    Sam3Infer(
        const std::string &vision_encoder_path,
        const std::string &text_encoder_path,
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

    bool load_engines();
    void setup_text_inputs(const std::string &input_text, const std::array<int64_t, 32> &input_ids, const std::array<int64_t, 32> &attention_mask) override;
    virtual InferResult forward(const cv::Mat &input_image, const std::string &input_text, void *stream = nullptr) override;
    virtual InferResultArray forwards(const std::vector<cv::Mat> &input_images, const std::string &input_text, void *stream = nullptr) override;
    InferResult forward(const cv::Mat &input_image, 
                    const std::string &input_text, 
                    const std::vector<BoxPrompt> &boxes,
                    void *stream = nullptr) override;

    virtual InferResultArray forwards(const std::vector<cv::Mat> &input_images, 
                          const std::string &input_text, 
                          const std::vector<BoxPrompt> &boxes,
                          void *stream = nullptr) override;
    virtual ~Sam3Infer() = default;

private:
    void preprocess(const cv::Mat &input_image, int ibatch, void *stream = nullptr);
    bool encode_image(void *stream = nullptr);
    bool encode_text(const std::string &input_text, void *stream = nullptr);
    bool encode_boxes(const std::vector<BoxPrompt> &boxes, void *stream = nullptr);
    bool decode(void *stream = nullptr);
    void postprocess(InferResult &result, int ibatch, const std::string &label, void *stream = nullptr);
    void adjust_memory(int batch_size);

private:
    bool isdynamic_model_ = true;
    int input_image_width_ = 1008;
    int input_image_height_ = 1008;

    std::vector<std::pair<int, int>> original_image_sizes_;

    int num_queries_ = 200;
    int mask_height_ = 288;
    int mask_width_ = 288;

private:
    norm_image::Norm preprocess_norm_ = norm_image::Norm::alpha_beta(
        1.0f / 127.5f,
        -1.0f,
        norm_image::ChannelType::SwapRB);

    std::vector<int> vision_encoder_input_images_shape_;
    std::vector<int> text_encoder_input_input_ids_shape_;
    std::vector<int> text_encoder_input_attention_mask_shape_;
    std::vector<int> geometry_encoder_input_input_boxes_shape_;
    std::vector<int> geometry_encoder_input_input_boxes_labels_shape_;
    std::vector<int> geometry_encoder_input_fpn_feat_2_shape_;
    std::vector<int> geometry_encoder_input_fpn_pos_2_shape_;
    std::vector<int> decoder_input_fpn_feat_0_shape_;
    std::vector<int> decoder_input_fpn_feat_1_shape_;
    std::vector<int> decoder_input_fpn_feat_2_shape_;
    std::vector<int> decoder_input_fpn_pos_2_shape_;
    std::vector<int> decoder_input_prompt_features_shape_;
    std::vector<int> decoder_input_prompt_mask_shape_;

    std::vector<int> vision_encoder_output_fpn_feat_0_shape_;
    std::vector<int> vision_encoder_output_fpn_feat_1_shape_;
    std::vector<int> vision_encoder_output_fpn_feat_2_shape_;
    std::vector<int> vision_encoder_output_fpn_pos_2_shape_;

    std::vector<int> text_encoder_output_text_features_shape_;
    std::vector<int> text_encoder_output_text_mask_shape_;

    std::vector<int> geometry_encoder_output_geometry_features_shape_;
    std::vector<int> geometry_encoder_output_geometry_mask_shape_;

    std::vector<int> decoder_output_pred_masks_shape_;
    std::vector<int> decoder_output_pred_boxes_shape_;
    std::vector<int> decoder_output_pred_logits_shape_;
    std::vector<int> decoder_output_presence_logits_shape_;

    tensor::Memory<float> preprocessed_images_tensor_;
    std::vector<std::shared_ptr<tensor::Memory<uint8_t>>> original_images_tensor_;

    // Image Encoder input tensors
    /**
     *  image                  [batch, 3, 1008, 1008]    FLOAT
     */
    tensor::Memory<float> vision_encoder_input_images_tensor_;
    // Image Encoder output tensors
    /**
     * Outputs:
        fpn_feat_0            [batch, 256, 288, 288]    FLOAT
        fpn_feat_1            [batch, 256, 144, 144]    FLOAT
        fpn_feat_2            [batch, 256, 72, 72]      FLOAT
        fpn_pos_2             [batch, 256, 72, 72]      FLOAT
     */
    tensor::Memory<float> vision_encoder_output_fpn_feat_0_tensor_;
    tensor::Memory<float> vision_encoder_output_fpn_feat_1_tensor_;
    tensor::Memory<float> vision_encoder_output_fpn_feat_2_tensor_;
    tensor::Memory<float> vision_encoder_output_fpn_pos_2_tensor_;

    /** Text Encoder input tensors
     *  input_ids             [batch, 32]               INT64
        attention_mask        [batch, 32]               INT64
     */
    tensor::Memory<int64_t> text_encoder_input_input_ids_tensor_;
    tensor::Memory<int64_t> text_encoder_input_attention_mask_tensor_;
    /** Text Encoder output tensors:
        text_features         [batch, 32, 256]          FLOAT
        text_mask             [batch, 32]               BOOL
     */
    tensor::Memory<float> text_encoder_output_text_features_tensor_;
    tensor::Memory<bool> text_encoder_output_text_mask_tensor_;

    // decode input tensors
    /**
     *  fpn_feat_0            [batch, 256, 288, 288]    FLOAT
        fpn_feat_1            [batch, 256, 144, 144]    FLOAT
        fpn_feat_2            [batch, 256, 72, 72]      FLOAT
        fpn_pos_2             [batch, 256, 72, 72]      FLOAT
        prompt_features       [batch, prompt_len, 256]  FLOAT
        prompt_mask           [batch, prompt_len]       BOOL
     */
    // encode_image_output_fpn_feat_0_tensor_
    // encode_image_output_fpn_feat_1_tensor_
    // encode_image_output_fpn_feat_2_tensor_
    // encode_image_output_fpn_pos_2_tensor_
    // encode_text_output_text_features_tensor_
    // encode_text_output_text_mask_tensor_

    tensor::Memory<float> decoder_input_fpn_feat_0_tensor_;
    tensor::Memory<float> decoder_input_fpn_feat_1_tensor_;
    tensor::Memory<float> decoder_input_fpn_feat_2_tensor_;
    tensor::Memory<float> decoder_input_fpn_pos_2_tensor_;
    // 只有text encoder的时, decoder_input_prompt_features_tensor_ 和 decoder_input_prompt_mask_tensor_等于encode_text输出
    tensor::Memory<float> decoder_input_prompt_features_tensor_;
    tensor::Memory<bool> decoder_input_prompt_mask_tensor_;

    tensor::Memory<float> geometry_encoder_input_input_boxes_tensor_;
    tensor::Memory<int64_t> geometry_encoder_input_input_boxes_labels_tensor_;
    tensor::Memory<float> geometry_encoder_input_fpn_feat_2_tensor_; // = vision_encoder_output_fpn_feat_2_tensor_
    tensor::Memory<float> geometry_encoder_input_fpn_pos_2_tensor_;  // = vision_encoder_output_fpn_pos_2_tensor_
    tensor::Memory<float> geometry_encoder_output_geometry_features_tensor_;
    tensor::Memory<bool> geometry_encoder_output_geometry_mask_tensor_;

    // decode output tensors
    /**
     *  pred_masks : {-1 x 200 x 288 x 288} [float32]
        pred_boxes : {-1 x 200 x 4} [float32]
        pred_logits : {-1 x 200} [float32]
        presence_logits : {-1 x 1} [float32]
     */
    tensor::Memory<float> decoder_output_pred_masks_tensor_;
    tensor::Memory<float> decoder_output_pred_boxes_tensor_;
    tensor::Memory<float> decoder_output_pred_logits_tensor_;
    tensor::Memory<float> decoder_output_presence_logits_tensor_;

    // postprocess intermediate tensors
    tensor::Memory<float> filter_boxes_tensor_;
    tensor::Memory<float> filter_scores_tensor_;
    tensor::Memory<int> filter_indices_tensor_;
    tensor::Memory<int> box_count_;

    tensor::Memory<uint8_t> mask_buffer_;

    tensor::Memory<float> affine_matrix_tensor_;
    tensor::Memory<float> mask_affine_matrix_tensor_;

private:
    float confidence_threshold_;
    std::string vision_encoder_path_;
    std::string text_encoder_path_;
    std::string geometry_encoder_path_;
    std::string decoder_path_;
    int gpu_id_;

    // text : text_features, text_mask 先硬编码直接存储，后续可以改成动态生成
    /**
     *  >>> from tokenizers import Tokenizer
        >>> token_path = "facebook-sam3/tokenizer.json"
        >>> tokenizer = Tokenizer.from_file(token_path)
        >>> tokenizer.enable_padding(length=32, pad_id=49407)
        >>> tokenizer.enable_truncation(max_length=32)
        >>> text = 'hello'
        >>> encoded = tokenizer.encode(text)
     */
    std::unordered_map<std::string, std::pair<std::array<int64_t, 32>, std::array<int64_t, 32>>> text_input_map_;

    std::shared_ptr<TensorRT::Engine> vision_encoder_trt_;
    std::shared_ptr<TensorRT::Engine> text_encoder_trt_;
    std::shared_ptr<TensorRT::Engine> decoder_trt_;
    std::shared_ptr<TensorRT::Engine> geometry_encoder_trt_;
};

#endif // SAM3INFER_HPP__