#ifndef INFER_HPP__
#define INFER_HPP__

#include "common/object.hpp"
#include <iostream>
#include <variant>

using InferResult = object::DetectionBoxArray;
class InferBase
{
public:
    virtual InferResult forward(const cv::Mat &input_image, const std::string &input_text, void *stream = nullptr) = 0;
    virtual void setup_text_inputs(const std::string &input_text, const std::array<int64_t, 32> &input_ids, const std::array<int64_t, 32> &attention_mask) {}
    virtual ~InferBase() = default;
};

std::shared_ptr<InferBase> load(
    const std::string &vision_encoder_path,
    const std::string &text_encoder_path,
    const std::string &decoder_path,
    int gpu_id = 0,
    float confidence_threshold = 0.5f);

#endif // INFER_HPP__