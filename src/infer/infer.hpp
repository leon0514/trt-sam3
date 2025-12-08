#ifndef INFER_HPP__
#define INFER_HPP__

#include "infer/sam3type.hpp"
#include "common/object.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <array>

using InferResult = object::DetectionBoxArray;
using InferResultArray = std::vector<object::DetectionBoxArray>;

class InferBase
{
public:
    virtual ~InferBase() = default;

    // -------------------------------------------------------
    // 统一入口: 无论单图、多图、带框、带字，都走这里
    // -------------------------------------------------------

    // 批量推理 (Core API)
    virtual InferResultArray forwards(const std::vector<Sam3Input> &inputs, void *stream = nullptr) = 0;

    // 单个推理 (Wrapper)
    virtual InferResult forward(const Sam3Input &input, void *stream = nullptr)
    {
        return forwards({input}, stream)[0];
    }

    // 预设文本 Token (用于 Tokenizer 缓存)
    virtual void setup_text_inputs(const std::string &input_text,
                                   const std::array<int64_t, 32> &input_ids,
                                   const std::array<int64_t, 32> &attention_mask) {}
};

// 工厂函数声明
std::shared_ptr<InferBase> load(
    const std::string &vision_encoder_path,
    const std::string &text_encoder_path,
    const std::string &geometry_encoder_path,
    const std::string &decoder_path,
    int gpu_id = 0,
    float confidence_threshold = 0.5f);

#endif // INFER_HPP__