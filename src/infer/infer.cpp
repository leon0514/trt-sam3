#include "infer/infer.hpp"
#include "infer/sam3infer.hpp"

std::shared_ptr<InferBase> load(
    const std::string &vision_encoder_path,
    const std::string &text_encoder_path,
    const std::string &decoder_path,
    int gpu_id,
    float confidence_threshold)
{
    auto engine = std::make_shared<Sam3Infer>(
        vision_encoder_path,
        text_encoder_path,
        decoder_path,
        gpu_id,
        confidence_threshold);
    if (!engine->load_engines())
    {
        return nullptr;
    }
    return engine;
}