#include "infer/sam3infer.hpp"
#include "common/timer.hpp"
#include "osd/osd.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <random>

// 模型路径
const std::string VISION_MODEL = "engine-models/vision-encoder.engine";
const std::string TEXT_MODEL = "engine-models/text-encoder.engine";
const std::string DECODER_MODEL = "engine-models/decoder.engine";
const std::string GEOMETRY_ENCODER_PATH = ""; 
const int GPU_ID = 0;

const std::array<int64_t, 32> PERSON_IDS = {49406, 2533, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407};
const std::array<int64_t, 32> PERSON_MASK = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

const std::array<int64_t, 32> HELMET_IDS = {49406, 11122, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407};
const std::array<int64_t, 32> HELMET_MASK = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

const std::array<int64_t, 32> CLOTHES_IDS = {49406, 7080, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407};
const std::array<int64_t, 32> CLOTHES_MASK = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

const std::array<int64_t, 32> HAND_IDS = {49406, 2463, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407};
const std::array<int64_t, 32> HAND_MASK = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// 设置已知的文本 Prompt ID
void setup_data(std::shared_ptr<InferBase> engine)
{
    if (engine)
    {
        engine->setup_text_inputs("person", PERSON_IDS, PERSON_MASK);
        engine->setup_text_inputs("helmet", HELMET_IDS, HELMET_MASK);
        engine->setup_text_inputs("clothes", CLOTHES_IDS, CLOTHES_MASK);
        engine->setup_text_inputs("hand", HAND_IDS, HAND_MASK);
    }
}
int main()
{
    auto engine = load(VISION_MODEL, TEXT_MODEL, GEOMETRY_ENCODER_PATH, DECODER_MODEL, GPU_ID);
    if (!engine) {
        std::cerr << "Failed to load engine!" << std::endl;
        return -1;
    }
    setup_data(engine);

    cv::Mat img = cv::imread("images/smx.jpg"); 

    std::vector<Sam3PromptUnit> prompts;
    prompts.emplace_back("person"); 
    prompts.emplace_back("hand");


    cv::Mat img1 = cv::imread("images/smx.jpg");

    std::vector<Sam3PromptUnit> prompts1;
    prompts1.emplace_back("person");
    prompts1.emplace_back("hand");

    std::vector<Sam3Input> inputs;
    inputs.emplace_back(img, prompts, 0.5);
    inputs.emplace_back(img1, prompts1, 0.5);

    printf("Input constructed: 1 Image with %zu Prompts.\n", prompts.size());

    for (int i = 0; i < 10; i++)
    engine->forwards(inputs);

    nv::EventTimer timer;
    timer.start();
    for (int i = 0; i < 200; i++)
        auto results = engine->forwards(inputs, true);
    float ms = timer.stop();
    printf("Inference 400 images finished in %.2f ms, fps %f.\n", ms, 400 / (ms / 1000) );

    // osd(img, results[0]);
    // cv::imwrite("output/persons.jpg", img);
    // osd(img1, results[1]);
    // cv::imwrite("output/smx.jpg", img1);
    return 0;
}