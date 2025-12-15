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
const std::string GEOMETRY_ENCODER_PATH = "engine-models/geometry-encoder.engine";
const int GPU_ID = 1;

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

void speed_test()
{
    auto engine = load(VISION_MODEL, TEXT_MODEL, GEOMETRY_ENCODER_PATH, DECODER_MODEL, GPU_ID);
    // prepare inputs
    std::vector<std::string> image_paths = {"images/smx.jpg"};

    std::vector<Sam3Input> inputs;
    for (const auto &image_path : image_paths)
    {
        cv::Mat img = cv::imread(image_path);
        std::vector<Sam3PromptUnit> prompts;
        prompts.emplace_back("person");
        prompts.emplace_back("hand");
        inputs.emplace_back(img, prompts, 0.5);
    }

    nv::EventTimer timer;
    timer.start();
    for (int i = 0; i < 1000; i++)
        engine->forwards(inputs);
    float ms = timer.stop();
    printf("Inference 1000 images finished in %.2f ms, fps %f.\n", ms, 1000 / (ms / 1000));
}

void test_text_prompt()
{
    auto engine = load(VISION_MODEL, TEXT_MODEL, GEOMETRY_ENCODER_PATH, DECODER_MODEL, GPU_ID);
    setup_data(engine);
    std::vector<std::string> image_paths = {"images/smx.jpg"};

    std::vector<Sam3Input> inputs;
    for (const auto &image_path : image_paths)
    {
        cv::Mat img = cv::imread(image_path);
        std::vector<Sam3PromptUnit> prompts;
        prompts.emplace_back("person");
        prompts.emplace_back("hand");
        inputs.emplace_back(img, prompts, 0.5);
    }
    auto results = engine->forwards(inputs, true);

    for (size_t i = 0; i < results.size(); i++)
    {
        std::string image_path = image_paths[i];
        cv::Mat img = inputs[i].image;
        osd(img, results[i]);
        std::string output_path = "output/" + image_path;
        cv::imwrite(output_path, img);
    }
}

void test_box_prompt()
{
    auto engine = load(VISION_MODEL, TEXT_MODEL, GEOMETRY_ENCODER_PATH, DECODER_MODEL, GPU_ID);
    setup_data(engine);
    std::vector<std::string> image_paths = {"images/3.jpg", "images/7.jpg"};

    cv::Mat image = cv::imread(image_paths[0]);
    std::vector<std::pair<std::string, std::array<float, 4>>> boxes;
    std::array<float, 4> pos = {169, 40, 420, 176};
    std::array<float, 4> pos1 = {349, 197, 926, 713};
    boxes.push_back(std::make_pair("pos", pos));
    boxes.push_back(std::make_pair("neg", pos1));
    engine->setup_geometry_input(image, "xxxx", boxes);

    std::vector<Sam3Input> inputs;
    for (const auto &image_path : image_paths)
    {
        cv::Mat img = cv::imread(image_path);
        std::vector<Sam3PromptUnit> prompts;
        inputs.emplace_back(img, prompts, 0.5);
    }
    auto results = engine->forwards(inputs, "xxxx", true);
    for (size_t i = 0; i < results.size(); i++)
    {
        std::string image_path = image_paths[i];
        std::cout << "image_path : " << image_path << std::endl;
        cv::Mat img = inputs[i].image;
        osd(img, results[i]);
        for (const auto &r : results[i])
            std::cout << r << std::endl;
        std::string output_path = "output/" + image_path;
        cv::imwrite(output_path, img);
    }
}

int main()
{
    // speed_test();
    // test_text_prompt();
    test_box_prompt();
    return 0;
}