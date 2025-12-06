#include "infer/infer.hpp"
#include "common/timer.hpp"
#include "osd/osd.hpp"
#include <opencv2/opencv.hpp>

int main()
{
    const std::string vision_encoder_path = "model/vision-encoder.engine";
    const std::string text_encoder_path = "model/text-encoder.engine";
    const std::string decoder_path = "model/decoder.engine";

    auto infer_engine = load(vision_encoder_path, text_encoder_path, decoder_path, 0, 0.5f);
    if (!infer_engine)
    {
        std::cerr << "Failed to load inference engine." << std::endl;
        return -1;
    }
    printf("Inference engine loaded successfully.\n");

    cv::Mat input_image = cv::imread("images/persons.jpg");

    // [ 49406, 3306, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407 ] >>> encoded.attention_mask
    // [ 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

    const std::array<int64_t, 32> input_ids = {49406, 2533, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407};
    const std::array<int64_t, 32> attention_mask = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    infer_engine->setup_text_inputs("person", input_ids, attention_mask);

    for (int i = 0; i < 5; ++i)
    {
        auto result = infer_engine->forward(input_image, "person");
        printf("Iteration %d: Detected %zu objects.\n", i + 1, result.size());
    }

    nv::EventTimer timer;
    timer.start();
    for (int i = 0; i < 10; ++i)
    {
        auto result = infer_engine->forward(input_image, "person");
    }
    timer.stop("10 inferences");
    auto result = infer_engine->forward(input_image, "person");
    printf("Detected %zu objects.\n", result.size());

    osd(input_image, result);
    cv::imwrite("output/output_osd.jpg", input_image);
    return 0;
}