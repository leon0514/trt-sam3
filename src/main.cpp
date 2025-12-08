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

    std::vector<cv::Mat> input_images;
    cv::Mat input_image1 = cv::imread("images/persons.jpg");
    cv::Mat input_image2 = cv::imread("images/smx.jpg");
    input_images.push_back(input_image1);
    input_images.push_back(input_image2);

    // [ 49406, 3306, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407 ] >>> encoded.attention_mask
    // [ 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    const std::array<int64_t, 32> input_ids = {49406, 2533, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407};
    const std::array<int64_t, 32> attention_mask = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    infer_engine->setup_text_inputs("person", input_ids, attention_mask);

    nv::EventTimer timer;
    timer.start();
    for (int i = 0; i < 10; ++i)
    {
        auto result = infer_engine->forwards(input_images, "person");
    }
    timer.stop("10 inferences");
    auto results = infer_engine->forwards(input_images, "person");
    for (size_t i = 0; i < results.size(); ++i)
    {
        cv::Mat img_with_osd = input_images[i].clone();
        osd(img_with_osd, results[i], true, 0.04);
        cv::imwrite("output/osd_" + std::to_string(i) + ".jpg", img_with_osd);
    }
    return 0;
}