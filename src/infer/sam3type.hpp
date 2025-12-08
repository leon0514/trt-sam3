#ifndef SAM3TYPE_HPP__
#define SAM3TYPE_HPP__

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <array>
#include <utility>

// 定义 BoxPrompt: <Label("pos"/"neg"), {x1, y1, x2, y2}>
using BoxPrompt = std::pair<std::string, std::array<float, 4>>;

// 统一输入结构体
struct Sam3Input
{
    cv::Mat image;                      // 必须: 输入图像
    std::string text_prompt;            // 可选: 文本提示 (用于查找 Tokenizer ID)
    std::vector<BoxPrompt> box_prompts; // 可选: 框提示

    // 构造函数便于使用
    Sam3Input() = default;
    Sam3Input(const cv::Mat &img, const std::string &txt = "", const std::vector<BoxPrompt> &boxes = {})
        : image(img), text_prompt(txt), box_prompts(boxes) {}
};

#endif // SAM3TYPE_HPP__