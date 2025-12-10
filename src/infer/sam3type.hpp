#ifndef SAM3TYPE_HPP__
#define SAM3TYPE_HPP__

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <array>
#include <utility>

// 定义 BoxPrompt: <Label("pos"/"neg"), {x1, y1, x2, y2}>
using BoxPrompt = std::pair<std::string, std::array<float, 4>>;

// 单个提示词单元：包含一段文字和可选的一组框
struct Sam3PromptUnit {
    std::string text;
    std::vector<BoxPrompt> boxes;
    Sam3PromptUnit() = default;
    Sam3PromptUnit(const std::string& t, const std::vector<BoxPrompt>& b = {}) 
        : text(t), boxes(b) {}
};

// 统一输入结构体
struct Sam3Input
{
    float confidence_threshold;
    cv::Mat image;                          // 必须: 输入图像
    std::vector<Sam3PromptUnit> prompts;    // 必须: 该图对应的所有提示词列表

    Sam3Input() = default;
    Sam3Input(const cv::Mat &img, const std::vector<Sam3PromptUnit>& p, float conf)
        : image(img), prompts(p), confidence_threshold(conf) {}
};

#endif // SAM3TYPE_HPP__