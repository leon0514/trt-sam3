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
    
    // 构造便于使用
    Sam3PromptUnit() = default;
    Sam3PromptUnit(const std::string& t, const std::vector<BoxPrompt>& b = {}) 
        : text(t), boxes(b) {}
};

// 统一输入结构体
struct Sam3Input
{
    cv::Mat image;                          // 必须: 输入图像
    std::vector<Sam3PromptUnit> prompts;    // 必须: 该图对应的所有提示词列表

    // 构造函数
    Sam3Input() = default;
    // 兼容旧接口的构造函数: 单图单Prompt
    Sam3Input(const cv::Mat &img, const std::string &txt = "", const std::vector<BoxPrompt> &boxes = {})
        : image(img) {
        prompts.emplace_back(txt, boxes);
    }
    // 新接口构造函数: 单图多Prompt
    Sam3Input(const cv::Mat &img, const std::vector<Sam3PromptUnit>& p)
        : image(img), prompts(p) {}
};

#endif // SAM3TYPE_HPP__