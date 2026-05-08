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
struct Sam3PromptUnit
{
    std::string text;
    std::vector<BoxPrompt> boxes;
    Sam3PromptUnit() = default;
    Sam3PromptUnit(const std::string &t, const std::vector<BoxPrompt> &b = {})
        : text(t), boxes(b) {}
};

// 统一输入结构体
struct Sam3Input
{
    float confidence_threshold;
    cv::Mat image;                       // 必须: 输入图像
    bool merge_results = false;                // 如果在预先检测的情况下，是否将原始图的识别结果和裁剪后的图片识别结果进行合并
    std::vector<std::string> pre_detect_labels; // 可选: 预检测得到的标签列表（仅文本提示）
    std::vector<Sam3PromptUnit> prompts; // 必须: 该图对应的所有提示词列表

    // --- ominicrop 配置参数 ---
    int pre_crop_max_size = 640;         // crop 最大尺寸
    int pre_crop_padding = 20;           // 边缘 padding
    float pre_crop_w_diou = 30.0f;       // 距离惩罚权重
    float pre_crop_w_expansion = 5.0f;   // 扩展惩罚权重
    float pre_crop_count_penalty = 120.0f; // 裁剪数量惩罚
    float pre_crop_nms_threshold = 0.2f;   // 重叠 NMS 阈值
    bool pre_crop_enable_ar_fix = true;    // 是否启用长宽比修正
    float pre_crop_target_ar = 1.0f;       // 目标长宽比

    Sam3Input() = default;
    Sam3Input(const cv::Mat &img)
        : image(img) {}
    Sam3Input(const cv::Mat &img, const std::vector<Sam3PromptUnit> &p, float conf)
        : image(img), prompts(p), confidence_threshold(conf) {}
};

#endif // SAM3TYPE_HPP__