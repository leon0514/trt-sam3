#pragma once
#ifndef OSD_HPP
#define OSD_HPP

// 引入项目所需的自定义头文件
#include "common/object.hpp" // 假设此文件定义了 object::DetectionBox, object::DetectionBoxArray 等结构
#include "osd/position.hpp"  // 假设此文件定义了 PositionManager 类

// 引入标准库和OpenCV库
#include <memory>
#include <vector>
#include <tuple>
#include <string>
#include <unordered_map>
#include <opencv2/opencv.hpp>

/**
 * @brief 在图像上绘制基础的目标检测框（实线矩形）和信息标签。
 * @param img 要在其上进行绘制的图像 (cv::Mat)，会直接被修改。
 * @param box 单个检测结果对象。
 * @param pm PositionManager 智能指针，用于自动选择标签的最佳显示位置。
 * @param color 要用于绘制框和文本的颜色。
 * @param font_size 文本的绝对字体大小（以像素为单位）。
 */
void drawBaseInfo(cv::Mat &img, const object::DetectionBox &box, std::shared_ptr<PositionManager<float>> pm, const cv::Scalar &color, int font_size);

/**
 * @brief 在图像上绘制特殊的位置检测框（虚线矩形）和类别标签。
 * @param img 要在其上进行绘制的图像 (cv::Mat)。
 * @param box 单个检测结果对象，其类型应为 object::ObjectType::POSITION。
 * @param pm PositionManager 智能指针，用于自动选择标签的最佳显示位置。
 * @param color 要用于绘制框和文本的颜色。
 * @param font_size 文本的绝对字体大小（以像素为单位）。
 */
void drawPositionRectBox(cv::Mat &img, const object::DetectionBox &box, std::shared_ptr<PositionManager<float>> pm, const cv::Scalar &color, int font_size);

/**
 * @brief 在图像上绘制姿态估计的关键点和骨骼连接线。
 * @param img 要在其上进行绘制的图像 (cv::Mat)。
 * @param box 包含姿态关键点信息的单个检测结果对象。
 * @param thickness 线条的粗细（整数像素值）。
 */
void drawPoseSkeleton(cv::Mat &img, const object::DetectionBox &box, int thickness);

/**
 * @brief 在图像上绘制旋转目标框（Oriented Bounding Box）。
 * @param img 要在其上进行绘制的图像 (cv::Mat)。
 * @param box 包含旋转框信息的单个检测结果对象。
 * @param thickness 线条的粗细（整数像素值）。
 */
void drawObbBox(cv::Mat &img, const object::DetectionBox &box, int thickness);

/**
 * @brief 将目标的分割掩码以半透明颜色叠加到图像上。
 * @param img 要在其上进行绘制的图像 (cv::Mat)。
 * @param box 包含分割掩码的单个检测结果对象。
 */
void drawSegmentationMask(cv::Mat &img, const object::DetectionBox &box);

/**
 * @brief 在图像上绘制目标的跟踪轨迹和跟踪ID。
 * @param img 要在其上进行绘制的图像 (cv::Mat)。
 * @param box 包含跟踪信息的单个检测结果对象。
 * @param font_size 文本的绝对字体大小（以像素为单位），用于绘制ID。
 */
void drawTrackTrace(cv::Mat &img, const object::DetectionBox &box, int font_size);

/**
 * @brief 在图像上绘制目标的历史姿态轨迹。
 * @param img 要在其上进行绘制的图像 (cv::Mat)。
 * @param box 包含历史姿态信息的单个检测结果对象。
 * @param thickness 线条的粗细（整数像素值）。
 */
void drawTrackHistoryPose(cv::Mat &img, const object::DetectionBox &box, int thickness);

/**
 * @brief 将深度图以伪彩色的形式可视化并叠加到图像上。
 * @param img 要在其上进行绘制的图像 (cv::Mat)。
 * @param box 包含深度图的单个检测结果对象。
 */
void drawDepth(cv::Mat &img, const object::DetectionBox &box);

/**
 * @brief 在图像上绘制一个由点集定义的多边形。
 * @param img 要在其上进行绘制的图像 (cv::Mat)。
 * @param points 构成多边形的顶点坐标集合。
 * @param color 多边形线条的颜色。
 * @param thickness 线条的粗细。
 */
void drawPolygon(cv::Mat &img, const std::vector<std::tuple<float, float>> &points, const cv::Scalar &color, int thickness);

/**
 * @brief 【主函数】在图像上绘制所有检测结果。
 * @param img 要在其上进行绘制的图像 (cv::Mat)。
 * @param boxes 包含所有检测结果的数组。
 * @param osd_rect 是否绘制矩形框，默认为 true。
 * @param font_scale_ratio 字体大小与图像短边长度的比例，用于动态计算绝对字体大小。
 */
void osd(cv::Mat &img, const object::DetectionBoxArray &boxes, bool osd_rect = true, double font_scale_ratio = 0.04);

/**
 * @brief 在图像上绘制多个带标签的多边形区域。
 * @param img 要在其上进行绘制的图像 (cv::Mat)。
 * @param points 从标签(string)到顶点集合(vector)的映射。
 * @param color 绘制的颜色。
 * @param font_scale_ratio 字体大小的比例，用于动态计算。
 */
void osd(cv::Mat &img, const std::unordered_map<std::string, std::vector<std::tuple<float, float>>> &points, const cv::Scalar &color = cv::Scalar(0, 255, 0), double font_scale_ratio = 0.04);

/**
 * @brief 在图像上绘制单个带标签的多边形区域。
 * @param img 要在其上进行绘制的图像 (cv::Mat)。
 * @param fence_name 多边形的标签名称。
 * @param points 多边形的顶点坐标集合。
 * @param color 绘制的颜色。
 * @param font_scale_ratio 字体大小的比例，用于动态计算。
 */
void osd(cv::Mat &img, const std::string &fence_name, const std::vector<std::tuple<float, float>> &points, const cv::Scalar &color = cv::Scalar(0, 255, 0), double font_scale_ratio = 0.04);

/**
 * @brief 在图像的指定位置绘制一行文本。
 * @param img 要在其上进行绘制的图像 (cv::Mat)。
 * @param position 文本开始绘制的左下角坐标 (x, y)。
 * @param text 要绘制的字符串。
 * @param color 文本颜色。
 * @param font_size 绝对字体大小（以像素为单位），可以直接控制。
 */
void osd(cv::Mat &img, const std::tuple<float, float> &position, const std::string &text, const cv::Scalar &color = cv::Scalar(0, 255, 0), int font_size = 40);

#endif // OSD_HPP