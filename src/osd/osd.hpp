#pragma once
#ifndef OSD_HPP
#define OSD_HPP

#include "common/object.hpp"
#include <vector>
#include <tuple>
#include <string>
#include <unordered_map>
#include <opencv2/opencv.hpp>

// 基础绘制
void drawBaseInfoGeometry(cv::Mat &img, const object::DetectionBox &box, const cv::Scalar &color, int thickness);
void drawPositionRectGeometry(cv::Mat &img, const object::DetectionBox &box, const cv::Scalar &color, int thickness);

// 辅助绘制
void drawPoseSkeleton(cv::Mat &img, const object::DetectionBox &box, int thickness);
void drawObbBox(cv::Mat &img, const object::DetectionBox &box, int thickness);
void drawSegmentationMask(cv::Mat &img, const object::DetectionBox &box);
void drawTrackTrace(cv::Mat &img, const object::DetectionBox &box, int font_size);
void drawTrackHistoryPose(cv::Mat &img, const object::DetectionBox &box, int thickness);
void drawDepth(cv::Mat &img, const object::DetectionBox &box);
void drawPolygon(cv::Mat &img, const std::vector<std::tuple<float, float>> &points, const cv::Scalar &color, int thickness);

// 主 OSD 函数
void osd(cv::Mat &img, const object::DetectionBoxArray &boxes, bool osd_rect = true, double font_scale_ratio = 0.04);

// 多边形重载
void osd(cv::Mat &img, const std::unordered_map<std::string, std::vector<std::tuple<float, float>>> &points, const cv::Scalar &color = cv::Scalar(0, 255, 0), double font_scale_ratio = 0.04);
void osd(cv::Mat &img, const std::string &fence_name, const std::vector<std::tuple<float, float>> &points, const cv::Scalar &color = cv::Scalar(0, 255, 0), double font_scale_ratio = 0.04);
void osd(cv::Mat &img, const std::tuple<float, float> &position, const std::string &text, const cv::Scalar &color = cv::Scalar(0, 255, 0), int font_size = 40);

#endif