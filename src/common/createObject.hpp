#pragma once
#ifndef CREATEOBJECT_HPP
#define CREATEOBJECT_HPP
#include <string>
#include <vector>

namespace cv
{
    class Mat;
}

namespace object
{
    struct DetectionBox;
    struct PosePoint;
    struct Pose;
    struct Obb;
    struct Segmentation;
}

// 工厂方法，创建不同模型返回的DetectionBox对象
namespace object
{
    DetectionBox createBox(float left, float top, float right, float bottom, float score, int class_id, const std::string &class_name);
    DetectionBox createPositionBox(float left, float top, float right, float bottom, float score, int class_id, const std::string &class_name);
    DetectionBox createTrackBox(float left, float top, float right, float bottom, float score, int track_id, const std::string &class_name);
    DetectionBox createTrackBox(float left, float top, float right, float bottom, float score, int track_id, const std::string &class_name, const object::Pose &pose);
    DetectionBox createTrackBox(float left, float top, float right, float bottom, float score, int track_id, const std::string &class_name, const object::Obb &obb);
    DetectionBox createTrackBox(float left, float top, float right, float bottom, float score, int track_id, const std::string &class_name, const object::Segmentation &seg);
    DetectionBox createObbBox(float cx, float cy, float w, float h, float angle, float score, int class_id, const std::string &class_name);
    DetectionBox createPoseBox(float left, float top, float right, float bottom, const std::vector<PosePoint> &pose_points, float score, int class_id, const std::string &class_name);
    DetectionBox createSegmentationBox(float left, float top, float right, float bottom, const cv::Mat &mask, float score, int class_id, const std::string &class_name);
    DetectionBox createDepthProBox(const cv::Mat &depth, float fog_data);
    DetectionBox createDepthAnythingBox(const cv::Mat &depth);
}

#endif