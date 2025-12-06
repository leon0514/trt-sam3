#ifndef OBJECT_HPP
#define OBJECT_HPP

#include "common/check.hpp"
#include <string>
#include <vector>
#include <tuple>
#include <ostream>
#include <optional>
#include "opencv2/opencv.hpp"

namespace object
{
    enum class ObjectType
    {
        UNKNOW = -1,
        POSITION = 0,
        POSE = 1,
        OBB = 2,
        SEGMENTATION = 3,
        DEPTH_ANYTHING = 4,
        DEPTH_PRO = 5,
        TRACK = 6,
        DETECTION = 7,
    };

    struct Box
    {
        float left = 0.0f;
        float top = 0.0f;
        float right = 0.0f;
        float bottom = 0.0f;

        Box() = default;
        Box(float l, float t, float r, float b);

        float width() const noexcept { return right - left; }
        float height() const noexcept { return bottom - top; }
        float center_x() const noexcept { return (left + right) / 2; }
        float center_y() const noexcept { return (top + bottom) / 2; }
        float area() const noexcept { return width() * height(); }

        friend std::ostream &operator<<(std::ostream &os, const Box &box);
    };

    struct PosePoint
    {
        float x = 0.0f;
        float y = 0.0f;
        float vis = 0.0f;

        PosePoint() = default;
        PosePoint(float x, float y, float vis);
        PosePoint& operator=(const PosePoint& other);
        friend std::ostream &operator<<(std::ostream &os, const PosePoint &point);
    };

    struct Pose
    {
        std::vector<PosePoint> points;
        Pose& operator=(const Pose& other);
        friend std::ostream &operator<<(std::ostream &os, const Pose &pose);
    };

    struct Obb
    {
        float cx = 0.0f;
        float cy = 0.0f;
        float w = 0.0f;
        float h = 0.0f;
        float angle = 0.0f;

        Obb() = default;
        Obb(float cx, float cy, float w, float h, float angle);
        Obb& operator=(const Obb& other);

        float area() const { return w * h; }

        friend std::ostream &operator<<(std::ostream &os, const Obb &obb);
    };

    struct SegmentMap
    {
        int width = 0, height = 0;     // width % 8 == 0
        unsigned char *data = nullptr; // is width * height memory

        SegmentMap(int width, int height);
        
        virtual ~SegmentMap();

        // 1. Delete Copy Constructor
        SegmentMap(const SegmentMap &) = delete;

        // 2. Delete Copy Assignment Operator
        SegmentMap &operator=(const SegmentMap &) = delete;

        // 3. Move Constructor
        SegmentMap(SegmentMap &&other) noexcept
            : width(std::exchange(other.width, 0)),    // Transfer ownership and reset source
            height(std::exchange(other.height, 0)),  // Transfer ownership and reset source
            data(std::exchange(other.data, nullptr)) // Transfer ownership and reset source
        {
            // The moved-from object 'other' is now in a valid, empty state
        }

        // 4. Move Assignment Operator
        SegmentMap &operator=(SegmentMap &&other) noexcept;
    };

    struct Segmentation
    {
        cv::Mat mask;
        // 分割区域可能有多个部分，使用findContours函数将最大的部分保留为mask
        void keep_largest_part();
        Segmentation align_to_left_top(int left, int top, int width, int height) const;
        Segmentation& operator=(const Segmentation& other);

    };

    struct Depth
    {
        cv::Mat depth;
        float fog_data = 0.0f;

        // 成员函数声明
        float point_depth(int x, int y) const;
        float average_depth() const;
        float min_depth() const;
        float max_depth() const;
        float area_average_depth(const cv::Mat &seg) const;
        float area_average_depth(const Box &box) const;
    };

    struct Track
    {
        int track_id = -1;
        // 如果跟踪的是姿态的框，将其有的关键点也和追踪关联
        std::optional<std::vector<Pose>> history_pose;
        std::vector<std::tuple<float, float>> track_trace;
        friend std::ostream &operator<<(std::ostream &os, const Track &track);
    };

    struct DetectionBox
    {
        ObjectType type = ObjectType::UNKNOW;
        Box box;
        float score = 0.0f;
        int class_id = -1;
        std::string class_name;

        // --- 可选的数据 ---
        std::optional<Pose> pose;
        std::optional<Obb> obb;
        std::optional<Segmentation> segmentation;
        std::optional<Depth> depth;
        std::optional<Track> track;

        // 友元函数声明
        friend std::ostream &operator<<(std::ostream &os, const DetectionBox &box);
    };

    cv::Mat segmentMapToMat(const std::shared_ptr<SegmentMap>& map);

    using DetectionBoxArray = std::vector<DetectionBox>;

} // namespace object

#endif // OBJECT_HPP