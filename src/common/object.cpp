#include "common/object.hpp"
#include <iostream>
#include <string>
#include <vector>

namespace object
{
    std::string ObjectTypeToString(ObjectType type) {
        switch (type) {
            case ObjectType::DETECTION:      return "DETECTION";
            case ObjectType::POSE:           return "POSE";
            case ObjectType::OBB:            return "OBB";
            case ObjectType::SEGMENTATION:   return "SEGMENTATION";
            case ObjectType::TRACK:          return "TRACK";
            case ObjectType::DEPTH_ANYTHING: return "DEPTH_ANYTHING";
            case ObjectType::DEPTH_PRO:      return "DEPTH_PRO";
            case ObjectType::POSITION:       return "POSITION";
            case ObjectType::UNKNOW:
            default:                         return "UNKNOW";
        }
    }

    // Box 的构造函数实现
    Box::Box(float l, float t, float r, float b)
        : left(l), top(t), right(r), bottom(b) {}

    // Box 的输出流操作符重载，用于打印
    std::ostream &operator<<(std::ostream &os, const Box &box) {
        os << "{ \"left\": " << box.left
           << ", \"top\": " << box.top
           << ", \"right\": " << box.right
           << ", \"bottom\": " << box.bottom
           << " }";
        return os;
    }

    // PosePoint 的构造函数实现
    PosePoint::PosePoint(float x, float y, float vis) : x(x), y(y), vis(vis) {}

    // PosePoint 的输出流操作符重载
    std::ostream &operator<<(std::ostream &os, const PosePoint &point) {
        os << "{ \"x\": " << point.x
           << ", \"y\": " << point.y
           << ", \"vis\": " << point.vis
           << " }";
        return os;
    }

    PosePoint& PosePoint::operator=(const PosePoint& other)
    {
        if (this == &other) {
            return *this;
        }
        this->x   = other.x;
        this->y   = other.y;
        this->vis = other.vis;

        // 3. Return a reference to the current object to allow for chained assignments (e.g., a = b = c).
        return *this;
    }
    
    // Pose 的输出流操作符重载
    std::ostream &operator<<(std::ostream &os, const Pose &pose) {
        os << "[";
        for (size_t i = 0; i < pose.points.size(); ++i) {
            os << pose.points[i];
            if (i < pose.points.size() - 1) os << ", ";
        }
        os << "]";
        return os;
    }

    Pose& Pose::operator=(const Pose& other)
    {
        if (this == &other) {
            return *this;
        }
        this->points = other.points;
        return *this;
    }

    // Obb 的构造函数实现
    Obb::Obb(float cx, float cy, float w, float h, float angle)
        : cx(cx), cy(cy), w(w), h(h), angle(angle) {}

    Obb& Obb::operator=(const Obb& other)
    {
        if (this == &other) {
            return *this;
        }
        this->cx    = other.cx;
        this->cy    = other.cy;
        this->w     = other.w;
        this->h     = other.h;
        this->angle = other.angle;

        // 3. Return a reference to the current object to allow for chained assignments (e.g., a = b = c).
        return *this;
    }

    // Obb 的输出流操作符重载
    std::ostream &operator<<(std::ostream &os, const Obb &obb) {
        os << "{ \"cx\": " << obb.cx
           << ", \"cy\": " << obb.cy
           << ", \"w\": " << obb.w
           << ", \"h\": " << obb.h
           << ", \"angle\": " << obb.angle
           << " }";
        return os;
    }


    SegmentMap::SegmentMap(int width, int height)
    {
        this->width  = width;
        this->height = height;
        checkRuntime(cudaMallocHost(&this->data, width * height));
    }

    SegmentMap::~SegmentMap()
    {
        if (this->data)
        {
            checkRuntime(cudaFreeHost(this->data));
            this->data = nullptr;
        }
        this->width  = 0;
        this->height = 0;
    }

    SegmentMap &SegmentMap::operator=(SegmentMap &&other) noexcept
    {
        // Prevent self-assignment (though unlikely with &&)
        if (this != &other)
        {
            // Free existing resource first
            if (this->data)
            {
                checkRuntime(cudaFreeHost(this->data));
            }

            // Transfer ownership from 'other'
            width  = std::exchange(other.width, 0);
            height = std::exchange(other.height, 0);
            data   = std::exchange(other.data, nullptr);
        }
        return *this;
    }

    // Track 的输出流操作符重载
    std::ostream &operator<<(std::ostream &os, const Track &track) {
        os << "{ \"track_id\": " << track.track_id
           << ", \"trace\": [";
        for (size_t i = 0; i < track.track_trace.size(); ++i) {
            // std::get<index> 是访问 tuple 元素的正确方式
            os << "{ \"x\": " << std::get<0>(track.track_trace[i]) 
               << ", \"y\": " << std::get<1>(track.track_trace[i]) << " }";
            if (i < track.track_trace.size() - 1) os << ", ";
        }
        os << "] }";
        return os;
    }


    float Depth::point_depth(int x, int y) const {
        if (depth.empty() || y < 0 || y >= depth.rows || x < 0 || x >= depth.cols) {
            return 0.0f; // 边界检查，防止访问越界
        }
        // 假设深度图是 CV_32F (32位浮点型)
        return depth.at<float>(y, x);
    }

    float Depth::average_depth() const {
        if (depth.empty()) return 0.0f;
        // cv::mean 返回一个 Scalar，我们通常取第一个通道的均值
        return static_cast<float>(cv::mean(depth)[0]);
    }

    float Depth::min_depth() const {
        if (depth.empty()) return 0.0f;
        double min_val;
        cv::minMaxLoc(depth, &min_val, nullptr, nullptr, nullptr);
        return static_cast<float>(min_val);
    }

    float Depth::max_depth() const {
        if (depth.empty()) return 0.0f;
        double max_val;
        cv::minMaxLoc(depth, nullptr, &max_val, nullptr, nullptr);
        return static_cast<float>(max_val);
    }

    float Depth::area_average_depth(const cv::Mat &seg) const {
        if (depth.empty() || seg.empty()) return 0.0f;
        // 使用掩码计算特定区域的平均深度
        cv::Mat masked_depth;
        depth.copyTo(masked_depth, seg); // 只拷贝掩码为非零的区域
        
        float sum_depth = cv::sum(masked_depth)[0];
        int area = cv::countNonZero(seg);
        return area > 0 ? sum_depth / area : 0.0f;
    }

    float Depth::area_average_depth(const Box &box) const {
        if (depth.empty()) return 0.0f;
        // 定义感兴趣区域 (Region of Interest, ROI)
        cv::Rect roi(cv::Point(box.left, box.top), cv::Point(box.right, box.bottom));
        // 将 ROI 与图像边界取交集，确保 ROI 不会超出图像范围
        roi &= cv::Rect(0, 0, depth.cols, depth.rows);
        if (roi.area() == 0) return 0.0f;
        // 直接计算 ROI 区域的均值
        return static_cast<float>(cv::mean(depth(roi))[0]);
    }

    Segmentation& Segmentation::operator=(const Segmentation& other)
    {
        if (this == &other) {
            return *this;
        }
        this->mask = other.mask.clone();
        return *this;
    }

    void Segmentation::keep_largest_part()
    {
        if (mask.empty()) return;

        // 查找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (contours.empty()) return;

        // 找到最大轮廓
        auto max_contour = std::max_element(contours.begin(), contours.end(),
                                             [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b) {
                                                 return cv::contourArea(a) < cv::contourArea(b);
                                             });

        // 创建一个新的掩码，只保留最大轮廓
        cv::Mat new_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
        cv::drawContours(new_mask, std::vector<std::vector<cv::Point>>{*max_contour}, -1, cv::Scalar(255), cv::FILLED);
        mask = new_mask;
    }

    Segmentation Segmentation::align_to_left_top(int left, int top, int width, int height) const
    {
        object::Segmentation aligned_seg;
        // 原本的mask是相对于left,top的
        // 现在我们需要创建一个新的mask，大小为width x height，并将原mask放置在新的mask中
        cv::Mat aligned_mask = cv::Mat::zeros(height, width, mask.type());
        if (mask.empty()) return aligned_seg;
        // 计算放置位置
        int x_offset = std::max(0, left);
        int y_offset = std::max(0, top);
        // 计算原mask在新mask中的有效区域
        int copy_width = std::min(mask.cols, width - x_offset);
        int copy_height = std::min(mask.rows, height - y_offset);
        if (copy_width > 0 && copy_height > 0) {
            cv::Rect src_roi(0, 0, copy_width, copy_height);
            cv::Rect dst_roi(x_offset, y_offset, copy_width, copy_height);
            mask(src_roi).copyTo(aligned_mask(dst_roi));
        }
        aligned_seg.mask = aligned_mask;
        return aligned_seg;
    }

    //================================================================================
    // 已修改: DetectionBox 的输出流操作符重载
    // 这是最重要的修改。它不再使用僵化的 switch 结构，
    // 而是先打印所有必需字段，然后逐一检查每个 optional 成员是否存在值。
    // 这种方式更灵活，能准确地反映出任意组合的数据。
    //================================================================================
    std::ostream &operator<<(std::ostream &os, const DetectionBox &box) {
        os << "{";
        
        // --- 打印核心/必需字段 ---
        os << " \"type\": \"" << ObjectTypeToString(box.type) << "\""
           << ", \"class_id\": " << box.class_id
           << ", \"class_name\": \"" << box.class_name << "\""
           << ", \"score\": " << box.score
           << ", \"box\": " << box.box;

        // --- 逐一检查并打印可选字段 ---
        if (box.pose.has_value()) {
            os << ", \"pose\": " << box.pose.value();
        }
        if (box.obb.has_value()) {
            os << ", \"obb\": " << box.obb.value();
        }
        if (box.track.has_value()) {
            os << ", \"track\": " << box.track.value();
        }
        if (box.segmentation.has_value()) {
            // 直接打印整个掩码矩阵不现实，所以我们只打印它的尺寸信息
            const auto& mask = box.segmentation.value().mask;
            os << ", \"segmentation\": { \"width\": " << mask.cols 
               << ", \"height\": " << mask.rows << " }";
        }
        if (box.depth.has_value()) {
            // 同理，只打印深度图的尺寸信息
            const auto& depth = box.depth.value().depth;
            os << ", \"depth\": { \"width\": " << depth.cols
               << ", \"height\": " << depth.rows << " }";
        }

        os << " }";
        return os;
    }

    cv::Mat segmentMapToMat(const std::shared_ptr<SegmentMap>& map)
    {
        // 检查输入数据是否有效
        if (map->data == nullptr || map->width <= 0 || map->height <= 0) 
        {
            // 如果SegmentMap无效，返回一个空的cv::Mat
            return cv::Mat();
        }

        // 使用SegmentMap的数据指针、高度和宽度创建一个cv::Mat对象头。
        // CV_8UC1 表示这是一个8位无符号单通道图像，这通常是分割掩码的格式。
        // cv::Mat的这个构造函数不会复制数据，而是直接使用提供的指针。
        return cv::Mat(map->height, map->width, CV_8UC1, map->data);
    }

} // namespace object