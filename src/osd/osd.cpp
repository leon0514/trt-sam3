#include "osd.hpp"
#include "osd/cvx_text.hpp" // 请确保此头文件的路径相对于您的项目配置是正确的
#include <vector>
#include <tuple>
#include <functional>
#include <limits>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>

// 使用 constexpr 替代 #define，提供类型安全和作用域
constexpr int COCO_NUM_KEYPOINTS = 17;
constexpr int HAND_NUM_KEYPOINTS = 21;

// 匿名命名空间，用于存放仅在此文件内使用的辅助函数和常量
namespace
{
    // COCO 姿态关键点连接关系
    const std::vector<std::pair<int, int>> coco_pairs =
        {
            {0, 1}, {0, 2}, {1, 3}, {2, 4}, {0, 5}, {0, 6}, {5, 6}, {5, 11}, {6, 12}, {11, 12}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {11, 13}, {13, 15}, {12, 14}, {14, 16}};

    // 手部关键点连接关系
    const std::vector<std::pair<int, int>> hand_pairs =
        {
            {0, 1}, {0, 5}, {0, 9}, {0, 13}, {0, 17}, {5, 9}, {9, 13}, {13, 17}, {1, 2}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {7, 8}, {9, 10}, {10, 11}, {11, 12}, {13, 14}, {14, 15}, {15, 16}, {17, 18}, {18, 19}, {19, 20}};

    // --- 颜色生成辅助函数 ---
    std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v)
    {
        const int h_i = static_cast<int>(h * 6);
        const float f = h * 6 - h_i;
        const float p = v * (1 - s);
        const float q = v * (1 - f * s);
        const float t = v * (1 - (1 - f) * s);
        float r, g, b;
        switch (h_i)
        {
        case 0:
            r = v, g = t, b = p;
            break;
        case 1:
            r = q, g = v, b = p;
            break;
        case 2:
            r = p, g = v, b = t;
            break;
        case 3:
            r = p, g = q, b = v;
            break;
        case 4:
            r = t, g = p, b = v;
            break;
        case 5:
            r = v, g = p, b = q;
            break;
        default:
            r = 1, g = 1, b = 1;
            break;
        }
        return {static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255)};
    }

    std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id)
    {
        float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
        float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f * 0.5f + 0.5f;
        return hsv2bgr(h_plane, s_plane, 1.0f);
    }

    std::tuple<uint8_t, uint8_t, uint8_t> random_color(const std::string &label)
    {
        std::hash<std::string> hasher;
        return random_color(static_cast<int>(hasher(label) & 0x7FFFFFFF));
    }

    // --- 绘制辅助函数 ---
    void overlay_mask(cv::Mat &image, const cv::Mat &mask, const object::Box &box, const cv::Scalar &color, double alpha)
    {
        if (image.empty() || mask.empty() || image.type() != CV_8UC3 || mask.type() != CV_8UC1)
            return;
        alpha = std::max(0.0, std::min(1.0, alpha));
        // cv::Rect roi(cv::Point(box.left, box.top), cv::Point(box.right, box.bottom));
        // roi &= cv::Rect(0, 0, image.cols, image.rows);
        // if (roi.area() <= 0)
        //     return;

        // cv::Mat image_roi = image(roi);

        cv::Mat resized_mask;
        cv::resize(mask, resized_mask, image.size());
        cv::Mat color_patch(image.size(), image.type(), color);
        cv::Mat weighted_color;
        cv::addWeighted(color_patch, alpha, image, 1.0 - alpha, 0.0, weighted_color);
        weighted_color.copyTo(image, resized_mask);
    }

    void drawDashedRectangle(cv::Mat &img, cv::Rect rect, const cv::Scalar &color, int thickness, int dash_size, int gap_size)
    {
        auto draw_dashed_line = [&](cv::Point start, cv::Point end)
        {
            cv::LineIterator it(img, start, end, 8);
            for (int i = 0; i < it.count; i++, ++it)
            {
                if (i % (dash_size + gap_size) < dash_size)
                {
                    cv::circle(img, it.pos(), 0, color, thickness);
                }
            }
        };
        draw_dashed_line(rect.tl(), {rect.br().x, rect.tl().y});
        draw_dashed_line({rect.br().x, rect.tl().y}, rect.br());
        draw_dashed_line(rect.br(), {rect.tl().x, rect.br().y});
        draw_dashed_line({rect.tl().x, rect.br().y}, rect.tl());
    }

    std::tuple<float, float> calculatePolygonCentroid(const std::vector<std::tuple<float, float>> &pts)
    {
        if (pts.size() < 3)
        {
            float sum_x = 0.0f, sum_y = 0.0f;
            if (!pts.empty())
            {
                for (const auto &p : pts)
                {
                    sum_x += std::get<0>(p);
                    sum_y += std::get<1>(p);
                }
                return {sum_x / pts.size(), sum_y / pts.size()};
            }
            return {0.0f, 0.0f};
        }
        float signedArea = 0.0f, cx = 0.0f, cy = 0.0f;
        for (size_t i = 0; i < pts.size(); ++i)
        {
            float xi = std::get<0>(pts[i]), yi = std::get<1>(pts[i]);
            float xj = std::get<0>(pts[(i + 1) % pts.size()]), yj = std::get<1>(pts[(i + 1) % pts.size()]);
            float cross_product = (xi * yj - xj * yi);
            signedArea += cross_product;
            cx += (xi + xj) * cross_product;
            cy += (yi + yj) * cross_product;
        }
        signedArea *= 0.5f;
        if (std::abs(signedArea) < 1e-6)
        {
            return calculatePolygonCentroid({});
        }
        float sixA = 6.0f * signedArea;
        return {cx / sixA, cy / sixA};
    }

    static const char *font_path = "font/SIMKAI.TTF"; // !重要: 确保这个字体路径正确
    static CvxText text_renderer(font_path);
}

//================================================================================
// 各种类型检测结果的独立绘制函数 (已修改为使用 CvxText)
//================================================================================

void drawBaseInfo(cv::Mat &img, const object::DetectionBox &box, std::shared_ptr<PositionManager<float>> pm, const cv::Scalar &color, int font_size)
{
    int thickness = std::max(1, font_size / 15);
    cv::Rect rect(cv::Point(box.box.left, box.box.top), cv::Point(box.box.right, box.box.bottom));
    cv::rectangle(img, rect, color, thickness);

    std::ostringstream oss;
    oss << box.class_name << ":" << std::fixed << std::setprecision(3) << box.score;
    std::string text = oss.str();

    int x, y;
    std::tie(x, y) = pm->selectOptimalPosition({box.box.left, box.box.top, box.box.right, box.box.bottom}, img.cols, img.rows, text);

    text_renderer.putText(img, text, cv::Point(x, y), color, font_size);
}

void drawPositionRectBox(cv::Mat &img, const object::DetectionBox &box, std::shared_ptr<PositionManager<float>> pm, const cv::Scalar &color, int font_size)
{
    int thickness = std::max(1, font_size / 15);
    cv::Rect rect(cv::Point(box.box.left, box.box.top), cv::Point(box.box.right, box.box.bottom));
    drawDashedRectangle(img, rect, color, thickness, 15, 10);

    std::string text = box.class_name;
    int x, y;
    std::tie(x, y) = pm->selectOptimalPosition({box.box.left, box.box.top, box.box.right, box.box.bottom}, img.cols, img.rows, text);

    text_renderer.putText(img, text, cv::Point(x, y), color, font_size);
}

void drawTrackHistoryPose(cv::Mat &img, const object::DetectionBox &box, int thickness)
{
    if (!box.track || !box.track->history_pose.has_value())
        return;
    for (const auto &pose : box.track->history_pose.value())
    {
        const auto &points = pose.points;
        const auto *pairs = (points.size() == COCO_NUM_KEYPOINTS) ? &coco_pairs : (points.size() == HAND_NUM_KEYPOINTS) ? &hand_pairs
                                                                                                                        : nullptr;
        if (!pairs)
            continue;
        for (const auto &pair : *pairs)
        {
            const auto &p1 = points[pair.first];
            const auto &p2 = points[pair.second];
            if (p1.vis > 0 && p2.vis > 0)
            {
                auto line_color_tuple = random_color(static_cast<int>(&pair - &(*pairs)[0] + 100));
                cv::line(img, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), {std::get<0>(line_color_tuple), std::get<1>(line_color_tuple), std::get<2>(line_color_tuple)}, thickness, cv::LINE_AA);
            }
        }
        for (size_t i = 0; i < points.size(); ++i)
        {
            if (points[i].vis > 0)
            {
                auto pt_color_tuple = random_color(static_cast<int>(i));
                cv::circle(img, cv::Point(points[i].x, points[i].y), 3, {std::get<0>(pt_color_tuple), std::get<1>(pt_color_tuple), std::get<2>(pt_color_tuple)}, -1, cv::LINE_AA);
            }
        }
    }
}

void drawPoseSkeleton(cv::Mat &img, const object::DetectionBox &box, int thickness)
{
    if (box.type == object::ObjectType::TRACK)
        return;
    if (!box.pose)
        return;
    const auto &points = box.pose->points;
    const auto *pairs = (points.size() == COCO_NUM_KEYPOINTS) ? &coco_pairs : (points.size() == HAND_NUM_KEYPOINTS) ? &hand_pairs
                                                                                                                    : nullptr;
    if (!pairs)
        return;
    for (size_t i = 0; i < pairs->size(); ++i)
    {
        const auto &pair = (*pairs)[i];
        const auto &p1 = points[pair.first];
        const auto &p2 = points[pair.second];
        if (p1.vis > 0 && p2.vis > 0)
        {
            auto line_color_tuple = random_color(static_cast<int>(i + 100));
            cv::line(img, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), {std::get<0>(line_color_tuple), std::get<1>(line_color_tuple), std::get<2>(line_color_tuple)}, thickness, cv::LINE_AA);
        }
    }
    for (size_t i = 0; i < points.size(); ++i)
    {
        if (points[i].vis > 0)
        {
            auto pt_color_tuple = random_color(static_cast<int>(i));
            cv::circle(img, cv::Point(points[i].x, points[i].y), 3, {std::get<0>(pt_color_tuple), std::get<1>(pt_color_tuple), std::get<2>(pt_color_tuple)}, -1, cv::LINE_AA);
        }
    }
}

void drawObbBox(cv::Mat &img, const object::DetectionBox &box, int thickness)
{
    if (box.type == object::ObjectType::TRACK)
        return;
    if (!box.obb)
        return;
    auto color_tuple = random_color(box.class_name);
    cv::Scalar color(std::get<0>(color_tuple), std::get<1>(color_tuple), std::get<2>(color_tuple));
    cv::RotatedRect rRect(cv::Point2f(box.obb->cx, box.obb->cy), cv::Size2f(box.obb->w, box.obb->h), box.obb->angle);
    cv::Point2f vertices[4];
    rRect.points(vertices);
    for (int i = 0; i < 4; i++)
    {
        cv::line(img, vertices[i], vertices[(i + 1) % 4], color, thickness);
    }
}

void drawSegmentationMask(cv::Mat &img, const object::DetectionBox &box)
{
    if (box.type == object::ObjectType::TRACK)
        return;
    if (!box.segmentation || box.segmentation->mask.empty())
        return;
    auto color_tuple = random_color(box.class_name);
    cv::Scalar color(std::get<0>(color_tuple), std::get<1>(color_tuple), std::get<2>(color_tuple));
    overlay_mask(img, box.segmentation->mask, box.box, color, 0.5);
}

void drawTrackTrace(cv::Mat &img, const object::DetectionBox &box, int font_size)
{
    if (!box.track)
        return;
    auto color_tuple = random_color(box.track->track_id);
    cv::Scalar color(std::get<0>(color_tuple), std::get<1>(color_tuple), std::get<2>(color_tuple));

    int thickness = std::max(1, font_size / 15);
    const auto &trace = box.track->track_trace;

    for (size_t i = 1; i < trace.size(); ++i)
    {
        cv::Point p1(std::get<0>(trace[i - 1]), std::get<1>(trace[i - 1]));
        cv::Point p2(std::get<0>(trace[i]), std::get<1>(trace[i]));
        cv::line(img, p1, p2, color, thickness);
    }
    for (const auto &point_tuple : trace)
    {
        cv::circle(img, cv::Point(std::get<0>(point_tuple), std::get<1>(point_tuple)), 2, color, -1);
    }

    int id_font_size = static_cast<int>(font_size * 0.8);
    std::string text = "ID:" + std::to_string(box.track->track_id);
    text_renderer.putText(img, text, cv::Point(box.box.center_x() + 10, box.box.center_y()), color, id_font_size);
}

void drawDepth(cv::Mat &img, const object::DetectionBox &box)
{
    if (!box.depth || box.depth->depth.empty())
        return;
    cv::Mat depth_normalized, color_depth;
    cv::normalize(box.depth->depth, depth_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(depth_normalized, color_depth, cv::COLORMAP_JET);
    cv::addWeighted(img, 0.5, color_depth, 0.5, 0.0, img);
}

void drawPolygon(cv::Mat &img, const std::vector<std::tuple<float, float>> &points, const cv::Scalar &color, int thickness)
{
    if (points.size() < 2)
        return;
    std::vector<cv::Point> polygon;
    polygon.reserve(points.size());
    for (const auto &point : points)
    {
        polygon.emplace_back(std::get<0>(point), std::get<1>(point));
    }
    cv::polylines(img, polygon, true, color, thickness);
}

//================================================================================
// 主 OSD 函数 (已重构)
//================================================================================

void osd(cv::Mat &img, const object::DetectionBoxArray &boxes, bool osd_rect, double font_scale_ratio)
{
    int height = img.rows, width = img.cols;
    int final_font_size = std::max(20, static_cast<int>(std::min(width, height) * font_scale_ratio));

    auto get_text_size_for_pm = [final_font_size](const std::string &text) -> std::tuple<int, int, int>
    {
        int w, h, baseline;
        text_renderer.getTextSize(text, final_font_size, &w, &h, &baseline);
        return {w, h, baseline};
    };
    auto pm = std::make_shared<PositionManager<float>>(get_text_size_for_pm);

    for (const auto &box : boxes)
    {
        if (box.type == object::ObjectType::DEPTH_PRO || box.type == object::ObjectType::DEPTH_ANYTHING)
        {
            drawDepth(img, box);
            continue;
        }

        auto color_tuple = random_color(box.class_name);
        cv::Scalar color(std::get<0>(color_tuple), std::get<1>(color_tuple), std::get<2>(color_tuple));
        int final_thickness = std::max(1, final_font_size / 7);

        if (osd_rect)
        {
            if (box.type == object::ObjectType::POSITION)
            {
                drawPositionRectBox(img, box, pm, color, final_font_size);
            }
            else
            {
                drawBaseInfo(img, box, pm, color, final_font_size);
            }
        }

        drawSegmentationMask(img, box);
        drawPoseSkeleton(img, box, final_thickness);
        drawTrackHistoryPose(img, box, final_thickness);
        drawTrackTrace(img, box, final_font_size);
        drawObbBox(img, box, final_thickness);
    }
}

void osd(cv::Mat &img, const std::unordered_map<std::string, std::vector<std::tuple<float, float>>> &points, const cv::Scalar &color, double font_scale_ratio)
{
    int height = img.rows, width = img.cols;
    int final_font_size = std::max(20, static_cast<int>(std::min(width, height) * font_scale_ratio));
    int final_thickness = std::max(1, final_font_size / 7);

    for (const auto &[label, pts] : points)
    {
        if (pts.empty())
            continue;

        drawPolygon(img, pts, color, final_thickness);

        if (pts.size() >= 3)
        {
            auto center_tuple = calculatePolygonCentroid(pts);
            cv::Point center(std::get<0>(center_tuple), std::get<1>(center_tuple));
            text_renderer.putText(img, label, center, color, final_font_size);
        }
    }
}

void osd(cv::Mat &img, const std::string &fence_name, const std::vector<std::tuple<float, float>> &points, const cv::Scalar &color, double font_scale_ratio)
{
    int height = img.rows, width = img.cols;
    int final_font_size = std::max(20, static_cast<int>(std::min(width, height) * font_scale_ratio));
    int final_thickness = std::max(1, final_font_size / 7);

    if (points.empty())
        return;

    drawPolygon(img, points, color, final_thickness);

    if (points.size() >= 3)
    {
        auto center_tuple = calculatePolygonCentroid(points);
        cv::Point center(std::get<0>(center_tuple), std::get<1>(center_tuple));
        text_renderer.putText(img, fence_name, center, color, final_font_size);
    }
}

void osd(cv::Mat &img, const std::tuple<float, float> &position, const std::string &text, const cv::Scalar &color, int font_size)
{
    cv::Point text_position(std::get<0>(position), std::get<1>(position));
    text_renderer.putText(img, text, text_position, color, font_size);
}