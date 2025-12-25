#include "osd.hpp"
#include "osd/cvx_text.hpp" 
#include "osd/labelLayoutSolver.hpp" 
#include <vector>
#include <tuple>
#include <functional>
#include <limits>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <unordered_map>

constexpr int COCO_NUM_KEYPOINTS = 17;
constexpr int HAND_NUM_KEYPOINTS = 21;

namespace {
    
    // COCO 关键点连接
    const std::vector<std::pair<int, int>> coco_pairs =
    {
        {0, 1}, {0, 2}, {1, 3}, {2, 4}, {0, 5}, {0, 6}, {5, 6}, {5, 11}, 
        {6, 12}, {11, 12}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {11, 13}, 
        {13, 15}, {12, 14}, {14, 16}
    };
    const std::vector<std::pair<int, int>> hand_pairs =
    {
        {0, 1}, {0, 5}, {0, 9}, {0, 13}, {0, 17}, {5, 9}, {9, 13}, {13, 17}, 
        {1, 2}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {7, 8}, {9, 10}, {10, 11}, 
        {11, 12}, {13, 14}, {14, 15}, {15, 16}, {17, 18}, {18, 19}, {19, 20}
    };

    // HSV 转 BGR
    std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v) {
        const int h_i = static_cast<int>(h * 6);
        const float f = h * 6 - h_i;
        const float p = v * (1 - s);
        const float q = v * (1 - f * s);
        const float t = v * (1 - (1 - f) * s);
        float r, g, b;
        switch (h_i) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
        default: r = 1; g = 1; b = 1; break;
        }
        return {static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255)};
    }

    // 随机颜色
    std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id) {
        float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
        float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f * 0.5f + 0.5f;
        return hsv2bgr(h_plane, s_plane, 1.0f);
    }

    std::tuple<uint8_t, uint8_t, uint8_t> random_color(const std::string &label) {
        std::hash<std::string> hasher;
        return random_color(static_cast<int>(hasher(label) & 0x7FFFFFFF));
    }

    // Mask 叠加
    void overlay_mask(cv::Mat &image, const cv::Mat &mask, const object::Box& box, const cv::Scalar &color, double alpha) {
        if (image.empty() || mask.empty()) return;
        cv::Rect roi(cv::Point(box.left, box.top), cv::Point(box.right, box.bottom));
        roi &= cv::Rect(0, 0, image.cols, image.rows);
        if (roi.area() <= 0) return;
        cv::Mat image_roi = image(roi);
        cv::Mat resized_mask;
        cv::resize(mask, resized_mask, roi.size());
        cv::Mat color_patch(roi.size(), image.type(), color);
        cv::Mat weighted_color;
        cv::addWeighted(color_patch, alpha, image_roi, 1.0 - alpha, 0.0, weighted_color);
        weighted_color.copyTo(image_roi, resized_mask);
    }

    // 多边形中心
    std::tuple<float, float> calculatePolygonCentroid(const std::vector<std::tuple<float, float>> &pts) {
        if (pts.empty()) return {0.0f, 0.0f};
        float sum_x = 0, sum_y = 0;
        for(auto& p : pts) { sum_x += std::get<0>(p); sum_y += std::get<1>(p); }
        return {sum_x / pts.size(), sum_y / pts.size()};
    }

    static const char *font_path = "font/SIMKAI.TTF"; 
    static CvxText text_renderer(font_path);
}

// 几何绘制函数
void drawBaseInfoGeometry(cv::Mat &img, const object::DetectionBox &box, const cv::Scalar &color, int thickness) {
    cv::Rect rect(cv::Point(box.box.left, box.box.top), cv::Point(box.box.right, box.box.bottom));
    cv::rectangle(img, rect, color, thickness);
}

void drawPositionRectGeometry(cv::Mat &img, const object::DetectionBox &box, const cv::Scalar &color, int thickness) {
    // 简单实现虚线框，或者直接实线
    cv::Rect rect(cv::Point(box.box.left, box.box.top), cv::Point(box.box.right, box.box.bottom));
    cv::rectangle(img, rect, color, thickness); 
}

void drawPoseSkeleton(cv::Mat &img, const object::DetectionBox &box, int thickness) {
    if (box.type == object::ObjectType::TRACK || !box.pose) return;
    const auto &points = box.pose->points;
    const auto *pairs = (points.size() == COCO_NUM_KEYPOINTS) ? &coco_pairs : (points.size() == HAND_NUM_KEYPOINTS) ? &hand_pairs : nullptr;
    if (!pairs) return;
    for (const auto &pair : *pairs) {
        const auto &p1 = points[pair.first];
        const auto &p2 = points[pair.second];
        if (p1.vis > 0 && p2.vis > 0) {
            auto c = random_color(static_cast<int>(&pair - &(*pairs)[0] + 100));
            cv::line(img, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), { (double)std::get<0>(c), (double)std::get<1>(c), (double)std::get<2>(c) }, thickness);
        }
    }
}

void drawObbBox(cv::Mat &img, const object::DetectionBox &box, int thickness) {
    if (box.type == object::ObjectType::TRACK || !box.obb) return;
    auto c = random_color(box.class_name);
    cv::RotatedRect rRect(cv::Point2f(box.obb->cx, box.obb->cy), cv::Size2f(box.obb->w, box.obb->h), box.obb->angle);
    cv::Point2f vertices[4];
    rRect.points(vertices);
    for (int i = 0; i < 4; i++) cv::line(img, vertices[i], vertices[(i + 1) % 4], { (double)std::get<0>(c), (double)std::get<1>(c), (double)std::get<2>(c) }, thickness);
}

void drawSegmentationMask(cv::Mat &img, const object::DetectionBox &box) {
     if (!box.segmentation || box.segmentation->mask.empty()) return;
     auto c = random_color(box.class_name);
     overlay_mask(img, box.segmentation->mask, box.box, { (double)std::get<0>(c), (double)std::get<1>(c), (double)std::get<2>(c) }, 0.5);
}

void drawTrackHistoryPose(cv::Mat &img, const object::DetectionBox &box, int thickness) { 
    // 实现略，根据具体需求
}

void drawDepth(cv::Mat &img, const object::DetectionBox &box) {
    if (!box.depth || box.depth->depth.empty()) return;
    cv::Mat norm, cmap;
    cv::normalize(box.depth->depth, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(norm, cmap, cv::COLORMAP_JET);
    cv::addWeighted(img, 0.5, cmap, 0.5, 0.0, img);
}

void drawTrackTrace(cv::Mat &img, const object::DetectionBox &box, int font_size) {
    if (!box.track) return;
    auto c = random_color(box.track->track_id);
    cv::Scalar color((double)std::get<0>(c), (double)std::get<1>(c), (double)std::get<2>(c));
    int th = std::max(1, font_size / 15);
    const auto &trace = box.track->track_trace;
    for (size_t i = 1; i < trace.size(); ++i) {
        cv::line(img, cv::Point(std::get<0>(trace[i-1]), std::get<1>(trace[i-1])), 
                 cv::Point(std::get<0>(trace[i]), std::get<1>(trace[i])), color, th);
    }
    std::string text = "ID:" + std::to_string(box.track->track_id);
    text_renderer.putText(img, text, cv::Point(box.box.center_x(), box.box.center_y()), color, font_size * 0.8);
}

void drawPolygon(cv::Mat &img, const std::vector<std::tuple<float, float>> &points, const cv::Scalar &color, int thickness) {
    if (points.size() < 2) return;
    std::vector<cv::Point> poly;
    for (const auto &p : points) poly.emplace_back(std::get<0>(p), std::get<1>(p));
    cv::polylines(img, poly, true, color, thickness);
}

int calculateDynamicFontSize(int img_w, int img_h, const object::Box& box, double ratio) {
    // 只根据图像分辨率计算字体大小，保证全图一致
    // ratio 默认为 0.04 (即图像短边的 4%)
    int target_size = std::max(12, static_cast<int>(std::min(img_w, img_h) * ratio));    
    return target_size;
}
// =================================================================
// 主 OSD 函数
// =================================================================

void osd(cv::Mat &img, const object::DetectionBoxArray &boxes, bool osd_rect, double font_scale_ratio) {
    int height = img.rows, width = img.cols;
    const int PAD_X = 2; // 与 LayoutSolver 保持一致
    const int PAD_Y = 2;

    // 1. 初始化布局求解器
    LabelLayoutSolver solver(width, height, [&](const std::string& txt, int fontSize) -> TextSize {
        int w, h, base;
        text_renderer.getTextSize(txt, fontSize, &w, &h, &base);
        return {w, h, base};
    });
    
    std::vector<cv::Scalar> label_colors;
    std::vector<std::string> label_texts; 
    
    // 2. 遍历并绘制几何
    for (const auto &box : boxes) {
        if (box.type == object::ObjectType::DEPTH_PRO || box.type == object::ObjectType::DEPTH_ANYTHING) {
            drawDepth(img, box);
            continue;
        }

        auto c = random_color(box.class_name);
        cv::Scalar color(std::get<0>(c), std::get<1>(c), std::get<2>(c));
        
        int base_font_size = calculateDynamicFontSize(width, height, box.box, font_scale_ratio);
        int thickness = std::max(1, base_font_size / 10);

        if (osd_rect) {
            if (box.type == object::ObjectType::POSITION) drawPositionRectGeometry(img, box, color, thickness);
            else drawBaseInfoGeometry(img, box, color, thickness);
        }

        drawSegmentationMask(img, box);
        drawPoseSkeleton(img, box, thickness);
        drawTrackHistoryPose(img, box, thickness);
        drawTrackTrace(img, box, base_font_size);
        drawObbBox(img, box, thickness);

        // 添加文字任务
        if (osd_rect && (box.box.bottom - box.box.top) > 5) {
            std::string text;
            if (box.type == object::ObjectType::POSITION) text = box.class_name;
            else {
                std::ostringstream oss;
                oss << box.class_name << " " << std::fixed << std::setprecision(2) << box.score;
                text = oss.str();
            }

            solver.add(box.box.left, box.box.top, box.box.right, box.box.bottom, text, base_font_size);
            label_colors.push_back(color);
            label_texts.push_back(text);
        }
    }

    // 3. 求解
    solver.solve();

    // 4. 绘制文字
    auto results = solver.getResults();

    for (size_t i = 0; i < results.size() && i < label_colors.size(); ++i) {
        const auto& res = results[i];
        
        cv::Rect bg_rect(static_cast<int>(res.x), static_cast<int>(res.y), res.width, res.height);
        bg_rect &= cv::Rect(0, 0, width, height);
        if (bg_rect.area() <= 0) continue;

        // 实心填充背景
        // cv::rectangle(img, bg_rect, label_colors[i], cv::FILLED);
        
        // 绘制文字 (修正 Baseline 偏移)
        int text_x = bg_rect.x + PAD_X;
        // 坐标 = BoxTop + Padding + Ascent
        int text_y = bg_rect.y + PAD_Y + res.textAscent;

        text_renderer.putText(img, label_texts[i], cv::Point(text_x, text_y), label_colors[i], res.fontSize);
    }
}

// 多边形重载
void osd(cv::Mat &img, const std::unordered_map<std::string, std::vector<std::tuple<float, float>>> &points, const cv::Scalar &color, double font_scale_ratio) {
    int h = img.rows, w = img.cols;
    int fs = std::max(20, static_cast<int>(std::min(w, h) * font_scale_ratio));
    int th = std::max(1, fs / 7);
    for (const auto &[l, pts] : points) {
        if (pts.empty()) continue;
        drawPolygon(img, pts, color, th);
        if (pts.size() >= 3) {
            auto c = calculatePolygonCentroid(pts);
            text_renderer.putText(img, l, cv::Point(std::get<0>(c), std::get<1>(c)), color, fs);
        }
    }
}

void osd(cv::Mat &img, const std::string &fence_name, const std::vector<std::tuple<float, float>> &points, const cv::Scalar &color, double font_scale_ratio) {
    int h = img.rows, w = img.cols;
    int fs = std::max(20, static_cast<int>(std::min(w, h) * font_scale_ratio));
    int th = std::max(1, fs / 7);
    if (points.empty()) return;
    drawPolygon(img, points, color, th);
    if (points.size() >= 3) {
        auto c = calculatePolygonCentroid(points);
        text_renderer.putText(img, fence_name, cv::Point(std::get<0>(c), std::get<1>(c)), color, fs);
    }
}

void osd(cv::Mat &img, const std::tuple<float, float> &position, const std::string &text, const cv::Scalar &color, int font_size) {
    text_renderer.putText(img, text, cv::Point(std::get<0>(position), std::get<1>(position)), color, font_size);
}