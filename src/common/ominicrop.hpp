#ifndef OMNICROP_HPP
#define OMNICROP_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <queue>

namespace omnicrop
{

    struct BBox
    {
        float x1, y1, x2, y2;
        BBox() : x1(0), y1(0), x2(0), y2(0) {}
        BBox(float _x1, float _y1, float _x2, float _y2) : x1(_x1), y1(_y1), x2(_x2), y2(_y2) {}

        inline float width() const { return std::max(0.0f, x2 - x1); }
        inline float height() const { return std::max(0.0f, y2 - y1); }
        inline float area() const { return width() * height(); }
        inline float center_x() const { return (x1 + x2) * 0.5f; }
        inline float center_y() const { return (y1 + y2) * 0.5f; }

        static BBox merge(const BBox &a, const BBox &b)
        {
            return BBox(std::min(a.x1, b.x1), std::min(a.y1, b.y1),
                        std::max(a.x2, b.x2), std::max(a.y2, b.y2));
        }

        inline float iou(const BBox &other) const
        {
            float xx1 = std::max(x1, other.x1);
            float yy1 = std::max(y1, other.y1);
            float xx2 = std::min(x2, other.x2);
            float yy2 = std::min(y2, other.y2);
            float inter_area = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
            float union_area = area() + other.area() - inter_area;
            return (union_area <= 0) ? 0.0f : inter_area / union_area;
        }
    };

    struct Config
    {
        float w_diou = 10.0f;
        float w_expansion = 5.0f;
        // 迭代优化的惩罚项：每多一个 Crop 增加的“成本”感。
        // 增加此值会更倾向于合并，减小此值会更倾向于保留独立的小框。
        float crop_count_penalty = 50.0f;

        float nms_threshold = 0.3f;
        bool enable_aspect_ratio_fix = true;
        float target_aspect_ratio = 1.0f;
    };

    struct Cluster
    {
        BBox bbox;
        float person_area_sum;
        bool active;
        int generation;

        Cluster(BBox _bbox, float _sum)
            : bbox(_bbox), person_area_sum(_sum), active(true), generation(0) {}
    };

    struct MergeCandidate
    {
        float loss;
        int u, v;
        int gen_u, gen_v;
        bool operator>(const MergeCandidate &other) const { return loss > other.loss; }
    };

    class OmniCropEngine
    {
    public:
        OmniCropEngine(int max_crop_size = 1008, int padding = 30)
            : max_crop_size_(max_crop_size), padding_(padding) {}

        std::vector<BBox> cluster_and_crop(const std::vector<BBox> &boxes, int img_w, int img_h, Config cfg = Config())
        {
            if (boxes.empty())
                return {};

            std::vector<Cluster> clusters;
            float total_person_area = 0;
            for (const auto &box : boxes)
            {
                clusters.emplace_back(box, box.area());
                total_person_area += box.area();
            }

            // 用于保存每一个迭代步骤的状态
            struct State
            {
                std::vector<BBox> crops;
                float score;
            };
            std::vector<State> history;

            // 初始状态：每个目标一个框
            auto record_state = [&]()
            {
                std::vector<BBox> current_crops;
                float total_crop_area = 0;
                for (const auto &c : clusters)
                {
                    if (c.active)
                    {
                        BBox final_b = safe_finalize(c.bbox, img_w, img_h, cfg);
                        current_crops.push_back(final_b);
                        total_crop_area += final_b.area();
                    }
                }
                // 核心评价指标：利用率越大越好，框的数量越少越好
                // Score = (总目标面积 / 总裁剪面积) - (框数量 * 惩罚)
                float utilization = total_person_area / (total_crop_area + 1e-6f);
                float score = utilization * 100.0f - (current_crops.size() * cfg.crop_count_penalty / boxes.size());
                history.push_back({current_crops, score});
            };

            record_state();

            // 优先级队列驱动的贪婪合并
            std::priority_queue<MergeCandidate, std::vector<MergeCandidate>, std::greater<MergeCandidate>> pq;
            auto add_to_pq = [&](int i, int j)
            {
                float loss = calculate_affinity_loss(clusters[i], clusters[j], cfg);
                if (loss < std::numeric_limits<float>::infinity())
                {
                    pq.push({loss, i, j, clusters[i].generation, clusters[j].generation});
                }
            };

            for (size_t i = 0; i < clusters.size(); ++i)
            {
                for (size_t j = i + 1; j < clusters.size(); ++j)
                    add_to_pq((int)i, (int)j);
            }

            while (!pq.empty())
            {
                MergeCandidate top = pq.top();
                pq.pop();

                if (!clusters[top.u].active || !clusters[top.v].active)
                    continue;
                if (clusters[top.u].generation != top.gen_u || clusters[top.v].generation != top.gen_v)
                    continue;

                // 执行合并
                clusters[top.u].bbox = BBox::merge(clusters[top.u].bbox, clusters[top.v].bbox);
                clusters[top.u].person_area_sum += clusters[top.v].person_area_sum;
                clusters[top.u].generation++;
                clusters[top.v].active = false;

                // 记录合并后的新状态
                record_state();

                for (size_t k = 0; k < clusters.size(); ++k)
                {
                    if (clusters[k].active && (int)k != top.u)
                        add_to_pq(top.u, (int)k);
                }
            }

            // 从历史中选择得分最高的状态
            auto best_it = std::max_element(history.begin(), history.end(), [](const State &a, const State &b)
                                            { return a.score < b.score; });

            // 最后的 Crop 级别融合（处理可能存在的重叠）
            return resolve_overlaps(best_it->crops, img_w, img_h, cfg);
        }

    private:
        int max_crop_size_;
        int padding_;

        float calculate_affinity_loss(const Cluster &a, const Cluster &b, const Config &cfg) const
        {
            BBox union_box = BBox::merge(a.bbox, b.bbox);
            // 硬约束：合并后不能超过最大裁剪尺寸
            if (union_box.width() > max_crop_size_ || union_box.height() > max_crop_size_)
                return std::numeric_limits<float>::infinity();

            float center_dist_sq = std::pow(a.bbox.center_x() - b.bbox.center_x(), 2) +
                                   std::pow(a.bbox.center_y() - b.bbox.center_y(), 2);
            float diag_sq = std::pow(union_box.width(), 2) + std::pow(union_box.height(), 2) + 1e-6f;

            // Expansion 越小，说明合并后浪费的空间越少
            float expansion = 1.0f - ((a.person_area_sum + b.person_area_sum) / (union_box.area() + 1e-6f));

            return cfg.w_diou * (center_dist_sq / diag_sq) + cfg.w_expansion * expansion;
        }

        BBox safe_finalize(const BBox &b, int img_w, int img_h, const Config &cfg) const
        {
            float tw = b.width() + 2 * padding_;
            float th = b.height() + 2 * padding_;

            if (cfg.enable_aspect_ratio_fix)
            {
                float current_ar = tw / (th + 1e-6f);
                if (current_ar < cfg.target_aspect_ratio)
                    tw = th * cfg.target_aspect_ratio;
                else
                    th = tw / cfg.target_aspect_ratio;
            }

            tw = std::max(std::min({tw, (float)max_crop_size_, (float)img_w}), b.width());
            th = std::max(std::min({th, (float)max_crop_size_, (float)img_h}), b.height());

            float x1 = std::min(std::max(0.0f, b.center_x() - tw * 0.5f), b.x1);
            float y1 = std::min(std::max(0.0f, b.center_y() - th * 0.5f), b.y1);

            if (x1 + tw > img_w)
                x1 = img_w - tw;
            if (y1 + th > img_h)
                y1 = img_h - th;

            return BBox(x1, y1, x1 + tw, y1 + th);
        }

        std::vector<BBox> resolve_overlaps(const std::vector<BBox> &crops, int img_w, int img_h, const Config &cfg)
        {
            if (crops.empty())
                return {};
            std::vector<BBox> result = crops;
            bool merged = true;
            while (merged)
            {
                merged = false;
                for (size_t i = 0; i < result.size(); ++i)
                {
                    for (size_t j = i + 1; j < result.size(); ++j)
                    {
                        if (result[i].iou(result[j]) > cfg.nms_threshold)
                        {
                            BBox new_union = BBox::merge(result[i], result[j]);
                            if (new_union.width() <= max_crop_size_ && new_union.height() <= max_crop_size_)
                            {
                                result[i] = safe_finalize(new_union, img_w, img_h, cfg);
                                result.erase(result.begin() + j);
                                merged = true;
                                break;
                            }
                        }
                    }
                    if (merged)
                        break;
                }
            }
            return result;
        }
    };

}

#endif