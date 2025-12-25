#ifndef LABEL_LAYOUT_SOLVER_HPP
#define LABEL_LAYOUT_SOLVER_HPP

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <functional>

// ==========================================
// 1. 基础几何结构
// ==========================================
struct LayoutBox {
    float left, top, right, bottom;

    float width() const { return right - left; }
    float height() const { return bottom - top; }
    float area() const { return std::max(0.0f, width()) * std::max(0.0f, height()); }
    float centerX() const { return (left + right) / 2.0f; }
    float centerY() const { return (top + bottom) / 2.0f; }

    static bool intersects(const LayoutBox& a, const LayoutBox& b) {
        return (a.left < b.right && a.right > b.left &&
                a.top < b.bottom && a.bottom > b.top);
    }
    static float intersectArea(const LayoutBox& a, const LayoutBox& b) {
        float l = std::max(a.left, b.left);
        float t = std::max(a.top, b.top);
        float r = std::min(a.right, b.right);
        float bb = std::min(a.bottom, b.bottom);
        if (l < r && t < bb) return (r - l) * (bb - t);
        return 0.0f;
    }
};

struct TextSize {
    int width;
    int height;     // Ascent
    int baseline;   // Descent
};

struct LayoutResult {
    float x, y;
    int fontSize;
    int width;
    int height;
    int textAscent; 
    int textDescent;
};

// ==========================================
// 2. 空间索引：均匀网格 (Uniform Grid)
// ==========================================
class UniformGrid {
public:
    int rows, cols;
    float cellW, cellH;
    std::vector<std::vector<int>> cells;

    UniformGrid(int w, int h, int gridSize = 100) {
        if (gridSize <= 0) gridSize = 100;
        cols = (w + gridSize - 1) / gridSize;
        rows = (h + gridSize - 1) / gridSize;
        cellW = (float)gridSize;
        cellH = (float)gridSize;
        cells.resize(rows * cols);
    }

    void clear() {
        for (auto& c : cells) c.clear();
    }

    void insert(int id, const LayoutBox& box) {
        int c1 = std::max(0, std::min(cols - 1, (int)(box.left / cellW)));
        int r1 = std::max(0, std::min(rows - 1, (int)(box.top / cellH)));
        int c2 = std::max(0, std::min(cols - 1, (int)(box.right / cellW)));
        int r2 = std::max(0, std::min(rows - 1, (int)(box.bottom / cellH)));

        for (int r = r1; r <= r2; ++r) {
            for (int c = c1; c <= c2; ++c) {
                cells[r * cols + c].push_back(id);
            }
        }
    }

    // 查询并自动去重
    void query(const LayoutBox& box, std::vector<int>& outIds, std::vector<int>& visited, int cookie) {
        outIds.clear();
        int c1 = std::max(0, std::min(cols - 1, (int)(box.left / cellW)));
        int r1 = std::max(0, std::min(rows - 1, (int)(box.top / cellH)));
        int c2 = std::max(0, std::min(cols - 1, (int)(box.right / cellW)));
        int r2 = std::max(0, std::min(rows - 1, (int)(box.bottom / cellH)));

        for (int r = r1; r <= r2; ++r) {
            for (int c = c1; c <= c2; ++c) {
                const auto& cellIds = cells[r * cols + c];
                for (int id : cellIds) {
                    if (visited[id] != cookie) {
                        visited[id] = cookie; // 标记已访问
                        outIds.push_back(id);
                    }
                }
            }
        }
    }
};

// ==========================================
// 3. 布局求解器类
// ==========================================
class LabelLayoutSolver {
public:
    struct Candidate {
        LayoutBox box;
        float baseCost;      // 包含 位置偏好 + 缩放惩罚
        float occlusionCost; // 包含 遮挡物体惩罚
        int fontSize;
        int textAscent;
        int textDescent;
    };

    struct LayoutItem {
        int id;
        LayoutBox objectBox;
        std::string text;
        int baseFontSize;
        std::vector<Candidate> candidates;
        int selectedIndex;
        LayoutBox currentBox;
    };

private:
    std::vector<LayoutItem> items;
    int canvasWidth;
    int canvasHeight;
    std::function<TextSize(const std::string&, int)> measureFunc;

    // --- 权重配置 ---
    const float COST_TL_OUTER = 0.0f;    
    const float COST_TL_INNER = 50.0f;  
    const float COST_BL_OUTER = 10.0f;   
    const float COST_BL_INNER = 60.0f;  
    const float COST_TR_OUTER = 20.0f;   
    const float COST_TR_INNER = 70.0f;  
    const float COST_BR_OUTER = 30.0f;   
    const float COST_BR_INNER = 80.0f;  
    const float COST_SIDE     = 40.0f;

    const float COST_SLIDING_PENALTY = 5.0f; 

    // 【缩放惩罚阶梯】：100万，确保优先选 1.0x
    const float COST_SCALE_TIER = 1000000.0f; 

    const float COST_OCCLUDE_OBJ = 50000.0f;
    const float COST_OVERLAP_LABEL = 5000000.0f; 

    const int PAD_X = 2;
    const int PAD_Y = 2;

public:
    template <typename Func>
    LabelLayoutSolver(int w, int h, Func&& func) 
        : canvasWidth(w), canvasHeight(h), measureFunc(std::forward<Func>(func)) {}

    void clear() { items.clear(); }

    void add(float l, float t, float r, float b, const std::string& text, int baseFontSize) {
        LayoutBox objBox = {std::floor(l), std::floor(t), std::ceil(r), std::ceil(b)};
        
        LayoutItem item;
        item.id = (int)items.size();
        item.objectBox = objBox;
        item.text = text;
        item.baseFontSize = baseFontSize;

        item.candidates = generateCandidates(objBox, text, baseFontSize);
        
        // 初始保底
        if (!item.candidates.empty()) {
            item.selectedIndex = 0;
            item.currentBox = item.candidates[0].box;
        } else {
            TextSize ts = measureFunc(text, 10);
            float h = (float)(ts.height + ts.baseline + PAD_Y * 2);
            item.candidates.push_back({{0, 0, (float)ts.width, h}, 9999999.0f, 0.0f, 10, ts.height, ts.baseline});
            item.selectedIndex = 0;
            item.currentBox = item.candidates[0].box;
        }

        items.push_back(item);
    }

    void solve() {
        if (items.empty()) return;
        size_t N = items.size();

        UniformGrid grid(canvasWidth, canvasHeight, 100); 
        std::vector<int> visited(N, 0); 
        int cookie = 0;
        std::vector<int> nearbyIds;
        nearbyIds.reserve(N);

        // 1. 静态环境分析
        grid.clear();
        for (const auto& item : items) {
            grid.insert(item.id, item.objectBox);
        }

        for (auto& item : items) {
            for (auto& cand : item.candidates) {
                float penalty = 0.0f;
                float cArea = cand.box.area();
                if (cArea < 0.1f) continue;

                cookie++;
                grid.query(cand.box, nearbyIds, visited, cookie);

                for (int otherId : nearbyIds) {
                    const auto& other = items[otherId];
                    float inter = LayoutBox::intersectArea(cand.box, other.objectBox);
                    if (inter > 0) {
                        if (item.id != other.id) penalty += (inter / cArea) * COST_OCCLUDE_OBJ;
                        else penalty += (inter / cArea) * 500.0f; 
                    }
                }
                cand.occlusionCost = penalty;
            }

            std::sort(item.candidates.begin(), item.candidates.end(), 
                [](const Candidate& a, const Candidate& b) {
                    return (a.baseCost + a.occlusionCost) < (b.baseCost + b.occlusionCost);
                });
            
            item.selectedIndex = 0;
            item.currentBox = item.candidates[0].box;
        }

        // 2. 动态冲突解决
        int maxIter = 50; 
        bool changed = true;

        for (int iter = 0; iter < maxIter && changed; ++iter) {
            changed = false;

            grid.clear();
            for (const auto& item : items) {
                grid.insert(item.id, item.currentBox);
            }

            for (size_t i = 0; i < items.size(); ++i) {
                LayoutItem& itemA = items[i];
                
                bool hasConflict = false;
                cookie++;
                grid.query(itemA.currentBox, nearbyIds, visited, cookie);

                for (int j : nearbyIds) {
                    if (i == (size_t)j) continue;
                    if (LayoutBox::intersects(itemA.currentBox, items[j].currentBox)) {
                        hasConflict = true;
                        break;
                    }
                }

                if (hasConflict) {
                    int bestIdx = -1;
                    float minTotalCost = std::numeric_limits<float>::max();
                    int checkCount = 0;

                    for (int c = 0; c < (int)itemA.candidates.size(); ++c) {
                        const auto& cand = itemA.candidates[c];
                        float currentCost = cand.baseCost + cand.occlusionCost;

                        if (c > 0 && currentCost > minTotalCost && currentCost > COST_OVERLAP_LABEL) break;

                        bool overlapLabel = false;
                        cookie++;
                        grid.query(cand.box, nearbyIds, visited, cookie);

                        for (int j : nearbyIds) {
                            if (i == (size_t)j) continue;
                            if (LayoutBox::intersects(cand.box, items[j].currentBox)) {
                                overlapLabel = true; break;
                            }
                        }

                        if (overlapLabel) currentCost += COST_OVERLAP_LABEL;

                        if (currentCost < minTotalCost) {
                            minTotalCost = currentCost;
                            bestIdx = c;
                        }

                        checkCount++;
                        if (checkCount > 150 && minTotalCost < COST_OVERLAP_LABEL) break;
                    }

                    if (bestIdx != -1 && bestIdx != itemA.selectedIndex) {
                        itemA.selectedIndex = bestIdx;
                        itemA.currentBox = itemA.candidates[bestIdx].box;
                        changed = true;
                    }
                }
            }
        }
    }

    std::vector<LayoutResult> getResults() const {
        std::vector<LayoutResult> results;
        results.reserve(items.size());
        for (const auto& item : items) {
            const auto& cand = item.candidates[item.selectedIndex];
            results.push_back({
                cand.box.left, cand.box.top, 
                cand.fontSize, 
                (int)cand.box.width(), (int)cand.box.height(),
                cand.textAscent, cand.textDescent
            });
        }
        return results;
    }

private:
    std::vector<Candidate> generateCandidates(const LayoutBox& obj, const std::string& text, int baseSize) {
        std::vector<Candidate> cands;
        cands.reserve(256); 

        struct ScaleLevel { float scale; int tier; };
        std::vector<ScaleLevel> levels = {
            {1.0f, 0}, {0.9f, 1}, {0.8f, 2}, {0.7f, 3}, {0.6f, 4}, {0.5f, 5}
        };

        for (const auto& lvl : levels) {
            int fontSize = static_cast<int>(baseSize * lvl.scale);
            if (fontSize < 9) break;

            TextSize ts = measureFunc(text, fontSize);
            
            float w = std::ceil((float)ts.width + PAD_X * 2);
            float h = std::ceil((float)(ts.height + ts.baseline + PAD_Y * 2));
            float scalePenalty = lvl.tier * COST_SCALE_TIER;

            int steps = (lvl.tier <= 1) ? 12 : 4;

            // 水平滑动
            float minX = obj.left;
            float maxX = std::max(obj.left, obj.right - w);
            for (int i = 0; i <= steps; ++i) {
                float r = (float)i / steps; 
                float x = minX + (maxX - minX) * r;
                float dist = std::min(r, 1.0f - r); 
                float posP = dist * COST_SLIDING_PENALTY * 2.0f; 

                tryAdd(cands, x, obj.top - h, w, h, COST_TL_OUTER + scalePenalty + posP, fontSize, ts);
                tryAdd(cands, x, obj.top, w, h, COST_TL_INNER + scalePenalty + posP, fontSize, ts);
                tryAdd(cands, x, obj.bottom, w, h, COST_BL_OUTER + scalePenalty + posP, fontSize, ts);
                tryAdd(cands, x, obj.bottom - h, w, h, COST_BL_INNER + scalePenalty + posP, fontSize, ts);
            }

            // 垂直滑动
            float minY = obj.top;
            float maxY = std::max(obj.top, obj.bottom - h);
            for (int i = 0; i <= steps; ++i) {
                float r = (float)i / steps;
                float y = minY + (maxY - minY) * r;
                float posP = COST_SIDE + std::min(r, 1.0f - r) * COST_SLIDING_PENALTY * 2.0f;

                tryAdd(cands, obj.left - w, y, w, h, posP + scalePenalty, fontSize, ts);
                tryAdd(cands, obj.right, y, w, h, posP + scalePenalty, fontSize, ts);
            }
        }
        return cands;
    }

    void tryAdd(std::vector<Candidate>& cands, float x, float y, float w, float h, 
                float cost, int fs, const TextSize& ts) {
        if (x < 0 || y < 0 || x + w > canvasWidth || y + h > canvasHeight) return;
        Candidate cand;
        cand.box = {std::floor(x), std::floor(y), std::floor(x + w), std::floor(y + h)};
        cand.baseCost = cost;
        cand.occlusionCost = 0.0f;
        cand.fontSize = fs;
        cand.textAscent = ts.height;
        cand.textDescent = ts.baseline;
        cands.push_back(cand);
    }
};

#endif