#ifndef LABEL_LAYOUT_SOLVER_HPP
#define LABEL_LAYOUT_SOLVER_HPP

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <functional>
#include <cstring>
#include <cstdint>
#include <random>

struct LayoutBox {
    float left, top, right, bottom;

    inline float width() const { return right - left; }
    inline float height() const { return bottom - top; }
    inline float area() const { return std::max(0.0f, right - left) * std::max(0.0f, bottom - top); }

    static inline float intersectArea(const LayoutBox& box1, const LayoutBox& box2) {
        float l = std::max(box1.left, box2.left);
        float r = std::min(box1.right, box2.right);
        float t = std::max(box1.top, box2.top);
        float b = std::min(box1.bottom, box2.bottom);
        float w = std::max(0.0f, r - l);
        float h = std::max(0.0f, b - t);
        return w * h;
    }

    static inline bool intersects(const LayoutBox& box1, const LayoutBox& box2) {
        return (box1.left < box2.right && box1.right > box2.left &&
                box1.top < box2.bottom && box1.bottom > box2.top);
    }
};

struct TextSize {
    int width, height, baseline;
};

struct LayoutResult {
    float x, y;
    int fontSize;
    int width;
    int height;
    int textAscent;
};

struct LayoutConfig {
    int gridSize = 100;
    int spatialIndexThreshold = 20;
    int maxIterations = 20;
    int paddingX = 2;
    int paddingY = 2;

    float costTlOuter = 0.0f;
    float costTrOuter = 10.0f;
    float costBlOuter = 20.0f;
    float costBrOuter = 30.0f;
    float costSide    = 40.0f;

    float costSlidingPenalty = 5.0f;
    float costScaleTier      = 10000.0f; 
    float costOccludeObj     = 100000.0f;  
    float costOverlapBase    = 100000.0f;
};


class FlatUniformGrid {
public:
    int rows = 0, cols = 0;
    float cellW = 100.0f, cellH = 100.0f;
    float invCellW = 0.01f, invCellH = 0.01f;
    
    std::vector<int> gridHead;
    struct Node { int id; int next; };
    std::vector<Node> nodes; 

    FlatUniformGrid() { nodes.reserve(4096); }

    void resize(int w, int h, int gridSize) {
        if (gridSize <= 0) gridSize = 100;
        int newCols = (w + gridSize - 1) / gridSize;
        int newRows = (h + gridSize - 1) / gridSize;
        cellW = (float)gridSize;
        cellH = (float)gridSize;
        invCellW = 1.0f / cellW;
        invCellH = 1.0f / cellH;

        if (newCols * newRows > (int)gridHead.size()) {
            gridHead.resize(newCols * newRows, -1);
        }
        cols = newCols;
        rows = newRows;
    }

    void clear() {
        if (!gridHead.empty()) {
            std::fill(gridHead.begin(), gridHead.begin() + (rows * cols), -1);
        }
        nodes.clear();
    }

    inline void insert(int id, const LayoutBox& box) {
        int c1 = std::max(0, std::min(cols - 1, (int)(box.left * invCellW)));
        int r1 = std::max(0, std::min(rows - 1, (int)(box.top * invCellH)));
        int c2 = std::max(0, std::min(cols - 1, (int)(box.right * invCellW)));
        int r2 = std::max(0, std::min(rows - 1, (int)(box.bottom * invCellH)));

        for (int r = r1; r <= r2; ++r) {
            int rowOffset = r * cols;
            for (int c = c1; c <= c2; ++c) {
                int idx = rowOffset + c;
                nodes.push_back({id, gridHead[idx]});
                gridHead[idx] = (int)nodes.size() - 1;
            }
        }
    }

    template <typename Visitor>
    inline void query(const LayoutBox& box, std::vector<int>& visitedToken, int cookie, Visitor&& visitor) {
        int c1 = std::max(0, std::min(cols - 1, (int)(box.left * invCellW)));
        int r1 = std::max(0, std::min(rows - 1, (int)(box.top * invCellH)));
        int c2 = std::max(0, std::min(cols - 1, (int)(box.right * invCellW)));
        int r2 = std::max(0, std::min(rows - 1, (int)(box.bottom * invCellH)));

        for (int r = r1; r <= r2; ++r) {
            int rowOffset = r * cols;
            for (int c = c1; c <= c2; ++c) {
                int nodeIdx = gridHead[rowOffset + c];
                while (nodeIdx != -1) {
                    const auto& node = nodes[nodeIdx];
                    if (visitedToken[node.id] != cookie) {
                        visitedToken[node.id] = cookie;
                        visitor(node.id);
                    }
                    nodeIdx = node.next;
                }
            }
        }
    }
};


class LabelLayoutSolver {
public:
    struct Candidate {
        LayoutBox box;
        float geometricCost; 
        float staticCost;    
        float area;          
        float invArea;       
        int16_t fontSize;
        int16_t textAscent;
    };

private:
    struct LayoutItem {
        int id;
        LayoutBox objectBox; 
        uint32_t candStart; 
        uint16_t candCount; 
        int selectedRelIndex; 
        LayoutBox currentBox;
        float currentArea;
        float currentTotalCost;
    };

    LayoutConfig config;
    int canvasWidth, canvasHeight;
    std::function<TextSize(const std::string&, int)> measureFunc;

    std::vector<LayoutItem> items;
    std::vector<Candidate> candidatePool;
    std::vector<int> processOrder; 
    FlatUniformGrid grid;
    std::vector<int> visitedCookie;
    int currentCookie = 0;
    std::mt19937 rng;

public:
    template <typename Func>
    LabelLayoutSolver(int w, int h, Func&& func, const LayoutConfig& cfg = LayoutConfig())
        : config(cfg), canvasWidth(w), canvasHeight(h), measureFunc(std::forward<Func>(func)), rng(12345)
    {
        items.reserve(128);
        candidatePool.reserve(4096); 
        visitedCookie.reserve(128);
    }

    void setConfig(const LayoutConfig& cfg) { config = cfg; }
    void setCanvasSize(int w, int h) { canvasWidth = w; canvasHeight = h; }

    void clear() {
        items.clear();
        candidatePool.clear();
        processOrder.clear();
    }

    void add(float l, float t, float r, float b, const std::string& text, int baseFontSize) {
        if (r - l < 2.0f) { float cx = (l+r)*0.5f; l = cx-1; r = cx+1; }
        if (b - t < 2.0f) { float cy = (t+b)*0.5f; t = cy-1; b = cy+1; }

        LayoutItem item;
        item.id = (int)items.size();
        item.objectBox = {std::floor(l), std::floor(t), std::ceil(r), std::ceil(b)};
        item.candStart = (uint32_t)candidatePool.size();
        
        generateCandidatesInternal(item, text, baseFontSize);
        item.candCount = (uint16_t)(candidatePool.size() - item.candStart);

        if (item.candCount > 0) {
            item.selectedRelIndex = 0;
            const auto& c = candidatePool[item.candStart];
            item.currentBox = c.box;
            item.currentArea = c.area;
            item.currentTotalCost = c.geometricCost;
        } else {
            Candidate dummy;
            dummy.box = {0,0,0,0}; dummy.geometricCost = 1e9f; dummy.staticCost = 0;
            dummy.area = 0.1f; dummy.invArea = 10.0f;
            dummy.fontSize = (int16_t)baseFontSize; dummy.textAscent = 0;
            candidatePool.push_back(dummy);
            item.candCount = 1; item.selectedRelIndex = 0;
            item.currentBox = dummy.box; item.currentArea = 0.1f; item.currentTotalCost = 1e9f;
        }
        items.push_back(std::move(item));
    }

    void solve() {
        if (items.empty()) return;
        const size_t N = items.size();

        if (visitedCookie.size() < N) visitedCookie.resize(N, 0);
        bool useGrid = (N >= (size_t)config.spatialIndexThreshold);

        if (useGrid) {
            grid.resize(canvasWidth, canvasHeight, config.gridSize);
            grid.clear();
            for (const auto& item : items) grid.insert(item.id, item.objectBox);
        }

        for (auto& item : items) {
            float minCost = std::numeric_limits<float>::max();
            int bestIdx = 0;

            for (uint32_t i = 0; i < item.candCount; ++i) {
                Candidate& cand = candidatePool[item.candStart + i];
                float penalty = 0.0f;

                auto checkStaticConflict = [&](int otherId) {
                    const auto& other = items[otherId];
                    // 先做快速 AABB 判定
                    if (LayoutBox::intersects(cand.box, other.objectBox)) {
                        float inter = LayoutBox::intersectArea(cand.box, other.objectBox);
                        penalty += (inter * cand.invArea) * config.costOccludeObj;
                    }
                };

                if (useGrid) {
                    currentCookie++;
                    grid.query(cand.box, visitedCookie, currentCookie, checkStaticConflict);
                } else {
                    for (const auto& other : items) checkStaticConflict(other.id);
                }
                cand.staticCost = penalty;
                
                float total = cand.geometricCost + cand.staticCost;
                if (total < minCost) { minCost = total; bestIdx = (int)i; }
            }
            item.selectedRelIndex = bestIdx;
            const auto& bestCand = candidatePool[item.candStart + bestIdx];
            item.currentBox = bestCand.box;
            item.currentArea = bestCand.area;
            item.currentTotalCost = minCost;
        }

        // 加入随机化与剪枝
        processOrder.resize(N);
        for(size_t i=0; i<N; ++i) processOrder[i] = (int)i;

        for (int iter = 0; iter < config.maxIterations; ++iter) {
            std::shuffle(processOrder.begin(), processOrder.end(), rng);
            int changeCount = 0;

            if (useGrid) {
                grid.clear();
                for (const auto& item : items) grid.insert(item.id, item.currentBox);
            }

            for (int idx : processOrder) {
                auto& item = items[idx];
                
                auto calculateDynamicCost = [&](const LayoutBox& box, float invBoxArea, int selfId) -> float {
                    float overlapCost = 0.0f;
                    auto visitor = [&](int otherId) {
                        if (selfId == otherId) return;
                        const auto& otherBox = items[otherId].currentBox;
                        // 先做 AABB 判定
                        if (LayoutBox::intersects(box, otherBox)) {
                            float inter = LayoutBox::intersectArea(box, otherBox);
                            overlapCost += (inter * invBoxArea) * config.costOverlapBase;
                        }
                    };
                    if (useGrid) {
                        currentCookie++; 
                        grid.query(box, visitedCookie, currentCookie, visitor);
                    } else {
                        for (size_t j = 0; j < N; ++j) visitor((int)j);
                    }
                    return overlapCost;
                };

                const auto& curCand = candidatePool[item.candStart + item.selectedRelIndex];
                float curDyn = calculateDynamicCost(item.currentBox, curCand.invArea, item.id);
                float currentRealTotal = curCand.geometricCost + curCand.staticCost + curDyn;

                if (currentRealTotal < 1.0f) continue; // 足够好，跳过

                float bestIterCost = currentRealTotal;
                int bestRelIdx = -1;

                for (int i = 0; i < (int)item.candCount; ++i) {
                    if (i == item.selectedRelIndex) continue;
                    const auto& cand = candidatePool[item.candStart + i];

                    // 启发式剪枝
                    // 如果基础成本已经超过目前最优，则不需要进行动态重叠计算
                    float baseCost = cand.geometricCost + cand.staticCost;
                    if (baseCost >= bestIterCost) continue;

                    float newOverlap = calculateDynamicCost(cand.box, cand.invArea, item.id);
                    float newTotal = baseCost + newOverlap;

                    if (newTotal < bestIterCost) {
                        bestIterCost = newTotal;
                        bestRelIdx = i;
                    }
                }

                if (bestRelIdx != -1) {
                    item.selectedRelIndex = bestRelIdx;
                    const auto& newCand = candidatePool[item.candStart + bestRelIdx];
                    item.currentBox = newCand.box;
                    item.currentArea = newCand.area;
                    changeCount++;
                }
            }
            if (changeCount == 0) break;
        }
    }

    std::vector<LayoutResult> getResults() const {
        std::vector<LayoutResult> results;
        results.reserve(items.size());
        for (const auto& item : items) {
            const auto& cand = candidatePool[item.candStart + item.selectedRelIndex];
            results.push_back({
                cand.box.left, cand.box.top, (int)cand.fontSize, 
                (int)cand.box.width(), (int)cand.box.height(), (int)cand.textAscent
            });
        }
        return results;
    }

private:
    void generateCandidatesInternal(LayoutItem& item, const std::string& text, int baseFontSize) {
        static const struct { float scale; int tier; } levels[] = {
            {1.0f, 0}, {0.9f, 1}, {0.8f, 2}, {0.75f, 3} 
        };

        const auto& obj = item.objectBox; 
        for (const auto& lvl : levels) {
            int fontSize = (int)(baseFontSize * lvl.scale);
            if (fontSize < 9) break;

            TextSize ts = measureFunc(text, fontSize);
            float fW = std::ceil((float)ts.width + config.paddingX * 2);
            float fH = std::ceil((float)(ts.height + ts.baseline + config.paddingY * 2));
            float scalePenalty = lvl.tier * config.costScaleTier;
            float area = fW * fH;
            float invArea = 1.0f / (area > 0.1f ? area : 1.0f);

            auto addCand = [&](float x, float y, float posCost) {
                if (x < 0 || y < 0 || x + fW > canvasWidth || y + fH > canvasHeight) return;
                candidatePool.emplace_back();
                auto& c = candidatePool.back();
                c.box = {x, y, x + fW, y + fH};
                c.geometricCost = posCost; c.staticCost = 0;
                c.area = area; c.invArea = invArea;
                c.fontSize = (int16_t)fontSize; c.textAscent = (int16_t)ts.height;
            };

            // 采样优化
            int steps = (lvl.tier <= 1) ? 8 : 4; 
            float invSteps = (steps > 0) ? 1.0f / steps : 0.0f;

            // Top/Bottom
            float rangeX = std::max(0.0f, obj.right - fW - obj.left);
            for (int i = 0; i <= steps; ++i) {
                float r = i * invSteps;
                float x = obj.left + rangeX * r;
                float posP = std::abs(r - 0.5f) * 2.0f * config.costSlidingPenalty + scalePenalty;
                addCand(x, obj.top - fH, config.costTlOuter + posP); 
                addCand(x, obj.bottom, config.costBlOuter + posP); 
            }
            // Left/Right
            float rangeY = std::max(0.0f, obj.bottom - fH - obj.top);
            for (int i = 0; i <= steps; ++i) {
                float r = i * invSteps;
                float y = obj.top + rangeY * r;
                float posP = config.costSide + std::abs(r - 0.5f) * 2.0f * config.costSlidingPenalty + scalePenalty;
                addCand(obj.left - fW, y, posP); 
                addCand(obj.right, y, posP); 
            }
        }
    }
};

#endif