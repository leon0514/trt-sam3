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
#include <string>

template<typename T>
const T& clamp(const T& v, const T& lo, const T& hi) {
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

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
    float left, top;
    int fontSize;
    int padding_x;
    int padding_y;
    int width;
    int height;
    int textAscent;
    int textDescent;
};

struct LayoutConfig {
    int gridSize = 100;
    int spatialIndexThreshold = 20;
    int maxIterations = 30;
    int paddingX = 2;
    int paddingY = 2;

    // 基础位置成本
    float costPos1_Top    = 0.0f;   // 对应左上
    float costPos2_Right  = 10.0f;  // 对应右上
    float costPos3_Bottom = 20.0f;  // 对应左下
    float costPos4_Left   = 30.0f;  // 对应左上

    float costSlidingPenalty = 100.0f; 
    float costScaleTier      = 1000.0f; // 降低缩放惩罚，让挤不下时更容易缩小
    
    // 遮挡惩罚
    float costOccludeObj     = 100000.0f; // 遮挡别人的惩罚
    float costSelfOcclude    = 10.0f;     // 新增：遮挡自己（目标框）的惩罚，设得很低
    float costOverlapBase    = 100000.0f; // 标签间重叠惩罚
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

class LabelLayout {
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
    LabelLayout(int w, int h, Func&& func, const LayoutConfig& cfg = LayoutConfig())
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
                    if (LayoutBox::intersects(cand.box, other.objectBox)) {
                        float inter = LayoutBox::intersectArea(cand.box, other.objectBox);
                        float overlapRatio = inter * cand.invArea;
                        // 如果是遮挡自己的目标框，使用极低惩罚
                        float costFactor = (otherId == item.id) ? config.costSelfOcclude : config.costOccludeObj;
                        penalty += overlapRatio * costFactor;
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

                if (currentRealTotal < 1.0f) continue;

                float bestIterCost = currentRealTotal;
                int bestRelIdx = -1;

                for (int i = 0; i < (int)item.candCount; ++i) {
                    if (i == item.selectedRelIndex) continue;
                    const auto& cand = candidatePool[item.candStart + i];

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

    std::vector<LayoutResult> layout() const {
        std::vector<LayoutResult> results;
        results.reserve(items.size());
        for (const auto& item : items) {
            const auto& cand = candidatePool[item.candStart + item.selectedRelIndex];
            results.push_back({
                cand.box.left, cand.box.top, (int)cand.fontSize, (int)config.paddingX, (int)config.paddingY,
                (int)cand.box.width(), (int)cand.box.height(), (int)cand.textAscent, (int)(cand.box.height() - cand.textAscent)
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

            // 核心修正：自动 Clamp 坐标不出画面
            auto addCand = [&](float x, float y, float posCost) {
                float nx = std::max(0.0f, std::min(x, (float)canvasWidth - fW));
                float ny = std::max(0.0f, std::min(y, (float)canvasHeight - fH));
                
                if (fW <= (float)canvasWidth && fH <= (float)canvasHeight) {
                    candidatePool.emplace_back();
                    auto& c = candidatePool.back();
                    c.box = {nx, ny, nx + fW, ny + fH};
                    // 贴边偏移补偿，确保尽量接近原始锚点
                    float shift = std::abs(nx - x) + std::abs(ny - y);
                    c.geometricCost = posCost + scalePenalty + (shift * 0.1f);
                    c.staticCost = 0;
                    c.area = area; 
                    c.invArea = invArea;
                    c.fontSize = (int16_t)fontSize; 
                    c.textAscent = (int16_t)ts.height;
                }
            };
            
            // 全屏时，这些锚点会自动通过 nx/ny 推到对应的内部角
            addCand(obj.left, obj.top - fH, config.costPos1_Top);      // 左上内
            addCand(obj.right, obj.top, config.costPos2_Right);       // 右上内
            addCand(obj.left, obj.bottom, config.costPos3_Bottom);    // 左下内
            addCand(obj.left - fW, obj.top, config.costPos4_Left);    // 左侧内部（也是左上）

            // 滑动采样
            float rangeX = std::max(0.0f, obj.right - fW - obj.left);
            if (rangeX > 1.0f) {
                int stepsX = clamp((int)(rangeX / 40.0f), 3, 15);
                float invStepsX = 1.0f / (float)stepsX;
                for (int i = 1; i < stepsX; ++i) {
                    float r = i * invStepsX;
                    float x = obj.left + rangeX * r;
                    float p = config.costSlidingPenalty + (r * 10.0f);
                    addCand(x, obj.top - fH, config.costPos1_Top + p);
                    addCand(x, obj.bottom, config.costPos3_Bottom + p);
                }
            }

            float rangeY = std::max(0.0f, obj.bottom - fH - obj.top);
            if (rangeY > 1.0f) {
                int stepsY = clamp((int)(rangeY / 40.0f), 3, 15);
                float invStepsY = 1.0f / (float)stepsY;
                for (int i = 1; i < stepsY; ++i) {
                    float r = i * invStepsY;
                    float y = obj.top + rangeY * r;
                    float p = config.costSlidingPenalty + (r * 10.0f);
                    addCand(obj.right, y, config.costPos2_Right + p);
                    addCand(obj.left - fW, y, config.costPos4_Left + p);
                }
            }
        }
    }
};

#endif