#ifndef POSITION_HPP__
#define POSITION_HPP__

#include <algorithm>
#include <functional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <list>

template <typename T>
struct Box
{
    T l, t, r, b;
};

template <typename T>
constexpr T area(const Box<T> &b)
{
    return std::max(T(0), b.r - b.l) * std::max(T(0), b.b - b.t);
}

template <typename T>
constexpr T intersectionArea(const Box<T> &a, const Box<T> &b)
{
    const T w = std::max(T(0), std::min(a.r, b.r) - std::max(a.l, b.l));
    const T h = std::max(T(0), std::min(a.b, b.b) - std::max(a.t, b.t));
    return w * h;
}

template <typename T>
inline float computeIoU(const Box<T> &a, const Box<T> &b)
{
    const T inter = intersectionArea(a, b);
    const T unionArea = area(a) + area(b) - inter;
    return (unionArea > 0) ? static_cast<float>(inter) / unionArea : 0.f;
}

template <typename T>
inline float computeOverlap(const Box<T> &a, const Box<T> &b)
{
    const T inter = intersectionArea(a, b);
    const T minArea = std::min(area(a), area(b));
    return (minArea > 0) ? static_cast<float>(inter) / minArea : 0.f;
}

template <typename T>
class SpatialIndex
{
public:
    SpatialIndex(int gridSize = 100) : gridSize(gridSize) {}

    void insert(const Box<T> &box)
    {
        forEachCell(box, [&](Key key)
                    { grid[key].push_back(&boxRefStorage.emplace_back(box)); });
    }

    void clear()
    {
        grid.clear();
        boxRefStorage.clear();
    }

    std::vector<const Box<T> *> query(const Box<T> &region) const
    {
        std::vector<const Box<T> *> results;
        std::unordered_set<const Box<T> *> seen;
        forEachCell(region, [&](Key key)
                    {
            auto it = grid.find(key);
            if (it != grid.end()) {
                for (auto* b : it->second) {
                    if (seen.find(b) == seen.end()) {
                        results.push_back(b);
                        seen.insert(b);
                    }
                }
            } });
        return results;
    }

private:
    struct Key
    {
        int x, y;
        bool operator==(const Key &o) const { return x == o.x && y == o.y; }
    };
    struct KeyHash
    {
        std::size_t operator()(const Key &k) const
        {
            return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1);
        }
    };

    void forEachCell(const Box<T> &box, const std::function<void(Key)> &fn) const
    {
        int gx0 = static_cast<int>(box.l) / gridSize;
        int gy0 = static_cast<int>(box.t) / gridSize;
        int gx1 = static_cast<int>(box.r) / gridSize;
        int gy1 = static_cast<int>(box.b) / gridSize;
        for (int gx = gx0; gx <= gx1; ++gx)
            for (int gy = gy0; gy <= gy1; ++gy)
                fn({gx, gy});
    }

    int gridSize;
    std::unordered_map<Key, std::vector<const Box<T> *>, KeyHash> grid;
    std::list<Box<T>> boxRefStorage;
};

// ===== 主类，外部接口不变 =====
template <typename T>
class PositionManager
{
private:
    std::function<std::tuple<int, int, int>(const std::string &)> getFontSizeFunc;
    SpatialIndex<T> index;

    static Box<T> tupleToBox(const std::tuple<T, T, T, T> &tpl)
    {
        return {std::get<0>(tpl), std::get<1>(tpl), std::get<2>(tpl), std::get<3>(tpl)};
    }

public:
    template <typename Func>
    PositionManager(Func &&fontSizeFunc, int gridSize = 100)
        : getFontSizeFunc(std::forward<Func>(fontSizeFunc)), index(gridSize) {}

    std::tuple<T, T> selectOptimalPosition(const std::tuple<T, T, T, T> &box,
                                           int canvasWidth, int canvasHeight,
                                           const std::string &text)
    {
        int textWidth, textHeight, baseline;
        std::tie(textWidth, textHeight, baseline) = getFontSizeFunc(text);

        auto candidates = findCandidatePositions(box, canvasWidth, canvasHeight,
                                                 textWidth, textHeight, baseline);

        if (candidates.empty())
        {
            // 如果一个候选位置都没有（例如文本框比画布还大），需要一个安全的回退策略
            Box<T> fallbackBox = {T(0), T(0), T(textWidth), T(textHeight)};
            index.insert(fallbackBox);
            return {fallbackBox.l, fallbackBox.t + textHeight};
        }

        float minIoU = 1.1f; // 初始值设为大于1，确保任何合法值都能替换它
        Box<T> best = tupleToBox(candidates.front());

        for (const auto &cposition : candidates)
        {
            Box<T> cb = tupleToBox(cposition);
            float maxIoU = 0.f;

            // 查询时只与可能碰撞的 Box 计算 IoU
            for (auto *m : index.query(cb))
            {
                maxIoU = std::max(maxIoU, computeIoU(cb, *m));
            }

            if (maxIoU == 0.f) // 找到完全不重叠的位置，直接采纳
            {
                best = cb;
                minIoU = 0.f;
                break;
            }

            if (maxIoU < minIoU)
            {
                minIoU = maxIoU;
                best = cb;
            }
        }
        index.insert(best);
        return {best.l, best.t + textHeight};
    }

    void clearMarkedPositions()
    {
        index.clear();
    }

    std::vector<std::tuple<T, T, T, T>> findCandidatePositions(
        const std::tuple<T, T, T, T> &box, int canvasWidth,
        int canvasHeight, int textWidth, int textHeight, int baseline)
    {
        std::vector<std::tuple<T, T, T, T>> candidates;
        candidates.reserve(10);

        T left, top, right, bottom;
        std::tie(left, top, right, bottom) = box;

        Box<T> canvas{0, 0, T(canvasWidth), T(canvasHeight)};

        const std::vector<std::tuple<T, T, T, T>> positions = {
            {left, top - textHeight - baseline, left + textWidth, top},
            {right, top, right + textWidth, top + textHeight + baseline},
            {left - textWidth, top, left, top + textHeight + baseline},
            {left, bottom, left + textWidth, bottom + textHeight + baseline},
            {right - textWidth, top - textHeight - baseline, right, top},
            {right - textWidth, bottom, right, bottom + textHeight + baseline},
            {left, top, left + textWidth, top + textHeight + baseline},
            {right - textWidth, top, right, top + textHeight + baseline},
            {right - textWidth, bottom - textHeight - baseline, right, bottom},
            {left, bottom - textHeight - baseline, left + textWidth, bottom}};

        for (const auto &p : positions)
        {
            Box<T> pb = tupleToBox(p);
            // 使用直接的坐标比较代替浮点数运算，更清晰且稳健
            if (pb.l >= canvas.l && pb.t >= canvas.t && pb.r <= canvas.r && pb.b <= canvas.b)
            {
                candidates.push_back(p);
            }
        }
        if (candidates.empty())
        {
            // 如果所有预设位置都不在画布内，尝试一个基本位置
            Box<T> baseBox = {left, top, left + textWidth, top + textHeight + baseline};
            if (intersectionArea(canvas, baseBox) > 0)
            { // 只要有部分重叠即可
                candidates.push_back({left, top, left + textWidth, top + textHeight + baseline});
            }
        }
        return candidates;
    }
};

#endif // POSITION_HPP__