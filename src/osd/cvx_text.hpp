// cvx_text.h

#ifndef CVX_TEXT_H
#define CVX_TEXT_H

#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

// FreeType Headers
#include <ft2build.h>
#include FT_FREETYPE_H

class CvxText
{
public:
    explicit CvxText(const char *font_path);
    virtual ~CvxText();

    void putText(cv::Mat &img, const std::string &text, cv::Point org,
                 cv::Scalar color, int font_size);

    // --- NEW: 添加了获取文本尺寸的方法 ---
    /**
     * @brief 计算文本渲染后的大致包围框尺寸
     *
     * @param text 要计算的 UTF-8 字符串
     * @param font_size 字体大小（像素）
     * @param[out] w 存储计算出的宽度
     * @param[out] h 存储计算出的高度
     * @param[out] baseline 存储计算出的基线
     */
    void getTextSize(const std::string &text, int font_size, int *w, int *h, int *baseline);

private:
    CvxText(const CvxText &) = delete;
    CvxText &operator=(const CvxText &) = delete;

    FT_Face getFace(const char *font_path);
    void utf8_to_ucs4(const std::string &str, std::vector<long> &ucs4);

private:
    FT_Library m_library;
    std::map<std::string, FT_Face> m_faces;
};

#endif // CVX_TEXT_H