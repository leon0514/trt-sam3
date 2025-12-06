#include "osd/cvx_text.hpp"
#include <iostream>

CvxText::CvxText(const char *font_path)
{
    if (FT_Init_FreeType(&m_library))
    {
        throw std::runtime_error("错误: 无法初始化 FreeType 库");
    }
    getFace(font_path);
}

CvxText::~CvxText()
{
    for (auto &pair : m_faces)
    {
        FT_Done_Face(pair.second);
    }
    FT_Done_FreeType(m_library);
}

FT_Face CvxText::getFace(const char *font_path)
{
    if (m_faces.count(font_path))
    {
        return m_faces[font_path];
    }

    // 否则，加载新字体
    FT_Face face;
    if (FT_New_Face(m_library, font_path, 0, &face))
    {
        std::cerr << "错误: 无法从文件加载字体 " << font_path << std::endl;
        return nullptr;
    }

    // 缓存并返回
    m_faces[font_path] = face;
    return face;
}

void CvxText::putText(cv::Mat &img, const std::string &text, cv::Point org,
                      cv::Scalar color, int font_size)
{
    // 默认使用构造函数中加载的字体
    if (m_faces.empty())
    {
        // std::cerr << "错误: 没有加载任何字体" << std::endl;
        return;
    }
    FT_Face face = m_faces.begin()->second; // 获取第一个（默认）字体

    // 设置字体大小
    FT_Set_Pixel_Sizes(face, 0, font_size);
    FT_Select_Charmap(face, FT_ENCODING_UNICODE);

    // 将 UTF-8 解码为 Unicode
    std::vector<long> ucs4_codes;
    utf8_to_ucs4(text, ucs4_codes);

    FT_GlyphSlot slot = face->glyph;

    for (long code : ucs4_codes)
    {
        if (FT_Load_Char(face, code, FT_LOAD_RENDER))
        {
            std::cerr << "警告: 无法加载字符码点 " << code << std::endl;
            continue;
        }

        FT_Bitmap &bitmap = slot->bitmap;
        int y_start = org.y - slot->bitmap_top;
        int x_start = org.x + slot->bitmap_left;

        for (unsigned int r = 0; r < bitmap.rows; ++r)
        {
            for (unsigned int c = 0; c < bitmap.width; ++c)
            {
                int y = y_start + r;
                int x = x_start + c;
                if (x >= 0 && x < img.cols && y >= 0 && y < img.rows)
                {
                    unsigned char alpha = bitmap.buffer[r * bitmap.pitch + c];
                    if (alpha == 0)
                        continue;

                    double alpha_norm = alpha / 255.0;
                    cv::Vec3b &dst_pixel = img.at<cv::Vec3b>(y, x);

                    for (int k = 0; k < 3; ++k)
                    {
                        dst_pixel[k] = cv::saturate_cast<uchar>(
                            color.val[k] * alpha_norm + dst_pixel[k] * (1 - alpha_norm));
                    }
                }
            }
        }
        org.x += (slot->advance.x >> 6);
    }
}

void CvxText::getTextSize(const std::string &text, int font_size, int *w, int *h, int *baseline)
{
    if (w)
        *w = 0;
    if (h)
        *h = 0;
    if (baseline)
        *baseline = 0;
    if (m_faces.empty())
        return;

    FT_Face face = m_faces.begin()->second;
    FT_Set_Pixel_Sizes(face, 0, font_size);

    std::vector<long> ucs4_codes;
    utf8_to_ucs4(text, ucs4_codes);

    if (ucs4_codes.empty())
    {
        return;
    }

    int pen_x = 0;
    int max_ascent = 0;  // 最高点（相对于基线）
    int max_descent = 0; // 最低点（相对于基线，为正值）

    for (size_t i = 0; i < ucs4_codes.size(); ++i)
    {
        long code = ucs4_codes[i];
        if (FT_Load_Char(face, code, FT_LOAD_DEFAULT))
        {
            continue;
        }

        FT_GlyphSlot slot = face->glyph;

        int ascent = slot->bitmap_top;
        int descent = slot->bitmap.rows - slot->bitmap_top;

        if (ascent > max_ascent)
            max_ascent = ascent;
        if (descent > max_descent)
            max_descent = descent;

        // 计算最后一个字符的宽度
        if (i == ucs4_codes.size() - 1)
        {
            pen_x += slot->bitmap_left + slot->bitmap.width;
        }
        else
        {
            pen_x += (slot->advance.x >> 6);
        }
    }

    if (w)
        *w = pen_x;
    if (h)
        *h = max_ascent + max_descent;
    if (baseline)
        *baseline = 0.25 * (*h);
}

void CvxText::utf8_to_ucs4(const std::string &str, std::vector<long> &ucs4)
{
    ucs4.clear();
    for (size_t i = 0; i < str.length();)
    {
        unsigned char c = str[i];
        long code = 0;
        int len = 0;
        if ((c & 0x80) == 0)
        {
            code = c;
            len = 1;
        }
        else if ((c & 0xE0) == 0xC0)
        {
            code = ((str[i] & 0x1F) << 6) | (str[i + 1] & 0x3F);
            len = 2;
        }
        else if ((c & 0xF0) == 0xE0)
        {
            code = ((str[i] & 0x0F) << 12) | ((str[i + 1] & 0x3F) << 6) | (str[i + 2] & 0x3F);
            len = 3;
        }
        else if ((c & 0xF8) == 0xF0)
        {
            code = ((str[i] & 0x07) << 18) | ((str[i + 1] & 0x3F) << 12) | ((str[i + 2] & 0x3F) << 6) | (str[i + 3] & 0x3F);
            len = 4;
        }
        ucs4.push_back(code);
        i += len;
    }
}