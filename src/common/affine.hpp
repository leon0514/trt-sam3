#ifndef AFFINE_HPP__
#define AFFINE_HPP__
#include <algorithm>
#include <tuple>

namespace affine
{

    struct CropResizeMatrix
    {
        float i2d[6]; // image to dst(network), 2x3 matrix
        float d2i[6]; // dst to image, 2x3 matrix

        void compute(const std::tuple<int, int> &from,
                     const std::tuple<int, int> &to,
                     const std::tuple<int, int> &start)
        {
            float scale_x = std::get<0>(to) / (float)std::get<0>(from);
            float scale_y = std::get<1>(to) / (float)std::get<1>(from);
            int start_x = std::get<0>(start);
            int start_y = std::get<1>(start);

            // crop resize
            i2d[0] = scale_x;
            i2d[1] = 0.0f;
            i2d[2] = -scale_x * start_x;
            i2d[3] = 0.0f;
            i2d[4] = scale_y;
            i2d[5] = -scale_y * start_y;

            double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
            D = D != 0. ? double(1.) / D : double(0.);
            double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D,
                   A21 = -i2d[3] * D;
            double b1 = -A11 * i2d[2] - A12 * i2d[5];
            double b2 = -A21 * i2d[2] - A22 * i2d[5];

            d2i[0] = A11;
            d2i[1] = A12;
            d2i[2] = b1;
            d2i[3] = A21;
            d2i[4] = A22;
            d2i[5] = b2;
        }
    };

    struct ResizeMatrix
    {
        float i2d[6]; // image to dst(network), 2x3 matrix
        float d2i[6]; // dst to image, 2x3 matrix

        void compute(const std::tuple<int, int> &from,
                     const std::tuple<int, int> &to)
        {
            float scale_x = std::get<0>(to) / (float)std::get<0>(from);
            float scale_y = std::get<1>(to) / (float)std::get<1>(from);
            float scale = std::min(scale_x, scale_y);

            // resize
            // i2d[0] = scale_x;
            // i2d[1] = 0;
            // i2d[2] = 0;
            // i2d[3] = 0;
            // i2d[4] = scale_y;
            // i2d[5] = 0;

            // double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
            // D = D != 0. ? double(1.) / D : double(0.);
            // double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D,
            //        A21 = -i2d[3] * D;
            // double b1 = -A11 * i2d[2] - A12 * i2d[5];
            // double b2 = -A21 * i2d[2] - A22 * i2d[5];

            // d2i[0] = A11;
            // d2i[1] = A12;
            // d2i[2] = b1;
            // d2i[3] = A21;
            // d2i[4] = A22;
            // d2i[5] = b2;

            float inv_scale_x = 1.0f / scale_x;
            float inv_scale_y = 1.0f / scale_y;

            d2i[0] = inv_scale_x;
            d2i[1] = 0;
            d2i[3] = 0;
            d2i[4] = inv_scale_y;

            // --- 核心修改 Start ---
            // 增加 0.5 的偏移补偿，以匹配 PIL/OpenCV 的中心对齐逻辑
            // Formula: translation = 0.5 * scale - 0.5
            d2i[2] = 0.5f * inv_scale_x - 0.5f;
            d2i[5] = 0.5f * inv_scale_y - 0.5f;
        }
    };

    struct LetterBoxMatrix
    {
        float i2d[6]; // image to dst(network), 2x3 matrix
        float d2i[6]; // dst to image, 2x3 matrix

        void compute(const std::tuple<int, int> &from,
                     const std::tuple<int, int> &to)
        {
            float scale_x = std::get<0>(to) / (float)std::get<0>(from);
            float scale_y = std::get<1>(to) / (float)std::get<1>(from);
            float scale = std::min(scale_x, scale_y);

            // letter box
            i2d[0] = scale;
            i2d[1] = 0;
            i2d[2] = -scale * std::get<0>(from) * 0.5 + std::get<0>(to) * 0.5 +
                     scale * 0.5 - 0.5;
            i2d[3] = 0;
            i2d[4] = scale;
            i2d[5] = -scale * std::get<1>(from) * 0.5 + std::get<1>(to) * 0.5 +
                     scale * 0.5 - 0.5;

            double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
            D = D != 0. ? double(1.) / D : double(0.);
            double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D,
                   A21 = -i2d[3] * D;
            double b1 = -A11 * i2d[2] - A12 * i2d[5];
            double b2 = -A21 * i2d[2] - A22 * i2d[5];

            d2i[0] = A11;
            d2i[1] = A12;
            d2i[2] = b1;
            d2i[3] = A21;
            d2i[4] = A22;
            d2i[5] = b2;
        }
    };

} // end namespace affine

#endif