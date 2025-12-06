#ifndef __IMAGE_HPP__
#define __IMAGE_HPP__
#include "opencv2/opencv.hpp"

namespace tensor
{

struct Image
{
    const void *bgrptr = nullptr;
    int width = 0, height = 0;

    Image() = default;
    Image(const void *bgrptr, int width, int height) : bgrptr(bgrptr), width(width), height(height) {}
};

Image cvimg(const cv::Mat &image);

} // namespace tensor

#endif // __IMAGE_HPP__