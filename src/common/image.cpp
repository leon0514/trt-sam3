#include "common/image.hpp"

namespace tensor
{

tensor::Image cvimg(const cv::Mat &image) { return Image(image.data, image.cols, image.rows); }

} // namespace tensor