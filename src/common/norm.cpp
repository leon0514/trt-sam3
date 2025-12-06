#include "common/norm.hpp"
#include <cstring>
#include <memory>

namespace norm_image
{

Norm Norm::mean_std(const float mean[3], const float std[3], float alpha, ChannelType channel_type)
{
    Norm out;
    out.type         = NormType::MeanStd;
    out.alpha        = alpha;
    out.channel_type = channel_type;
    memcpy(out.mean, mean, sizeof(out.mean));
    memcpy(out.std, std, sizeof(out.std));
    return out;
}

Norm Norm::alpha_beta(float alpha, float beta, ChannelType channel_type)
{
    Norm out;
    out.type         = NormType::AlphaBeta;
    out.alpha        = alpha;
    out.beta         = beta;
    out.channel_type = channel_type;
    return out;
}

Norm Norm::None() { return Norm(); }

} // namespace norm