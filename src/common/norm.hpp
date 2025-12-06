#ifndef NORM_HPP__
#define NORM_HPP__

namespace norm_image
{

enum class NormType : int
{
    None      = 0,
    MeanStd   = 1,
    AlphaBeta = 2
};

enum class ChannelType : int
{
    None   = 0,
    SwapRB = 1
};

struct Norm
{
    float mean[3];
    float std[3];
    float alpha, beta;
    NormType type            = NormType::None;
    ChannelType channel_type = ChannelType::None;

    // out = (x * alpha - mean) / std
    static Norm mean_std(const float mean[3],
                         const float std[3],
                         float alpha              = 1 / 255.0f,
                         ChannelType channel_type = ChannelType::None);

    // out = x * alpha + beta
    static Norm alpha_beta(float alpha, float beta = 0, ChannelType channel_type = ChannelType::None);

    // None
    static Norm None();
};

} // namespace norm_image

#endif