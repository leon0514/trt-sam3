#include <cuda_runtime.h>

class AutoDevice
{
public:
    explicit AutoDevice(int device_id)
    {
        cudaGetDevice(&prev_device_);
        if (prev_device_ != device_id)
        {
            cudaSetDevice(device_id);
            switched_ = true;
        }
    }

    ~AutoDevice()
    {
        if (switched_)
        {
            cudaSetDevice(prev_device_);
        }
    }

    // 禁止拷贝和赋值
    AutoDevice(const AutoDevice &) = delete;
    AutoDevice &operator=(const AutoDevice &) = delete;

private:
    int prev_device_ = 0;
    bool switched_ = false;
};