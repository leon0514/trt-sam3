#ifndef __MEMORY_HPP__
#define __MEMORY_HPP__

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace tensor
{

class BaseMemory
{
  public:
    BaseMemory() = default;
    BaseMemory(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes);
    virtual ~BaseMemory();
    virtual void *gpu_realloc(size_t bytes);
    virtual void *cpu_realloc(size_t bytes);
    void release_gpu();
    void release_cpu();
    void release();
    inline bool owner_gpu() const { return owner_gpu_; }
    inline bool owner_cpu() const { return owner_cpu_; }
    inline size_t cpu_bytes() const { return cpu_bytes_; }
    inline size_t gpu_bytes() const { return gpu_bytes_; }
    virtual inline void *get_gpu() const { return gpu_; }
    virtual inline void *get_cpu() const { return cpu_; }
    void reference(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes);

  protected:
    void *cpu_           = nullptr;
    size_t cpu_bytes_    = 0;
    size_t cpu_capacity_ = 0;
    bool owner_cpu_      = true;

    void *gpu_           = nullptr;
    size_t gpu_bytes_    = 0;
    size_t gpu_capacity_ = 0;
    bool owner_gpu_      = true;
};

template <typename _DT> class Memory : public BaseMemory
{
  public:
    Memory()                               = default;
    Memory(const std::vector<int> shape)
    {
        size_t size = 1;
        for (const auto& s : shape)
        {
            size *= s;
        }
        gpu(size);
        cpu(size);
    }
    Memory(const Memory &other)            = delete;
    Memory &operator=(const Memory &other) = delete;
    virtual _DT *gpu(size_t size) { return (_DT *)BaseMemory::gpu_realloc(size * sizeof(_DT)); }
    virtual _DT *cpu(size_t size) { return (_DT *)BaseMemory::cpu_realloc(size * sizeof(_DT)); }

    inline size_t cpu_size() const { return cpu_bytes_ / sizeof(_DT); }
    inline size_t gpu_size() const { return gpu_bytes_ / sizeof(_DT); }

    virtual inline _DT *gpu() const { return (_DT *)gpu_; }
    virtual inline _DT *cpu() const { return (_DT *)cpu_; }
};

} // namespace tensor

#endif