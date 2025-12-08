#ifndef __MEMORY_HPP__
#define __MEMORY_HPP__

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>
#include <functional>

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

    // 状态查询
    inline size_t cpu_bytes() const { return cpu_bytes_; }
    inline size_t gpu_bytes() const { return gpu_bytes_; }
    
    // 获取裸指针（用于计算）
    virtual inline void *get_gpu() const { return gpu_; }
    virtual inline void *get_cpu() const { return cpu_; }

    // 引用外部内存（不拥有所有权）
    void reference(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes);

    // [新增]：共享另一个 Memory 对象的内存（拥有共同所有权）
    void set_shared_memory(const BaseMemory& other);

  protected:
    // 裸指针保留用于快速访问，但生命周期由 shared_ptr 管理
    void *cpu_           = nullptr;
    size_t cpu_bytes_    = 0;
    size_t cpu_capacity_ = 0;

    void *gpu_           = nullptr;
    size_t gpu_bytes_    = 0;
    size_t gpu_capacity_ = 0;

    // 智能指针管理生命周期
    std::shared_ptr<void> cpu_ptr_ = nullptr;
    std::shared_ptr<void> gpu_ptr_ = nullptr;
};

template <typename _DT> class Memory : public BaseMemory
{
  public:
    Memory()                               = default;
    Memory(const Memory &other) { this->set_shared_memory(other); }
    Memory &operator=(const Memory &other) { 
        if(this != &other) this->set_shared_memory(other); 
        return *this; 
    }

    virtual _DT *gpu(size_t size) { return (_DT *)BaseMemory::gpu_realloc(size * sizeof(_DT)); }
    virtual _DT *cpu(size_t size) { return (_DT *)BaseMemory::cpu_realloc(size * sizeof(_DT)); }

    inline size_t cpu_size() const { return cpu_bytes_ / sizeof(_DT); }
    inline size_t gpu_size() const { return gpu_bytes_ / sizeof(_DT); }

    virtual inline _DT *gpu() const { return (_DT *)gpu_; }
    virtual inline _DT *cpu() const { return (_DT *)cpu_; }
};

} // namespace tensor

#endif