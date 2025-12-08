#include "common/check.hpp"
#include "common/memory.hpp" // 假设文件名是 memory.hpp
#include <cuda_runtime.h>

namespace tensor
{

using namespace std;

static size_t upbound(size_t n, size_t align) { return (n + align - 1) / align * align; }

BaseMemory::BaseMemory(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes)
{
    reference(cpu, cpu_bytes, gpu, gpu_bytes);
}

BaseMemory::~BaseMemory() { release(); }

// [新增] 实现内存共享逻辑
void BaseMemory::set_shared_memory(const BaseMemory& other)
{
    // 1. 拷贝智能指针（引用计数 +1）
    this->cpu_ptr_ = other.cpu_ptr_;
    this->gpu_ptr_ = other.gpu_ptr_;

    // 2. 同步裸指针和元数据
    this->cpu_ = other.cpu_;
    this->cpu_bytes_ = other.cpu_bytes_;
    this->cpu_capacity_ = other.cpu_capacity_;

    this->gpu_ = other.gpu_;
    this->gpu_bytes_ = other.gpu_bytes_;
    this->gpu_capacity_ = other.gpu_capacity_;
}

void BaseMemory::reference(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes)
{
    release();

    // 定义空删除器，因为 reference 表示引用外部内存，不负责释放
    auto no_op_deleter = [](void*){};

    if (cpu && cpu_bytes > 0)
    {
        this->cpu_ptr_ = std::shared_ptr<void>(cpu, no_op_deleter);
        this->cpu_ = cpu;
        this->cpu_bytes_ = cpu_bytes;
        this->cpu_capacity_ = cpu_bytes;
    }
    else
    {
        this->cpu_ptr_.reset();
        this->cpu_ = nullptr;
        this->cpu_bytes_ = 0;
        this->cpu_capacity_ = 0;
    }

    if (gpu && gpu_bytes > 0)
    {
        this->gpu_ptr_ = std::shared_ptr<void>(gpu, no_op_deleter);
        this->gpu_ = gpu;
        this->gpu_bytes_ = gpu_bytes;
        this->gpu_capacity_ = gpu_bytes;
    }
    else
    {
        this->gpu_ptr_.reset();
        this->gpu_ = nullptr;
        this->gpu_bytes_ = 0;
        this->gpu_capacity_ = 0;
    }
}

void *BaseMemory::gpu_realloc(size_t bytes)
{
    size_t size = upbound(bytes, 32);
    
    // 如果容量不够，重新分配
    if (gpu_capacity_ < size)
    {
        // 释放旧的
        release_gpu();

        void* ptr = nullptr;
        checkRuntime(cudaMalloc(&ptr, size));
        
        // 创建智能指针，绑定 cudaFree 作为删除器
        gpu_ptr_ = std::shared_ptr<void>(ptr, [](void* p){
            checkRuntime(cudaFree(p));
        });

        gpu_ = ptr;
        gpu_capacity_ = size;
    }
    gpu_bytes_ = bytes;
    return gpu_;
}

void *BaseMemory::cpu_realloc(size_t bytes)
{
    size_t size = upbound(bytes, 32);

    if (cpu_capacity_ < size)
    {
        release_cpu();

        void* ptr = nullptr;
        checkRuntime(cudaMallocHost(&ptr, size));
        Assert(ptr != nullptr);
        
        // 创建智能指针，绑定 cudaFreeHost 作为删除器
        cpu_ptr_ = std::shared_ptr<void>(ptr, [](void* p){
            checkRuntime(cudaFreeHost(p));
        });

        cpu_ = ptr;
        cpu_capacity_ = size;
    }
    cpu_bytes_ = bytes;
    return cpu_;
}

void BaseMemory::release_cpu()
{
    // shared_ptr reset 会减少引用计数，归零时自动调用 deleter
    cpu_ptr_.reset();
    cpu_ = nullptr;
    cpu_capacity_ = 0;
    cpu_bytes_    = 0;
}

void BaseMemory::release_gpu()
{
    gpu_ptr_.reset();
    gpu_ = nullptr;
    gpu_capacity_ = 0;
    gpu_bytes_    = 0;
}

void BaseMemory::release()
{
    release_cpu();
    release_gpu();
}

} // namespace tensor