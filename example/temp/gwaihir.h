#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <assert.h>

#ifdef __CUDACC__
#define DEVICE_HOST_FUNCTION __device__ __host__
#else
#define DEVICE_HOST_FUNCTION
#endif

inline void validateCudaStatusOk(cudaError_t status, const char* file, int line) {
    if (status != cudaSuccess) {
        std::cerr << "Cuda Error: " << status << ": " << cudaGetErrorString(status) << " at " << file << ":" << line << std::endl;
        throw "cuda error";
    }
}

#define checkCudaStatus(status) validateCudaStatusOk(status, __FILE__, __LINE__)

template<typename T>
struct DevPtr {
    T* content;

    DEVICE_HOST_FUNCTION inline DevPtr() : content(nullptr) {}

    DEVICE_HOST_FUNCTION inline DevPtr(DevPtr<T>&& rhs) : content(rhs.content) {
        rhs.content = nullptr;
    }

    DEVICE_HOST_FUNCTION inline DevPtr& operator=(DevPtr<T>&& rhs) {
        content = rhs.content;
        rhs.content = nullptr;
        return *this;
    }

    DEVICE_HOST_FUNCTION inline ~DevPtr() {
#ifdef __CUDA_ARCH__ 
        free(content);
#else
        checkCudaStatus(cudaFree(content));
#endif
    }

    DEVICE_HOST_FUNCTION inline operator T* () {
        return content;
    }

    DEVICE_HOST_FUNCTION T& operator[] (size_t index) {
        return content[index];
    }
};

template <typename T>
DEVICE_HOST_FUNCTION inline float min(T t)
{
    return t;
}

template <typename T, typename... Args>
DEVICE_HOST_FUNCTION inline float min(T head, Args... tail)
{
    float tail_min = min(tail...);
    if (tail_min < head) {
        return tail_min;
    }
    else {
        return head;
    }
}

template <typename T>
DEVICE_HOST_FUNCTION inline float max(T t)
{
    return t;
}

template <typename T, typename... Args>
DEVICE_HOST_FUNCTION inline float max(T head, Args... tail)
{
    float tail_max = max(tail...);
    if (tail_max < head) {
        return head;
    }
    else {
        return tail_max;
    }
}
