#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <assert.h>

inline void validateCudaStatusOk(cudaError_t status, const char* file, int line) {
    if (status != cudaSuccess) {
        std::cerr << "Cuda Error: " << status << ": " << cudaGetErrorString(status) << " at " << file << ":" << line << std::endl;
        throw "cuda error";
    }
}

#define checkCudaStatus(status) validateCudaStatusOk(status, __FILE__, __LINE__)

template <typename T>
__device__ __host__ inline T min(T t) 
{
    return t;
}

template <typename T, typename... Args>
__device__ __host__ inline T min(T head, Args... tail) 
{
    T tail_min = min(tail...);
    if (tail_min < head) {
        return tail_min;
    } else {
        return head;
    }
}

template <typename T>
__device__ __host__ inline T max(T t) 
{
    return t;
}

template <typename T, typename... Args>
__device__ __host__ inline T max(T head, Args... tail) 
{
    T tail_max = max(tail...);
    if (tail_max < head) {
        return head;
    } else {
        return tail_max;
    }
}

__device__ __host__ inline unsigned int len_(int* ptr, int len) {
    return len;
}

__host__ inline void allocate1d_(int len, int** out, unsigned int* out_d0) {
    *out_d0 = len;
    checkCudaStatus(cudaMalloc(out, len * sizeof(int)));
}