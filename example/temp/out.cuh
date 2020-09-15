#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <assert.h>
#include "gwaihir.h"
#include "native.cuh"


__global__ inline void kernel1(int* a_, unsigned int a_d0, int* result_, unsigned int result_d0, const unsigned int kernel1d0, const int kernel1o0) {
    const int i_ = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x) + kernel1o0;
    if (threadIdx.x + blockIdx.x * blockDim.x < kernel1d0) {
        result_[i_] = a_[i_];
    };
}

__global__ inline void kernel2(int* a_, unsigned int a_d0, const unsigned int kernel2d0, const int kernel2o0) {
    const int i_ = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x) + kernel2o0;
    if (threadIdx.x + blockIdx.x * blockDim.x < kernel2d0) {
        a_[i_] = i_;
    };
}

__global__ inline void kernel3(int* a_, unsigned int a_d0, int* b_, unsigned int b_d0, const unsigned int kernel3d0, const int kernel3o0) {
    const int i_ = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x) + kernel3o0;
    if (threadIdx.x + blockIdx.x * blockDim.x < kernel3d0) {
        a_[i_] = a_[i_] + b_[i_];
    };
}

__host__ inline void copy_(int* a_, unsigned int a_d0, DevPtr<int>* result, unsigned int* result0) {
    DevPtr<int> result_zero_vec_;
    unsigned int result_zero_vec_d0;
    zero_vec_(len_(a_, a_d0), &result_zero_vec_, &result_zero_vec_d0);
    DevPtr<int> result_;
    unsigned int result_d0;
    {
        result_ = std::move(result_zero_vec_);
        result_d0 = result_zero_vec_d0;
    };
    {
        const unsigned int array0shape0 = a_d0;
        const unsigned int array1shape0 = result_d0;
        const int kernel1o0 = round(min(array0shape0, 0, array1shape0, 0));
        const unsigned int kernel1d0 = round(max(array0shape0, 0, array1shape0, 0)) - kernel1o0;
        kernel1 <<< dim3((kernel1d0 - 1) / 256 + 1), dim3(256), 0 >>> (a_, a_d0, result_, result_d0, kernel1d0, kernel1o0);
    };
    {
        *result = std::move(result_);
        *result0 = result_d0;
        return;
    };
}

__host__ inline void vec_add_(int* a_, unsigned int a_d0, int* b_, unsigned int b_d0) {
    {
        const unsigned int array0shape0 = a_d0;
        const unsigned int array1shape0 = b_d0;
        const int kernel3o0 = round(min(array0shape0, 0, array1shape0, 0));
        const unsigned int kernel3d0 = round(max(array0shape0, 0, array1shape0, 0)) - kernel3o0;
        kernel3 <<< dim3((kernel3d0 - 1) / 256 + 1), dim3(256), 0 >>> (a_, a_d0, b_, b_d0, kernel3d0, kernel3o0);
    };
}

__host__ inline void main_() {
    DevPtr<int> result_zero_vec_;
    unsigned int result_zero_vec_d0;
    zero_vec_(10, &result_zero_vec_, &result_zero_vec_d0);
    DevPtr<int> a_;
    unsigned int a_d0;
    {
        a_ = std::move(result_zero_vec_);
        a_d0 = result_zero_vec_d0;
    };
    {
        const unsigned int array0shape0 = a_d0;
        const int kernel2o0 = round(min(array0shape0, 0));
        const unsigned int kernel2d0 = round(max(array0shape0, 0)) - kernel2o0;
        kernel2 <<< dim3((kernel2d0 - 1) / 256 + 1), dim3(256), 0 >>> (a_, a_d0, kernel2d0, kernel2o0);
    };
    DevPtr<int> result_copy_;
    unsigned int result_copy_d0;
    copy_(a_, a_d0, &result_copy_, &result_copy_d0);
    vec_add_(a_, a_d0, result_copy_, result_copy_d0);
    print_vec_(a_, a_d0);
}