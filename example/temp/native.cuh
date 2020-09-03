#include "gwaihir.h"
#include <memory>

void print_vec_(int* a_, const unsigned int a_d0) {
    std::unique_ptr<int[]> data(new int[a_d0]);
    checkCudaStatus(cudaMemcpy(data.get(), a_, a_d0 * sizeof(int), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < a_d0; ++i) {
        std::cout << data[i] << ", ";
    }
    std::cout << std::endl;
}

void zero_vec_(int len, DevPtr<int>* result, unsigned int* result_len) {
    *result_len = len;
    checkCudaStatus(cudaMalloc(&result->content, len * sizeof(int)));
}

int len_(int* a_, const unsigned int a_d0) {
    return a_d0;
}