#pragma once
#include <cuda_runtime.h>

__global__ void kernel_A(const float* tab, float* out, int N, int R, int k);
__global__ void kernel_B(const float* tab, float* out, int N, int R, int k);
__global__ void kernel_C(const float* tab, float* out, int N, int R, int k);
__global__ void kernel_D(const float* tab, float* out, int N, int R, int k);

