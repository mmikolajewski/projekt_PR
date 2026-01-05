#include "kernels.h"

// =========================
// KERNEL A: global + coalesced
// =========================
// Wariant A: czytamy TAB tylko z globalnej pamięci.
// Mapowanie wątków tak, żeby wątki w warpie często czytały sąsiednie adresy.
// k wyników na wątek realizujemy przez skok:
//   x = x_base + i * (gridDim.x * blockDim.x)
__global__ void kernel_A(const float* tab, float* out, int N, int R, int k) {
    int outSize = N - 2 * R;
    int x_base = blockIdx.x * blockDim.x + threadIdx.x;
    int y      = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < k; i++) {
        int x = x_base + i * (gridDim.x * blockDim.x);
        if (x < outSize && y < outSize) {
            float sum = 0.0f;
            int cy = y + R;
            int cx = x + R;
            for (int dy = -R; dy <= R; dy++) {
                for (int dx = -R; dx <= R; dx++) {
                    sum += tab[(cy + dy) * N + (cx + dx)];
                }
            }
            out[(size_t)y * outSize + x] = sum;
        }
    }
}