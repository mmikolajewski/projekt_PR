#include "kernels.h"

// =========================
// KERNEL B: global + non-coalesced
// =========================
// Wariant B: celowo psujemy lokalność (zamieniamy role tx/ty),
// przez co wątki mogą czytać "porozrzucane" adresy.

__global__ void kernel_B(const float* tab, float* out, int N, int R, int k) {
    int outSize = N - 2 * R;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // UWAGA: tu psujemy coalescing: warp idzie po tx, więc robimy żeby y zależało od tx
    int y = blockIdx.y * blockDim.y + tx;          // <--- to jest klucz
    int x0 = blockIdx.x * blockDim.x + ty;         // <--- i to

    for (int i = 0; i < k; i++) {
        int x = x0 + i * (gridDim.x * blockDim.x);
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