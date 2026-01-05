#include "kernels.h"

// =========================
// KERNEL D: shared + bank-conflicts (celowo gorzej)
// =========================
// Wariant D: ładowanie tile identyczne jak w C,
// ale przy obliczeniach celowo "przekręcamy" mapowanie (tx steruje Y, ty steruje X),
// co często prowadzi do konfliktów banków w shared (gorsza wydajność).
__global__ void kernel_D(const float* tab, float* out, int N, int R, int k) {
    extern __shared__ float s[];

    int outSize = N - 2 * R;
    int bs = blockDim.x;
    int tileW = bs + 2 * R;
    int tileH = bs + 2 * R;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nThreads = blockDim.x * blockDim.y;
    int tileSize = tileW * tileH;

    int blockOutY = blockIdx.y * bs;
    int blockOutXBase = blockIdx.x * bs;

    for (int i = 0; i < k; i++) {
        int blockOutX = blockOutXBase + i * (gridDim.x * bs);

        for (int idx = tid; idx < tileSize; idx += nThreads) {
            int sy = idx / tileW;
            int sx = idx - sy * tileW;
            int inY = blockOutY + sy;
            int inX = blockOutX + sx;
            if (inY < N && inX < N) s[(size_t)sy * tileW + sx] = tab[(size_t)inY * N + inX];
            else                    s[(size_t)sy * tileW + sx] = 0.0f;
        }
        __syncthreads();

        int y = blockOutY + threadIdx.x; // swap
        int x = blockOutX + threadIdx.y; // swap

        if (y < outSize && x < outSize) {
            float sum = 0.0f;
            int csy = threadIdx.x + R;
            int csx = threadIdx.y + R;
            for (int dy = -R; dy <= R; dy++) {
                for (int dx = -R; dx <= R; dx++) {
                    sum += s[(size_t)(csy + dy) * tileW + (csx + dx)];
                }
            }
            out[(size_t)y * outSize + x] = sum;
        }
        __syncthreads();
    }
}
