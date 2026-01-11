// Kompilacja:
//   make
//
// Uruchomienie:
//   ./paramk --N 4096 --R 8 --BS 16

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <helper_image.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include "kernels.h"

#define EPS_VERIFY 1e-5f

struct ResultRow
{
    double msA = 0, msB = 0, msC = 0, msD = 0;
    double gfA = 0, gfB = 0, gfC = 0, gfD = 0;
    bool okA = true, okB = true, okC = true, okD = true;
    bool skipC = false, skipD = false;
};


static inline int iDivUp(int a, int b) { return (a + b - 1) / b; }

void cpu_radius_sum(const float *tab, float *out, int N, int R)
{
    int outSize = N - 2 * R;
    for (int y = 0; y < outSize; y++)
    {
        for (int x = 0; x < outSize; x++)
        {
            float sum = 0.0f;
            int cy = y + R;
            int cx = x + R;
            for (int dy = -R; dy <= R; dy++)
            {
                for (int dx = -R; dx <= R; dx++)
                {
                    sum += tab[(cy + dy) * N + (cx + dx)];
                }
            }
            out[y * outSize + x] = sum;
        }
    }
}

// funkcja skopiowana z przykładów CUDA SDK Nvidia
// Allocates a matrix with random float entries.
void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = rand() / static_cast<float>(RAND_MAX);
    }
}

ResultRow run_one_case(int N, int R, int BS, int k, int nIter, cudaDeviceProp prop)
{
    ResultRow res{};
    int outSize = N - 2 * R;
    if (outSize <= 0)
    {
        res.okA = res.okB = res.okC = res.okD = false;
        res.skipC = res.skipD = true;
        return res;
    }

    size_t sizeIn = (size_t)N * (size_t)N * sizeof(float);
    size_t sizeOut = (size_t)outSize * (size_t)outSize * sizeof(float);

    float *h_in = (float *)malloc(sizeIn);
    float *h_out = (float *)malloc(sizeOut);

    // Wypełnienie danych wejściowych (losowo)
    randomInit(h_in, N * N);

    std::vector<float> ref((size_t)outSize * (size_t)outSize);
    cpu_radius_sum(h_in, ref.data(), N, R);

    float *d_in = nullptr, *d_out = nullptr;
    checkCudaErrors(cudaMalloc(&d_in, sizeIn));

    checkCudaErrors(cudaMalloc(&d_out, sizeOut));
    checkCudaErrors(cudaMemcpy(d_in, h_in, sizeIn, cudaMemcpyHostToDevice));

    dim3 block(BS, BS);
    int grid_w = iDivUp(outSize, BS * k);
    int grid_h = iDivUp(outSize, BS);
    dim3 grid(grid_w, grid_h);

    double opsPerPixel = (double)(2 * R + 1) * (double)(2 * R + 1);
    double totalOps = (double)outSize * (double)outSize * opsPerPixel;

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    auto measure = [&](auto kernel, double &msOut, double &gfOut, bool &okOut, size_t shBytes, bool &skipFlag)
    {
        if (shBytes > 0 && shBytes > (size_t)prop.sharedMemPerBlock)
        {
            skipFlag = true;
            msOut = 0.0;
            gfOut = 0.0;
            okOut = true; // skip to nie fail
            return;
        }

        // warmup
        // https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html
        // pierwsze uruchomienie kernela może być wolniejsze z powodu "lazy loading"

        kernel<<<grid, block, shBytes>>>(d_in, d_out, N, R, k);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // pomiar czasu za pomoscą funkcji z cuda_pomoc
        StopWatchInterface *timer = NULL;
        sdkCreateTimer(&timer);

        checkCudaErrors(cudaEventRecord(start));
        for (int i = 0; i < nIter; i++)
        {
            kernel<<<grid, block, shBytes>>>(d_in, d_out, N, R, k);
        }
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));

        float ms = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));
        msOut = (double)ms / (double)nIter;

        double sec = msOut / 1000.0;
        gfOut = (totalOps * 1e-9) / sec;

        checkCudaErrors(cudaMemcpy(h_out, d_out, sizeOut, cudaMemcpyDeviceToHost));
        okOut = compareData(ref.data(), h_out, outSize, EPS_VERIFY, 0.0f);
    };

    // A, B (Global)
    bool dummySkip = false;
    measure(kernel_A, res.msA, res.gfA, res.okA, 0, dummySkip);
    measure(kernel_B, res.msB, res.gfB, res.okB, 0, dummySkip);

    // C, D (Shared)
    size_t shBytes = (size_t)(BS + 2 * R) * (size_t)(BS + 2 * R) * sizeof(float);
    measure(kernel_C, res.msC, res.gfC, res.okC, shBytes, res.skipC);
    measure(kernel_D, res.msD, res.gfD, res.okD, shBytes, res.skipD);

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return res;
}

void run_auto_mode(cudaDeviceProp prop, int N, int R, int BS)
{
    int ks[3] = {1, 2, 8};

    printf("\n#################### paramk ####################\n");

    // --- LIMIT żeby nie zabić sesji na serwerze ---
    const int MAX_KTEST_N = 8192;
    if (N > MAX_KTEST_N)
        N = MAX_KTEST_N;

    if (R >= 64 && N > 4096)
        N = 4096;

    if (N <= 2 * R)
        N = 4 * R + 2;

    printf("\n==================== K TEST ====================\n");
    printf("| BS=%d  R=%d  N=%d \n", BS, R, N);

    printf("k list: 1 2 8\n");
    printf("------------------------------------------------\n");

    printf("k      |    ms_A   GF_A |    ms_B   GF_B |    ms_C   GF_C |    ms_D   GF_D |   CHECK\n");
    printf("----------------------------------------------------------------------------------------------\n");

    for (int ki = 0; ki < 3; ki++)
    {
        int k = ks[ki];
        ResultRow r = run_one_case(N, R, BS, k, 1, prop);

        printf("%-6d| %6.3f %6.1f | %6.3f %6.1f | ",
                k, r.msA, r.gfA, r.msB, r.gfB);

        if (r.skipC)
            printf(" [SKIP] [SKIP] | ");
        else
            printf("%6.3f %6.1f | ", r.msC, r.gfC);

        if (r.skipD)
            printf(" [SKIP] [SKIP] | ");
        else
            printf("%6.3f %6.1f | ", r.msD, r.gfD);

        char chk[64];
        snprintf(chk, sizeof(chk), "A:%s B:%s C:%s D:%s",
                    r.okA ? "OK" : "ERR",
                    r.okB ? "OK" : "ERR",
                    r.skipC ? "SKIP" : (r.okC ? "OK" : "ERR"),
                    r.skipD ? "SKIP" : (r.okD ? "OK" : "ERR"));
        printf("%s\n", chk);
    }
}

int main(int argc, const char **argv)
{
    // dekodowanie argumentów wejściowych podanych przy uruchomieniu
    int N = 512;
    int R = 8;
    int BS = 16;

    if (checkCmdLineFlag(argc, argv, "N"))
        N = getCmdLineArgumentInt(argc, argv, "N=");

    if (checkCmdLineFlag(argc, argv, "R"))
        R = getCmdLineArgumentInt(argc, argv, "R=");

    if (checkCmdLineFlag(argc, argv, "BS"))
        BS = getCmdLineArgumentInt(argc, argv, "BS=");

    // Ustwawienie losowości
    srand((unsigned int)time(NULL));

    // Wybór karty graficznej
    int dev = 0;
    cudaSetDevice(dev);

    // Pobranie danych o karcie graficznej
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);

    run_auto_mode(prop, N, R, BS);
    return 0;
}
