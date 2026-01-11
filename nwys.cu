// Kompilacja:
//   make
//
// Uruchomienie:
//   ./nwys

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

void print_table_header()
{
    printf("N      |  ms_A   GF_A |  ms_B   GF_B |  ms_C   GF_C |  ms_D   GF_D | CHECK\n");
    printf("----------------------------------------------------------------------------------------------\n");
}

void print_table_row(int N, const ResultRow &r)
{
    char chk[64];
    snprintf(chk, sizeof(chk), "A:%s B:%s C:%s D:%s",
             r.okA ? "OK" : "ERR",
             r.okB ? "OK" : "ERR",
             r.skipC ? "SKIP" : (r.okC ? "OK" : "ERR"),
             r.skipD ? "SKIP" : (r.okD ? "OK" : "ERR"));

    printf("%6d | %6.3f %6.1f | %6.3f %6.1f | ",
           N, r.msA, r.gfA, r.msB, r.gfB);

    if (r.skipC)
        printf(" [SKIP] [SKIP] | ");
    else
        printf("%6.3f %6.1f | ", r.msC, r.gfC);

    if (r.skipD)
        printf(" [SKIP] [SKIP] | ");
    else
        printf("%6.3f %6.1f | ", r.msD, r.gfD);

    printf("%s\n", chk);
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

int find_Nwys_from_rows(const std::vector<int> &Nlist, const std::vector<ResultRow> &rows, double eps)
{
    auto pickGF = [&](const ResultRow &r) -> double
    {
        if (!r.skipC)
            return r.gfC;
        return r.gfA;
    };

    for (int i = 0; i + 2 < (int)Nlist.size(); i++)
    {
        double g0 = pickGF(rows[i]);
        double g1 = pickGF(rows[i + 1]);
        double g2 = pickGF(rows[i + 2]);

        if (g0 <= 0 || g1 <= 0 || g2 <= 0)
            continue;

        double rel1 = (g1 - g0) / g0;
        double rel2 = (g2 - g1) / g1;

        if (rel1 < eps && rel2 < eps)
        {
            return Nlist[i];
        }
    }
    return Nlist.back();
}

void run_auto_mode(cudaDeviceProp prop)
{
    // std::vector<int> Nlist = {512, 1024, 2048, 4096, 8192, 16384}; za duzo wolno przy 16384
    // std::vector<int> Nlist = {1024, 2048, 4096, 8192};

    std::vector<int> Nlist = {128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048};
    int BSlist[3] = {8, 16, 32};

    double eps = 0.05;
    const int MAXN = 16;

    double stage1_ms[2][3][MAXN] = {}; // czas [ms] (wybrane: C)
    double stage1_gf[2][3][MAXN] = {}; // prędkość [GF/s] (wybrane: C)
    bool stage1_ok[2][3][MAXN] = {};   // poprawność

    int usedCount[2][3] = {};
    int usedNvals[2][3][MAXN] = {};

    printf("\n##############################################################################################\n");
    printf("# Poszukiwanie N_wys                                                                       ###\n");
    printf("##############################################################################################\n");
    printf("# Plan: BS in {8,16,32}, R1=BS/2 (R<BS), R2=2*BS (R>BS)                                    ###\n");
    printf("# N list: ");
    for (int v : Nlist)
        printf("%d ", v);
    printf("        ###\n");
    printf("##############################################################################################\n");
    printf("\n");

    for (int bi = 0; bi < 3; bi++)
    {
        int BS = BSlist[bi];
        int R1 = BS / 2;
        int R2 = 2 * BS;

        // summary[bi].R1 = R1;
        // summary[bi].R2 = R2;

        for (int rr = 0; rr < 2; rr++)
        {
            int R = (rr == 0) ? R1 : R2;

            printf("----------------------------------------------------------------------------------------------\n");
            printf("| PROBA dla BS=%d  R=%d   (%s)\n", BS, R, (R < BS ? "R < BS" : "R > BS"));
            printf("----------------------------------------------------------------------------------------------\n");

            print_table_header();

            std::vector<ResultRow> rows;
            std::vector<int> usedN;

            for (int Ni = 0; Ni < (int)Nlist.size(); Ni++)
            {
                int N = Nlist[Ni];
                ResultRow r = run_one_case(N, R, BS, 1, 3, prop);

                print_table_row(N, r);
                int rrIdx = rr;
                int biIdx = bi;

                int slot = usedCount[rrIdx][biIdx];
                if (slot < MAXN)
                {
                    usedNvals[rrIdx][biIdx][slot] = N;

                    double msPick = r.skipC ? r.msA : r.msC;
                    double gfPick = r.skipC ? r.gfA : r.gfC;
                    bool okPick = r.skipC ? r.okA : r.okC;

                    stage1_ms[rrIdx][biIdx][slot] = msPick;
                    stage1_gf[rrIdx][biIdx][slot] = gfPick;
                    stage1_ok[rrIdx][biIdx][slot] = okPick;

                    usedCount[rrIdx][biIdx] = slot + 1;
                }
                rows.push_back(r);
                usedN.push_back(N);
            }

            int Nwys = usedN.empty() ? -1 : find_Nwys_from_rows(usedN, rows, eps);

            printf("----------------------------------------------------------------------------------------------\n");
            printf("N_wys (estimated) = %d\n", Nwys);
            printf("\n");
        }
    }
    auto print_big_table_ms = [&](int rrIdx, const char *title)
    {
        printf("\n==================== %s (TIME ms) ====================\n", title);
        printf("-----------------------------------------------------------------------------------------------------------\n");
        int Rb0 = (rrIdx == 0) ? (BSlist[0] / 2) : (2 * BSlist[0]);
        int Rb1 = (rrIdx == 0) ? (BSlist[1] / 2) : (2 * BSlist[1]);
        int Rb2 = (rrIdx == 0) ? (BSlist[2] / 2) : (2 * BSlist[2]);

        printf("N      | BS=8  R=%d   | BS=16 R=%d   | BS=32 R=%d   |\n", Rb0, Rb1, Rb2);
        printf("-----------------------------------------------------------------------------------------------------------\n");

        int baseBi = 0;
        int baseCount = usedCount[rrIdx][baseBi];

        for (int i = 0; i < baseCount; i++)
        {
            int N = usedNvals[rrIdx][baseBi][i];
            printf("%6d | ", N);

            for (int bi = 0; bi < 3; bi++)
            {
                int found = -1;
                for (int j = 0; j < usedCount[rrIdx][bi]; j++)
                {
                    if (usedNvals[rrIdx][bi][j] == N)
                    {
                        found = j;
                        break;
                    }
                }

                if (found < 0)
                {
                    printf("     -    | ");
                }
                else
                {
                    double ms = stage1_ms[rrIdx][bi][found];
                    bool ok = stage1_ok[rrIdx][bi][found];
                    if (!ok)
                        printf("%8.3fE | ", ms); // E = error
                    else
                        printf("%8.3f  | ", ms);
                }
            }
            printf("\n");
        }
        printf("-----------------------------------------------------------------------------------------------------------\n");
    };

    auto print_big_table_gf = [&](int rrIdx, const char *title)
    {
        printf("\n==================== %s (SPEED GF/s) ====================\n", title);
        printf("-----------------------------------------------------------------------------------------------------------\n");
        int Rb0 = (rrIdx == 0) ? (BSlist[0] / 2) : (2 * BSlist[0]);
        int Rb1 = (rrIdx == 0) ? (BSlist[1] / 2) : (2 * BSlist[1]);
        int Rb2 = (rrIdx == 0) ? (BSlist[2] / 2) : (2 * BSlist[2]);

        printf("N      | BS=8  R=%d   | BS=16 R=%d   | BS=32 R=%d   |\n", Rb0, Rb1, Rb2);
        printf("-----------------------------------------------------------------------------------------------------------\n");

        int baseBi = 0;
        int baseCount = usedCount[rrIdx][baseBi];

        for (int i = 0; i < baseCount; i++)
        {
            int N = usedNvals[rrIdx][baseBi][i];
            printf("%6d | ", N);

            for (int bi = 0; bi < 3; bi++)
            {
                int found = -1;
                for (int j = 0; j < usedCount[rrIdx][bi]; j++)
                {
                    if (usedNvals[rrIdx][bi][j] == N)
                    {
                        found = j;
                        break;
                    }
                }

                if (found < 0)
                {
                    printf("     -    | ");
                }
                else
                {
                    double gf = stage1_gf[rrIdx][bi][found];
                    bool ok = stage1_ok[rrIdx][bi][found];
                    if (!ok)
                        printf("%8.1fE | ", gf);
                    else
                        printf("%8.1f  | ", gf);
                }
            }
            printf("\n");
        }
        printf("-----------------------------------------------------------------------------------------------------------\n");
    };

    print_big_table_ms(0, "TABLE R1 (R < BS)");
    print_big_table_gf(0, "TABLE R1 (R < BS)");

    print_big_table_ms(1, "TABLE R2 (R > BS)");
    print_big_table_gf(1, "TABLE R2 (R > BS)");
}

int main(int argc, const char **argv)
{
    // Ustwawienie losowości
    srand((unsigned int)time(NULL));

    // Wybór karty graficznej
    int dev = 0;
    cudaSetDevice(dev);

    // Pobranie danych o karcie graficznej
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);

    run_auto_mode(prop);

    return 0;
}
