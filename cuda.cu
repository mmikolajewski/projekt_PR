// radius_sum_auto.cu
// Projekt CUDA PKG: suma w promieniu R dla TAB[N*N] -> OUT[(N-2R)*(N-2R)]
// 4 warianty kernela: A (global coalesced), B (global non-coalesced),
// C (shared coalesced), D (shared bank conflicts)
// + tryb AUTO: leci wszystkie testy i drukuje blokami.
//
// Kompilacja:
//   nvcc -O3 radius_sum_auto.cu -o radius_sum
//
// Przykłady:
//   ./radius_sum --N 4096 --R 8 --BS 16 --k 1 --check
//   ./radius_sum --bench --N 4096 --R 8 --BS 16 --check
//   ./radius_sum --auto --check
//
// UWAGA: to jest kod "studencki" – prosto, czytelnie, bez wodotrysków.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#define CHECK_CUDA(call) do {                                  \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA ERROR: %s (%d) at %s:%d\n",       \
                cudaGetErrorString(err), (int)err, __FILE__, __LINE__); \
        exit(1);                                                \
    }                                                          \
} while(0)

static inline int iDivUp(int a, int b) { return (a + b - 1) / b; }

// =========================
// CPU reference (1 wątek)
// =========================
// Funkcja liczy OUT na CPU (sekwencyjnie). To jest "wzorzec" do testu poprawności GPU.
void cpu_radius_sum(const float* tab, float* out, int N, int R) {
    int outSize = N - 2 * R;
    for (int y = 0; y < outSize; y++) {
        for (int x = 0; x < outSize; x++) {
            float sum = 0.0f;
            int cy = y + R;
            int cx = x + R;
            for (int dy = -R; dy <= R; dy++) {
                for (int dx = -R; dx <= R; dx++) {
                    sum += tab[(cy + dy) * N + (cx + dx)];
                }
            }
            out[y * outSize + x] = sum;
        }
    }
}

// =========================
// Sprawdzenie poprawności
// =========================
// Porównuje GPU vs CPU. fullCheck=1 robi pełne porównanie (wolniej), inaczej próbkuje.
bool verify_result(const float* h_in, const float* h_out_gpu, int N, int R, bool fullCheck) {
    int outSize = N - 2 * R;
    if (outSize <= 0) return false;

    std::vector<float> ref((size_t)outSize * (size_t)outSize);
    cpu_radius_sum(h_in, ref.data(), N, R);

    int step = fullCheck ? 1 : 97; // "studencko": szybkie wykrywanie błędów indeksów
    for (int y = 0; y < outSize; y += step) {
        for (int x = 0; x < outSize; x += step) {
            float a = ref[(size_t)y * outSize + x];
            float b = h_out_gpu[(size_t)y * outSize + x];
            float d = fabsf(a - b);
            if (d > 1e-3f) {
                fprintf(stderr, "VERIFY FAIL at (y=%d,x=%d): CPU=%f GPU=%f diff=%f\n", y, x, a, b, d);
                return false;
            }
        }
    }
    return true;
}

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


// =========================
// KERNEL C: shared + efficient
// =========================
// Wariant C: najpierw ładowanie "tile" do shared (kolektywnie przez wszystkie wątki bloku),
// potem liczenie sum z shared.
// Shared ma rozmiar: (BS+2R) * (BS+2R).
__global__ void kernel_C(const float* tab, float* out, int N, int R, int k) {
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

        // Kolektywne ładowanie tile do shared
        for (int idx = tid; idx < tileSize; idx += nThreads) {
            int sy = idx / tileW;
            int sx = idx - sy * tileW;

            int inY = blockOutY + sy;
            int inX = blockOutX + sx;

            if (inY < N && inX < N) s[(size_t)sy * tileW + sx] = tab[(size_t)inY * N + inX];
            else                    s[(size_t)sy * tileW + sx] = 0.0f;
        }
        __syncthreads();

        int y = blockOutY + threadIdx.y;
        int x = blockOutX + threadIdx.x;

        if (y < outSize && x < outSize) {
            float sum = 0.0f;
            int csy = threadIdx.y + R;
            int csx = threadIdx.x + R;
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

// =========================
// Struktura wyniku jednego testu
// =========================
struct ResultRow {
    double msA=0, msB=0, msC=0, msD=0;
    double gfA=0, gfB=0, gfC=0, gfD=0;
    bool okA=true, okB=true, okC=true, okD=true;
    bool skipC=false, skipD=false;
};

// =========================
// Drukowanie nagłówka tabeli
// =========================
void print_table_header() {
    printf("N      |  ms_A   GF_A |  ms_B   GF_B |  ms_C   GF_C |  ms_D   GF_D | CHECK\n");
    printf("----------------------------------------------------------------------------------------------\n");
}

// =========================
// Drukowanie jednego wiersza tabeli
// =========================
void print_table_row(int N, const ResultRow& r) {
    char chk[64];
    snprintf(chk, sizeof(chk), "A:%s B:%s C:%s D:%s",
             r.okA ? "OK":"ERR",
             r.okB ? "OK":"ERR",
             r.skipC ? "SKIP" : (r.okC ? "OK":"ERR"),
             r.skipD ? "SKIP" : (r.okD ? "OK":"ERR"));

    printf("%6d | %6.3f %6.1f | %6.3f %6.1f | ",
           N, r.msA, r.gfA, r.msB, r.gfB);

    if (r.skipC) printf(" [SKIP] [SKIP] | ");
    else         printf("%6.3f %6.1f | ", r.msC, r.gfC);

    if (r.skipD) printf(" [SKIP] [SKIP] | ");
    else         printf("%6.3f %6.1f | ", r.msD, r.gfD);

    printf("%s\n", chk);
}

// =========================
// Wykonanie jednego testu (N,R,BS,k) – liczy A,B,C,D + opcjonalnie check
// =========================
// To jest kluczowa funkcja "pomiarowa":
// 1) alokuje tablice
// 2) wypełnia dane (niejednorodne)
// 3) kopiuje na GPU
// 4) uruchamia 4 kerneli i mierzy cudaEvent
// 5) liczy GFLOP/s
// 6) opcjonalnie porównuje z CPU
ResultRow run_one_case(int N, int R, int BS, int k, bool doCheck, bool fullCheck, int nIter, cudaDeviceProp prop) {
    ResultRow res{};
    int outSize = N - 2 * R;
    if (outSize <= 0) {
        res.okA=res.okB=res.okC=res.okD=false;
        res.skipC=res.skipD=true;
        return res;
    }

    size_t sizeIn  = (size_t)N * (size_t)N * sizeof(float);
    size_t sizeOut = (size_t)outSize * (size_t)outSize * sizeof(float);

    float* h_in  = (float*)malloc(sizeIn);
    float* h_out = (float*)malloc(sizeOut);

    // niejednorodne dane: random + lekki gradient
    srand(123);
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            float rnd = (float)rand() / (float)RAND_MAX;
            float g   = 0.001f * (float)(x + 3*y);
            h_in[(size_t)y * N + x] = rnd + g;
        }
    }

    float *d_in=nullptr, *d_out=nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, sizeIn));
    CHECK_CUDA(cudaMalloc(&d_out, sizeOut));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, sizeIn, cudaMemcpyHostToDevice));

    dim3 block(BS, BS);
    int grid_w = iDivUp(outSize, BS * k);
    int grid_h = iDivUp(outSize, BS);
    dim3 grid(grid_w, grid_h);

    // Operacje: tu liczymy (2R+1)^2 "dodawań na piksel" – prosto.
    // (W sprawozdaniu możesz dopisać dokładną definicję FLOP).
    double opsPerPixel = (double)(2*R + 1) * (double)(2*R + 1);
    double totalOps = (double)outSize * (double)outSize * opsPerPixel;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    auto measure = [&](auto kernel, double &msOut, double &gfOut, bool &okOut, size_t shBytes, bool &skipFlag) {
        if (shBytes > 0 && shBytes > (size_t)prop.sharedMemPerBlock) {
            skipFlag = true;
            msOut = 0.0;
            gfOut = 0.0;
            okOut = true; // skip to nie fail
            return;
        }

        // warmup
        kernel<<<grid, block, shBytes>>>(d_in, d_out, N, R, k);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < nIter; i++) {
            kernel<<<grid, block, shBytes>>>(d_in, d_out, N, R, k);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        msOut = (double)ms / (double)nIter;

        double sec = msOut / 1000.0;
        gfOut = (totalOps * 1e-9) / sec;

        if (doCheck) {
            CHECK_CUDA(cudaMemcpy(h_out, d_out, sizeOut, cudaMemcpyDeviceToHost));
            okOut = verify_result(h_in, h_out, N, R, fullCheck);
        } else {
            okOut = true;
        }
    };

    // A, B (global)
    bool dummySkip=false;
    measure(kernel_A, res.msA, res.gfA, res.okA, 0, dummySkip);
    measure(kernel_B, res.msB, res.gfB, res.okB, 0, dummySkip);

    // shared bytes for C/D
    size_t shBytes = (size_t)(BS + 2*R) * (size_t)(BS + 2*R) * sizeof(float);
    measure(kernel_C, res.msC, res.gfC, res.okC, shBytes, res.skipC);
    measure(kernel_D, res.msD, res.gfD, res.okD, shBytes, res.skipD);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return res;
}

// =========================
// Wyznaczenie N_wys (plateau) – prosto, studencko
// =========================
// Bierzemy prędkość (GFLOP/s) z wariantu C, a jeśli C jest SKIP, to bierzemy A.
// Kryterium plateau: wzrost < eps (np. 5%) przez 2 kolejne kroki.
int find_Nwys_from_rows(const std::vector<int>& Nlist, const std::vector<ResultRow>& rows, double eps) {
    auto pickGF = [&](const ResultRow& r) -> double {
        if (!r.skipC) return r.gfC;
        return r.gfA;
    };

    for (int i = 0; i + 2 < (int)Nlist.size(); i++) {
        double g0 = pickGF(rows[i]);
        double g1 = pickGF(rows[i+1]);
        double g2 = pickGF(rows[i+2]);

        if (g0 <= 0 || g1 <= 0 || g2 <= 0) continue;

        double rel1 = (g1 - g0) / g0;
        double rel2 = (g2 - g1) / g1;

        if (rel1 < eps && rel2 < eps) {
            // minimalne N, od którego plateau "widać"
            return Nlist[i];
        }
    }
    // Jak nie znaleziono plateau, bierzemy największe N
    return Nlist.back();
}

// =========================
// Proste parsowanie argumentów
// =========================
static int arg_int(int argc, char** argv, const char* key, int defVal) {
    for (int i = 1; i < argc - 1; i++) if (strcmp(argv[i], key) == 0) return atoi(argv[i + 1]);
    return defVal;
}
static bool arg_flag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; i++) if (strcmp(argv[i], key) == 0) return true;
    return false;
}

// =========================
// Tryb bench (k=1,2,8 dla jednego N,R,BS)
// =========================
void run_bench_mode(int N, int R, int BS, bool check, bool fullCheck, cudaDeviceProp prop) {
    int ks[3] = {1,2,8};
    printf("\n==================== BENCH MODE (k sweep) ====================\n");
    printf("N=%d  R=%d  BS=%d   (k = 1,2,8)\n", N, R, BS);
    printf("--------------------------------------------------------------\n");
    print_table_header();
    for (int i = 0; i < 3; i++) {
        ResultRow r = run_one_case(N, R, BS, ks[i], check, fullCheck, 10, prop);
        // tu drukujemy "N" jako wiersz, ale żeby było czytelnie w bench,
        // drukujemy w miejscu N wartość "k"
        // (studencko: prosto – zmieniamy znaczenie pierwszej kolumny)
        printf("k=%-4d| %6.3f %6.1f | %6.3f %6.1f | ",
               ks[i], r.msA, r.gfA, r.msB, r.gfB);
        if (r.skipC) printf(" [SKIP] [SKIP] | ");
        else         printf("%6.3f %6.1f | ", r.msC, r.gfC);
        if (r.skipD) printf(" [SKIP] [SKIP] | ");
        else         printf("%6.3f %6.1f | ", r.msD, r.gfD);

        char chk[64];
        snprintf(chk, sizeof(chk), "A:%s B:%s C:%s D:%s",
                 r.okA ? "OK":"ERR",
                 r.okB ? "OK":"ERR",
                 r.skipC ? "SKIP" : (r.okC ? "OK":"ERR"),
                 r.skipD ? "SKIP" : (r.okD ? "OK":"ERR"));
        printf("%s\n", chk);
    }
    printf("==============================================================\n");
}

// =========================
// Tryb auto (wszystko leci samo, blokami)
// =========================
// Etap 1: dla BS=8/16/32 i R1=BS/2, R2=2*BS robimy sweep po N i szukamy N_wys.
// Etap 2: dla N=2*N_wys robimy sweep po k=1/2/8.
void run_auto_mode(bool check, bool fullCheck, cudaDeviceProp prop) {
    // "studencka" lista N – łatwo widać plateau
    //std::vector<int> Nlist = {512, 1024, 2048, 4096, 8192, 16384}; za duzo wolno przy 16384
    std::vector<int> Nlist = {1024, 2048, 4096, 8192};


    int BSlist[3] = {8,16,32};
    int ks[3] = {1,2,8};

    // kryterium plateau: <5% wzrost przez 2 kroki
    double eps = 0.05;

    // zapamiętamy N_wys dla każdego BS i dla R1/R2
    struct NW { int R1; int R2; int Nwys1; int Nwys2; };
    NW summary[3];


        // =========================================================
    // STAGE 1: ZBIERANIE WYNIKÓW DO TABEL ZBIORCZYCH (R1 i R2)
    // rr = 0 -> R1 (R < BS)
    // rr = 1 -> R2 (R > BS)
    // bi = 0..2 -> BS = {8,16,32}
    // slot = index po N (kolejny element listy N)
    // =========================================================

    const int MAXN = 16; // na zapas, bo Nlist ma mało elementów

    double stage1_ms[2][3][MAXN] = {};   // czas [ms] (wybrane: C albo fallback A)
    double stage1_gf[2][3][MAXN] = {};   // prędkość [GF/s] (wybrane: C albo fallback A)
    bool   stage1_ok[2][3][MAXN] = {};   // poprawność (dla check)

    int    usedCount[2][3] = {};         // ile wpisów N zapisano dla (rr,bi)
    int    usedNvals[2][3][MAXN] = {};   // jakie N zapisano dla (rr,bi,slot)


    printf("\n==================== AUTO MODE ====================\n");
    printf("Plan: BS in {8,16,32}, R1=BS/2 (R<BS), R2=2*BS (R>BS)\n");
    printf("N list: ");
    for (int v : Nlist) printf("%d ", v);
    printf("\nPlateau rule: GF change < 5%% for next 2 steps (using GF_C, fallback GF_A if C=SKIP)\n");
    printf("===================================================\n");

    // ---------- ETAP 1: N_wys ----------
    printf("\n#################### STAGE 1: SATURATION (find N_wys) ####################\n");

    for (int bi = 0; bi < 3; bi++) {
        int BS = BSlist[bi];
        int R1 = BS / 2;
        int R2 = 2 * BS;

        summary[bi].R1 = R1;
        summary[bi].R2 = R2;

        for (int rr = 0; rr < 2; rr++) {
            int R = (rr == 0) ? R1 : R2;

            printf("\n==================== SATURATION TEST ====================\n");
            printf("BS=%d  R=%d   (%s)\n", BS, R, (R < BS ? "R < BS" : "R > BS"));
            printf("k = 1\n");
            printf("---------------------------------------------------------\n");

            print_table_header();

            std::vector<ResultRow> rows;
            std::vector<int> usedN;

            for (int Ni = 0; Ni < (int)Nlist.size(); Ni++) {
                int N = Nlist[Ni];
                if (N <= 2 * R) {
                    // warunek z treści: N > 2R
                    continue;
                }
                //ResultRow r = run_one_case(N, R, BS, 1, check, fullCheck, 10, prop); zmiana na 3 bo za długo idzie 
                bool doCheckThis = check && (rows.empty());  // check tylko dla pierwszego N w bloku
                ResultRow r = run_one_case(N, R, BS, 1, doCheckThis, fullCheck, 3, prop);

                print_table_row(N, r);
                int rrIdx = rr;  
                int biIdx = bi; 

                int slot = usedCount[rrIdx][biIdx];
                if (slot < MAXN) {
                    usedNvals[rrIdx][biIdx][slot] = N;

                    // bierzemy metrykę z C, a jak C jest SKIP to fallback A
                    double msPick = r.skipC ? r.msA : r.msC;
                    double gfPick = r.skipC ? r.gfA : r.gfC;
                    bool   okPick = r.skipC ? r.okA : r.okC;

                    stage1_ms[rrIdx][biIdx][slot] = msPick;
                    stage1_gf[rrIdx][biIdx][slot] = gfPick;
                    stage1_ok[rrIdx][biIdx][slot] = okPick;

                    usedCount[rrIdx][biIdx] = slot + 1;
                }
                rows.push_back(r);
                usedN.push_back(N);
            }

            int Nwys = usedN.empty() ? -1 : find_Nwys_from_rows(usedN, rows, eps);

            if (rr == 0) summary[bi].Nwys1 = Nwys;
            else         summary[bi].Nwys2 = Nwys;

            printf("---------------------------------------------------------\n");
            printf("N_wys (estimated) = %d\n", Nwys);
            printf("=========================================================\n");
        }
    }
    auto print_big_table_ms = [&](int rrIdx, const char* title) {
    printf("\n==================== %s (TIME ms) ====================\n", title);
    printf("-----------------------------------------------------------------------------------------------------------\n");
    int Rb0 = (rrIdx==0) ? (BSlist[0]/2) : (2*BSlist[0]);
    int Rb1 = (rrIdx==0) ? (BSlist[1]/2) : (2*BSlist[1]);
    int Rb2 = (rrIdx==0) ? (BSlist[2]/2) : (2*BSlist[2]);

    printf("N      | BS=8  R=%d   | BS=16 R=%d   | BS=32 R=%d   |\n", Rb0, Rb1, Rb2);
    printf("-----------------------------------------------------------------------------------------------------------\n");

    int baseBi = 0;
    int baseCount = usedCount[rrIdx][baseBi];

    for (int i = 0; i < baseCount; i++) {
        int N = usedNvals[rrIdx][baseBi][i];
        printf("%6d | ", N);

        for (int bi = 0; bi < 3; bi++) {
            int found = -1;
            for (int j = 0; j < usedCount[rrIdx][bi]; j++) {
                if (usedNvals[rrIdx][bi][j] == N) { found = j; break; }
            }

            if (found < 0) {
                printf("     -    | ");
            } else {
                double ms = stage1_ms[rrIdx][bi][found];
                bool ok   = stage1_ok[rrIdx][bi][found];
                if (!ok) printf("%8.3fE | ", ms);   // E = error
                else     printf("%8.3f  | ", ms);
            }
        }
        printf("\n");
    }
    printf("-----------------------------------------------------------------------------------------------------------\n");
};

auto print_big_table_gf = [&](int rrIdx, const char* title) {
    printf("\n==================== %s (SPEED GF/s) ====================\n", title);
    printf("-----------------------------------------------------------------------------------------------------------\n");
    int Rb0 = (rrIdx==0) ? (BSlist[0]/2) : (2*BSlist[0]);
    int Rb1 = (rrIdx==0) ? (BSlist[1]/2) : (2*BSlist[1]);
    int Rb2 = (rrIdx==0) ? (BSlist[2]/2) : (2*BSlist[2]);

    printf("N      | BS=8  R=%d   | BS=16 R=%d   | BS=32 R=%d   |\n", Rb0, Rb1, Rb2);
    printf("-----------------------------------------------------------------------------------------------------------\n");

    int baseBi = 0;
    int baseCount = usedCount[rrIdx][baseBi];

    for (int i = 0; i < baseCount; i++) {
        int N = usedNvals[rrIdx][baseBi][i];
        printf("%6d | ", N);

        for (int bi = 0; bi < 3; bi++) {
            int found = -1;
            for (int j = 0; j < usedCount[rrIdx][bi]; j++) {
                if (usedNvals[rrIdx][bi][j] == N) { found = j; break; }
            }

            if (found < 0) {
                printf("     -    | ");
            } else {
                double gf = stage1_gf[rrIdx][bi][found];
                bool ok   = stage1_ok[rrIdx][bi][found];
                if (!ok) printf("%8.1fE | ", gf);
                else     printf("%8.1f  | ", gf);
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



    // ---------- ETAP 2: k ----------
    printf("\n#################### STAGE 2: k impact (N = 2*N_wys) ####################\n");

    for (int bi = 0; bi < 3; bi++) {
        int BS = BSlist[bi];
        int Rvals[2] = {summary[bi].R1, summary[bi].R2};
        int NwysVals[2] = {summary[bi].Nwys1, summary[bi].Nwys2};

        for (int rr = 0; rr < 2; rr++) {
            int R = Rvals[rr];
            int Nwys = NwysVals[rr];
            int N = 2 * Nwys;

            // --- LIMIT żeby nie zabić sesji na serwerze ---
            const int MAX_KTEST_N = 8192;   // możesz dać 4096 jeśli nadal za długo
            if (N > MAX_KTEST_N) N = MAX_KTEST_N;


            //dla r =  64 idzie w cholere długo, wiec aby wgl sie skonczyło daje warunek ze dla >64 ograniczenie do 4096
            //Ze względu na ograniczenia czasowe środowiska uruchomieniowego zastosowano limit maksymalnego N w teście wpływu k.
            // Jeżeli 2*N_wys przekraczało limit, przyjęto N=8192 (a dla R≥64: N=4096) jako największą sensowną instancję możliwą do wykonania w czasie dostępnej sesji.

            if (R >= 64 && N > 4096) N = 4096;

            // jeśli z jakiegoś powodu N jest niepoprawne: jak co to zakoentwaoc gore i bedize miagac na 16..
            if (N <= 2 * R) N = 4 * R + 2;


            printf("\n==================== K TEST ====================\n");
            //printf("BS=%d  R=%d   N_wys=%d   => N=2*N_wys=%d\n", BS, R, Nwys, N);
            printf("BS=%d  R=%d   N_wys=%d   => N=%d (target 2*N_wys=%d)\n", BS, R, Nwys, N, 2*Nwys);

            printf("k list: 1 2 8\n");
            printf("------------------------------------------------\n");

            printf("k      |    ms_A   GF_A |    ms_B   GF_B |    ms_C   GF_C |    ms_D   GF_D |   CHECK\n");
            printf("----------------------------------------------------------------------------------------------\n");

            for (int ki = 0; ki < 3; ki++) {
                int k = ks[ki];
                bool doCheckThis = check && (ki == 0); // check tylko dla k=1
                //ResultRow r = run_one_case(N, R, BS, k, doCheckThis, fullCheck, 3, prop); to potona
                ResultRow r = run_one_case(N, R, BS, k, doCheckThis, fullCheck, 1, prop);





                // druk jak w bench: pierwsza kolumna to k
                printf("%-6d| %6.3f %6.1f | %6.3f %6.1f | ",
                       k, r.msA, r.gfA, r.msB, r.gfB);

                if (r.skipC) printf(" [SKIP] [SKIP] | ");
                else         printf("%6.3f %6.1f | ", r.msC, r.gfC);

                if (r.skipD) printf(" [SKIP] [SKIP] | ");
                else         printf("%6.3f %6.1f | ", r.msD, r.gfD);

                char chk[64];
                snprintf(chk, sizeof(chk), "A:%s B:%s C:%s D:%s",
                         r.okA ? "OK":"ERR",
                         r.okB ? "OK":"ERR",
                         r.skipC ? "SKIP" : (r.okC ? "OK":"ERR"),
                         r.skipD ? "SKIP" : (r.okD ? "OK":"ERR"));
                printf("%s\n", chk);
            }

            printf("================================================\n");
        }
    }

    // ---------- Summary ----------
    printf("\n#################### SUMMARY (N_wys) ####################\n");
    for (int bi = 0; bi < 3; bi++) {
        int BS = BSlist[bi];
        printf("BS=%d | R1=%d -> N_wys=%d | R2=%d -> N_wys=%d\n",
               BS, summary[bi].R1, summary[bi].Nwys1, summary[bi].R2, summary[bi].Nwys2);
    }
    printf("#########################################################\n");
}

// =========================
// main
// =========================
int main(int argc, char** argv) {
    int N  = arg_int(argc, argv, "--N", 2048);
    int R  = arg_int(argc, argv, "--R", 8);
    int BS = arg_int(argc, argv, "--BS", 16);
    int k  = arg_int(argc, argv, "--k", 1);

    bool check = arg_flag(argc, argv, "--check");
    bool full  = arg_flag(argc, argv, "--fullcheck");
    bool bench = arg_flag(argc, argv, "--bench");
    bool autoMode = arg_flag(argc, argv, "--auto");

    // GPU info
    int dev = 0;
    CHECK_CUDA(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s | CC %d.%d | sharedMemPerBlock=%zu bytes\n",
           prop.name, prop.major, prop.minor, (size_t)prop.sharedMemPerBlock);

    if (autoMode) {
        run_auto_mode(check, full, prop);
        return 0;
    }

    if (bench) {
        run_bench_mode(N, R, BS, check, full, prop);
        return 0;
    }

    // pojedynczy test (1 linia)
    printf("\n==================== SINGLE RUN ====================\n");
    printf("N=%d R=%d BS=%d k=%d\n", N, R, BS, k);
    printf("----------------------------------------------------\n");
    print_table_header();

    ResultRow r = run_one_case(N, R, BS, k, check, full, 10, prop);
    print_table_row(N, r);

    printf("====================================================\n");
    printf("Tip: --check (sampled) albo --fullcheck (pełne, wolniej)\n");
    printf("Try: --bench (k=1,2,8) albo --auto (pełen komplet testów)\n");

    return 0;
}
