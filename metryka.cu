// Kompilacja:
//   make
//
// Uruchomienie:
//   ./metryka

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

void tabliczkaZnamionowa(cudaDeviceProp prop)
{
    printf("|================================================|\n");
    printf("| TABLICZKA ZNAMIONOWA KARTY GRAFICZNEJ          |\n");
    printf("|================================================|\n");
    printf("| GPU: %30s | CC %2d.%-2d |\n", prop.name, prop.major, prop.minor);
    printf("|================================================|\n");
    printf("| Shared memory per block: %15zu bytes |\n", prop.sharedMemPerBlock);
    printf("| Max threads per block: %23d |\n", prop.maxThreadsPerBlock);
    printf("| Liczba SM: %35d |\n", prop.multiProcessorCount);
    printf("| Liczba rejestrów na SM: %22d |\n", prop.regsPerMultiprocessor);
    printf("|================================================|\n");
}

int main(int argc, const char **argv)
{
    // Wybór karty graficznej
    int dev = 0;
    cudaSetDevice(dev);

    // Pobranie danych o karcie graficznej
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);

    // Wyświetlenie tabliczki znamionowej
    tabliczkaZnamionowa(prop);

    return 0;
}
