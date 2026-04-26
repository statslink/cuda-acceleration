#ifndef STRUCT_CUH
#define STRUCT_CUH

#include <cuda.h>

#define GAMMA 1.4

#define FLATTEN(i, j, step) ((i) + (j)*(step))

struct vecu {
    double* rho;
    double* momx;
    double* momy;
    double* ene;
};

void allocate(vecu &u, size_t nx, size_t ny) {
    size_t size = nx * ny * sizeof(double);

    cudaMallocManaged(&u.rho, size);
    cudaMallocManaged(&u.momx, size);
    cudaMallocManaged(&u.momy, size);
    cudaMallocManaged(&u.ene, size);
}

void free(vecu &u) {
    cudaFree(u.rho);
    cudaFree(u.momx);
    cudaFree(u.momy);
    cudaFree(u.ene);
}

__host__ __device__ inline double compute_pre(double energy, double density, double momx, double momy) {
    return (GAMMA - 1.0) * (energy - (momx * momx + momy * momy) / (2.0 * density));
}

__host__ __device__ inline double compute_ene(double pressure, double density, double velx, double vely) {
    double momx = density * velx;
    double momy = density * vely;

    return pressure / (GAMMA - 1.0) + (momx * momx + momy * momy) / (2.0 * density);
}



#endif STRUCT_H