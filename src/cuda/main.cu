#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

#include <cuda.h>

#include "struct.cuh"
#include "data.cuh"
#include "schemes.cuh"

#define HALF 0.5

__global__ void wavespeed(const vecu u, int nx, int ny, double* speeds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    int local = threadIdx.y * blockDim.x +  threadIdx.x;

    extern __shared__ double shared[];

    double* s_rho = shared;
    double* s_ene = s_rho + blockDim.x * blockDim.y;
    double* s_momx = s_ene + blockDim.x * blockDim.y;
    double* s_momy = s_momx + blockDim.x * blockDim.y;

    if(i < nx - 1 && j < ny - 1) {
        int flat = FLATTEN(i, j, nx);

        s_rho[local]  = u.rho[flat];
        s_ene[local]  = u.ene[flat];
        s_momx[local] = u.momx[flat];
        s_momy[local] = u.momy[flat];
    }
    
    __syncthreads();
    
    if(i < nx - 1 && j < ny - 1) {
        int flat = FLATTEN(i, j, nx);

        double pressure = compute_pre(s_ene[local], s_rho[local], s_momx[local], s_momy[local]);

        double cs = sqrt(GAMMA * pressure / s_rho[local]);
        double speed_x = fabs(s_momx[local] / s_rho[local]) + cs;
        double speed_y = fabs(s_momy[local] / s_rho[local]) + cs;
        
        speeds[flat] = (speed_x > speed_y) ? speed_x : speed_y;
    }
}


int main() {
    std::string problem = "quadrant"; // or bubble
    
    std::string path = R"(\vis\xy\)";

    // std::ofstream density(path + problem + "-cuda-den.csv");
    // std::ofstream velx(path + problem + "-cuda-velx.csv");
    // std::ofstream vely(path + problem + "-cuda-vely.csv");
    std::ofstream pressure(path + problem + "-cuda-pre.csv");

    // std::ofstream time(path + problem + "-cuda-time.csv");
    

    // quadrant domain 
    
    
    int d_nx = 400, d_ny = 400;
    double tStart = 0.0, tEnd = 0.3;
    double x0 = 0.0, x1 = 1;
    double y0 = 0.0, y1 = 1; 
    

    // bubble domain
    
    /*
    int d_nx = 500, d_ny = 200;
    double tStart = 0.0, tEnd = 0.001;
    double x0 = 0.0, x1 = 0.225;
    double y0 = 0.0, y1 = 0.089;
    */


    double dx = (x1 - x0) / d_nx;
    double dy = (y1 - y0) / d_ny;
    double C = 0.7;

    int ghost = 2;

    int nx = d_nx + 2 * ghost;
    int ny = d_ny + 2 * ghost; 

    int shared = 4 * 16 * 16 * sizeof(double);

    vecu u, uBar, uPlus, flux_x, flux_y;

    allocate(u, nx, ny);
    allocate(uBar, nx, ny);
    allocate(uPlus, nx, ny);

    allocate(flux_x, nx, ny);
    allocate(flux_y, nx, ny);

    vecu uL_bar, uR_bar, uL_bar_half, uR_bar_half;
    
    allocate(uL_bar, nx, ny);
    allocate(uR_bar, nx, ny);
    allocate(uL_bar_half, nx, ny);
    allocate(uR_bar_half, nx, ny);

    dim3 block_size(16, 16);
    dim3 grid_size((nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y);

    if(problem == "quadrant"){
        quadrant<<<grid_size, block_size>>>(u, nx, ny, x0, y0, dx, dy);
        cudaDeviceSynchronize();
    }
    else if (problem == "bubble"){
        bubble<<<grid_size, block_size>>>(u, nx, ny, x0, y0, dx, dy);
        cudaDeviceSynchronize();
    }
    
    cudaFuncSetCacheConfig(wavespeed, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(fv_x, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(fv_y, cudaFuncCachePreferL1);

    cudaEvent_t algo_start, algo_stop;
    cudaEventCreate(&algo_start);
    cudaEventCreate(&algo_stop);

    cudaEventRecord(algo_start);

    double t = tStart;
    while(t < tEnd) {
        // time << t << ",";

        ubound<<<grid_size, block_size>>>(u.rho, nx, ny, ghost);
        ubound<<<grid_size, block_size>>>(u.momx, nx, ny, ghost);
        ubound<<<grid_size, block_size>>>(u.momy, nx, ny, ghost);
        ubound<<<grid_size, block_size>>>(u.ene, nx, ny, ghost);
        cudaDeviceSynchronize();

        double* speeds;
        cudaMallocManaged(&speeds, nx * ny * sizeof(double));
        
        dim3 wavespeed_grid((nx - 2 + block_size.x - 1) / block_size.x, (ny - 2 + block_size.y - 1) / block_size.y);
            
        wavespeed<<<wavespeed_grid, block_size, shared>>>(u, nx, ny, speeds);
        
        cudaDeviceSynchronize();
        double speed = 0.0;
        for (int k = 0; k < nx * ny; k++) { 
            if(speeds[k] > speed) {
                speed = speeds[k];
            }
        }

        cudaFree(speeds);
        
        double dt = C * (std::min(dx, dy) / speed);
        t += dt;
        std::cout << "t: " << t << std::endl;

        cudaDeviceSynchronize();

        SLIC_x_reconstruciton<<<grid_size, block_size>>>(u, uL_bar, uR_bar, dx, dt, nx, ny);
        cudaDeviceSynchronize();

        SLIC_x_halfstep<<<grid_size, block_size>>>(uL_bar, uR_bar, uL_bar_half, uR_bar_half, dx, dt, nx, ny);
        cudaDeviceSynchronize();

        FORCE_x<<<grid_size, block_size>>>(uL_bar_half, uR_bar_half, flux_x, dx, dt, nx, ny);
        cudaDeviceSynchronize();

        fv_x<<<grid_size, block_size>>>(u, flux_x, uBar, dx, dt, nx, ny, ghost);
        cudaDeviceSynchronize();

        ubound<<<grid_size, block_size>>>(uBar.rho, nx, ny, ghost);
        ubound<<<grid_size, block_size>>>(uBar.momx, nx, ny, ghost);
        ubound<<<grid_size, block_size>>>(uBar.momy, nx, ny, ghost);
        ubound<<<grid_size, block_size>>>(uBar.ene, nx, ny, ghost);
        cudaDeviceSynchronize();
        
        SLIC_y_reconstruction<<<grid_size, block_size>>>(uBar, uL_bar, uR_bar, dy, dt, nx, ny);
        cudaDeviceSynchronize();

        SLIC_y_halfstep<<<grid_size, block_size>>>(uL_bar, uR_bar, uL_bar_half, uR_bar_half, dy, dt, nx, ny);
        cudaDeviceSynchronize();

        FORCE_y<<<grid_size, block_size>>>(uL_bar_half, uR_bar_half, flux_y, dy, dt, nx, ny);
        cudaDeviceSynchronize();

        fv_y<<<grid_size, block_size>>>(uBar, flux_y, uPlus, dy, dt, nx, ny, ghost); //order
        cudaDeviceSynchronize();

        ubound<<<grid_size, block_size>>>(uPlus.rho, nx, ny, ghost);
        ubound<<<grid_size, block_size>>>(uPlus.momx, nx, ny, ghost);
        ubound<<<grid_size, block_size>>>(uPlus.momy, nx, ny, ghost);
        ubound<<<grid_size, block_size>>>(uPlus.ene, nx, ny, ghost);
        cudaDeviceSynchronize();
        
        
        for (int j = ghost; j < ny - ghost; j++) {
            for (int i = ghost; i < nx - ghost; i++) {
                int flat = FLATTEN(i, j, nx);
                //density << u.rho[flat] << ",";
                //velx << u.momx[flat] / u.rho[flat] << ",";
                //vely << u.momy[flat] / u.rho[flat] << ",";
                pressure << compute_pre(u.ene[flat], u.rho[flat], u.momx[flat], u.momy[flat]) << ",";
            }
        }

        //density << std::endl;
        //velx << std::endl;
        //vely << std::endl;
        pressure << std::endl;  
        
        
        cudaMemcpy(u.rho, uPlus.rho, nx * ny * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(u.momx, uPlus.momx, nx * ny * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(u.momy, uPlus.momy, nx * ny * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(u.ene, uPlus.ene, nx * ny * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(algo_stop);
    cudaEventSynchronize(algo_stop);

    float algo_time;
    cudaEventElapsedTime(&algo_time, algo_start, algo_stop);
    std::cout << problem + " main loop time: " << algo_time << " ms" << std::endl;

    cudaEventDestroy(algo_start);
    cudaEventDestroy(algo_stop);

    // manually running one timestep more to measure performances

    double* speeds;
    cudaMallocManaged(&speeds, nx * ny * sizeof(double));
    
    dim3 wavespeed_grid((nx - 2 + block_size.x - 1) / block_size.x, (ny - 2 + block_size.y - 1) / block_size.y);
    
    wavespeed<<<wavespeed_grid, block_size, shared>>>(u, nx, ny, speeds);
    cudaDeviceSynchronize();

    double speed = 0;

    for (int i = 0; i < nx * ny; i++) { 
        if(speeds[i] > speed) {
            speed = speeds[i]; 
        }
    }
    cudaFree(speeds);
    
    double dt = C * (std::min(dx, dy) / speed);

    // x-fluxes 

    cudaEvent_t x_fluxes_start, x_fluxes_stop;
    cudaEventCreate(&x_fluxes_start);
    cudaEventCreate(&x_fluxes_stop);

    cudaEventRecord(x_fluxes_start);
    
    
    SLIC_x_reconstruciton<<<grid_size, block_size>>>(u, uL_bar, uR_bar, dx, dt, nx, ny);
    cudaDeviceSynchronize();

    SLIC_x_halfstep<<<grid_size, block_size>>>(uL_bar, uR_bar, uL_bar_half, uR_bar_half, dx, dt, nx, ny);
    cudaDeviceSynchronize();

    FORCE_x<<<grid_size, block_size>>>(uL_bar_half, uR_bar_half, flux_x, dx, dt, nx, ny);
    cudaDeviceSynchronize();


    cudaEventRecord(x_fluxes_stop);
    cudaEventSynchronize(x_fluxes_stop);

    float x_fluxes_time;
    cudaEventElapsedTime(&x_fluxes_time, x_fluxes_start, x_fluxes_stop);
    std::cout << problem + " x_fluxes time: " << x_fluxes_time << " ms" << std::endl;

    cudaEventDestroy(x_fluxes_start);
    cudaEventDestroy(x_fluxes_stop);

    // x-fv udpate 

    cudaEvent_t x_fv_start, x_fv_stop;
    cudaEventCreate(&x_fv_start);
    cudaEventCreate(&x_fv_stop);

    cudaEventRecord(x_fv_start);
    
    fv_x<<<grid_size, block_size>>>(uBar, flux_y, uPlus, dx, dt, nx, ny, ghost);
    cudaDeviceSynchronize();

    cudaEventRecord(x_fv_stop);
    cudaEventSynchronize(x_fv_stop);

    float x_fv_time;
    cudaEventElapsedTime(&x_fv_time, x_fv_start, x_fv_stop);
    std::cout << problem + " x_fv time: " << x_fv_time << " ms" << std::endl;

    cudaEventDestroy(x_fv_start);
    cudaEventDestroy(x_fv_stop);

    // y-fluxes

    cudaEvent_t y_fluxes_start, y_fluxes_stop;
    cudaEventCreate(&y_fluxes_start);
    cudaEventCreate(&y_fluxes_stop);

    cudaEventRecord(y_fluxes_start);
    
    SLIC_y_reconstruction<<<grid_size, block_size>>>(uBar, uL_bar, uR_bar, dy, dt, nx, ny);
    cudaDeviceSynchronize();

    SLIC_y_halfstep<<<grid_size, block_size>>>(uL_bar, uR_bar, uL_bar_half, uR_bar_half, dy, dt, nx, ny);
    cudaDeviceSynchronize();

    FORCE_y<<<grid_size, block_size>>>(uL_bar_half, uR_bar_half, flux_y, dy, dt, nx, ny);
    cudaDeviceSynchronize();

    cudaEventRecord(y_fluxes_stop);
    cudaEventSynchronize(y_fluxes_stop);

    float y_fluxes_time;
    cudaEventElapsedTime(&y_fluxes_time, y_fluxes_start, y_fluxes_stop);
    std::cout << problem + " y_fluxes time: " << y_fluxes_time << " ms" << std::endl;

    cudaEventDestroy(y_fluxes_start);
    cudaEventDestroy(y_fluxes_stop);

    // y-fv

    cudaEvent_t y_fv_start, y_fv_stop;
    cudaEventCreate(&y_fv_start);
    cudaEventCreate(&y_fv_stop);

    cudaEventRecord(y_fv_start, 0);

    fv_y<<<grid_size, block_size>>>(uBar, flux_y, uPlus, dy, dt, nx, ny, ghost);
    cudaDeviceSynchronize();

    cudaEventRecord(y_fv_stop, 0);
    cudaEventSynchronize(y_fv_stop);

    float y_fv_time;
    cudaEventElapsedTime(&y_fv_time, y_fv_start, y_fv_stop);
    std::cout << problem + " y_fv time: " << y_fv_time << " ms" << std::endl;

    cudaEventDestroy(y_fv_start);
    cudaEventDestroy(y_fv_stop);

    free(u);
    free(uBar);
    free(uPlus);
    free(flux_x);
    free(flux_y);

    return 0;
}
