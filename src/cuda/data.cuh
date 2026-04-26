#ifndef DATA_CUH
#define DATA_CUH

#include <cuda.h>
#include "struct.cuh"


__global__ void quadrant(vecu u, int nx, int ny, double x0, double y0, double dx, double dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(i < nx && j < ny) {
        int flat = FLATTEN(i, j, nx);

        double x = x0 + (i + 0.5) * dx;
        double y = y0 + (j + 0.5) * dy;  

        if(x < 0.5 && y < 0.5) {
            constexpr double density = 0.138, vel_x = 1.206, vel_y = 1.206, pressure = 0.029;
            u.rho[flat]  = density;
            u.momx[flat] = density * vel_x;
            u.momy[flat] = density * vel_y;
            u.ene[flat]  = compute_ene(pressure, density, vel_x, vel_y);
        }

        else if(x < 0.5 && y >= 0.5) {
            constexpr double density = 0.5323, vel_x = 1.206, vel_y = 0, pressure = 0.3;
            u.rho[flat]  = density;
            u.momx[flat] = density * vel_x;
            u.momy[flat] = density * vel_y;
            u.ene[flat]  = compute_ene(pressure, density, vel_x, vel_y);
        }

        else if(x >= 0.5 && y < 0.5) {
            constexpr double density = 0.5323, vel_x = 0, vel_y = 1.206, pressure = 0.3;
            u.rho[flat]  = density;
            u.momx[flat] = density * vel_x;
            u.momy[flat] = density * vel_y;
            u.ene[flat]  = compute_ene(pressure, density, vel_x, vel_y);
        }

        else if (x >= 0.5 && y >= 0.5){
            constexpr double density = 1.5, vel_x = 0, vel_y = 0, pressure = 1.5;
            u.rho[flat]  = density;
            u.momx[flat] = density * vel_x;
            u.momy[flat] = density * vel_y;
            u.ene[flat]  = compute_ene(pressure, density, vel_x, vel_y);
        }
    }
}


__global__ void bubble(vecu u, int nx, int ny, double x0, double y0, double dx, double dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < nx && j < ny){
        int flat = FLATTEN(i, j, nx);
        double x = x0 + (i + 0.5) * dx;
        double y = y0 + (j + 0.5) * dy;

        const double x_center = 0.035;
        const double y_center = 0.0445;
        const double radius = 0.025;

        double distance = sqrt((x - x_center)*(x - x_center) + (y - y_center)*(y - y_center));

            if (distance < radius) {
                constexpr double density = 0.214, vel_x = 0.0, vel_y = 0.0, pressure = 101325;
                u.rho[flat]  = density;
                u.momx[flat] = density * vel_x;
                u.momy[flat] = density * vel_y;
                u.ene[flat]  = compute_ene(pressure, density, vel_x, vel_y);
            }

            else {
                constexpr double density = 1.29, vel_x = 0.0, vel_y = 0.0, pressure = 101325;
                u.rho[flat]  = density;
                u.momx[flat] = density * vel_x;
                u.momy[flat] = density * vel_y;
                u.ene[flat]  = compute_ene(pressure, density, vel_x, vel_y);
            }

            if(x < 0.005) {
                constexpr double density = 1.776, vel_x = 113.6, vel_y = 0, pressure = 159056;
                u.rho[flat]  = density;
                u.momx[flat] = density * vel_x;
                u.momy[flat] = density * vel_y;
                u.ene[flat]  = compute_ene(pressure, density, vel_x, vel_y);
            }
    }
}

#endif DATA_H