#ifndef DATA_H
#define DATA_H

#include "struct.h"

inline void quadrant(vecu& u, const double x0, const double y0, const double dx, const double dy){
    const int nx = u.rho.size();
    const int ny = u.rho[0].size();

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double x = x0 + (i + 0.5) * dx;
            double y = y0 + (j + 0.5) * dy;
            if(x < 0.5 && y < 0.5) {
                constexpr double density = 0.138, vel_x = 1.206, vel_y = 1.206, pressure = 0.029;
                u.rho[i][j]  = density;
                u.momx[i][j] = density * vel_x;
                u.momy[i][j] = density * vel_y;
                u.ene[i][j]  = compute_ene(pressure, density, vel_x, vel_y);
            }
            else if(x < 0.5 && y >= 0.5) {
                constexpr double density = 0.5323, vel_x = 1.206, vel_y = 0, pressure = 0.3;
                u.rho[i][j]  = density;
                u.momx[i][j] = density * vel_x;
                u.momy[i][j] = density * vel_y;
                u.ene[i][j]  = compute_ene(pressure, density, vel_x, vel_y);
            }
            else if(x >= 0.5 && y < 0.5) {
                constexpr double density = 0.5323, vel_x = 0, vel_y = 1.206, pressure = 0.3;
                u.rho[i][j]  = density;
                u.momx[i][j] = density * vel_x;
                u.momy[i][j] = density * vel_y;
                u.ene[i][j]  = compute_ene(pressure, density, vel_x, vel_y);
            }
            else if(x >= 0.5 && y >= 0.5) {
                constexpr double density = 1.5, vel_x = 0, vel_y = 0, pressure = 1.5;
                u.rho[i][j]  = density;
                u.momx[i][j] = density * vel_x;
                u.momy[i][j] = density * vel_y;
                u.ene[i][j]  = compute_ene(pressure, density, vel_x, vel_y);
            }
        }
    }
}

inline void bubble(vecu& u, const double x0, const double y0, const double dx, const double dy) {
    const int nx = u.rho.size();
    const int ny = u.rho[0].size();

    constexpr double x_center = 0.035;
    constexpr double y_center = 0.0445;

    constexpr double radius = 0.025;

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            const double x = x0 + (i + 0.5) * dx;
            const double y = y0 + (j + 0.5) * dy;

            double distance = std::sqrt((x - x_center) * (x - x_center) + (y - y_center) * (y - y_center));

            if (distance < radius) {
                constexpr double density = 0.214, vel_x = 0, vel_y = 0, pressure = 101325;
                u.rho[i][j]  = density;
                u.momx[i][j] = density * vel_x;
                u.momy[i][j] = density * vel_y;
                u.ene[i][j]  = compute_ene(pressure, density, vel_x, vel_y);
            }
            else {
                constexpr double density = 1.29, vel_x = 0, vel_y = 0, pressure = 101325;
                u.rho[i][j]  = density;
                u.momx[i][j] = density * vel_x;
                u.momy[i][j] = density * vel_y;
                u.ene[i][j]  = compute_ene(pressure, density, vel_x, vel_y);
            }

            if (x < 0.005) {
                constexpr double density = 1.776, vel_x = 113.6, vel_y = 0, pressure = 159056;
                u.rho[i][j]  = density;
                u.momx[i][j] = density * vel_x;
                u.momy[i][j] = density * vel_y;
                u.ene[i][j]  = compute_ene(pressure, density, vel_x, vel_y);
            }
        }
    }
}

#endif
