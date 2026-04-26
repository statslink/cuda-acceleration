#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <string>

#include "data.h"
#include "schemes.h"

double wavespeed(const vecu& u) {
    double speed = 0;
    const int nx = u.rho.size();
    const int ny = u.rho[0].size();

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            const double pressure = compute_pre(u.ene[i][j], u.rho[i][j], u.momx[i][j], u.momy[i][j]);
            const double cs = sqrt(GAMMA * pressure / u.rho[i][j]);

            double speed_x = fabs(u.momx[i][j] / u.rho[i][j]) + cs;
            double speed_y = fabs(u.momy[i][j] / u.rho[i][j]) + cs;

            speed = std::max({speed, speed_x, speed_y});
        }
    }
    return speed;
}


int main() {
    std::string problem = "quadrant";

    std::string path = R"(\vis\xy\)";

    // std::ofstream density(path + problem + "-cpu-den.csv");
    // std::ofstream velx(path + problem + "-cpu-velx.csv");
    // std::ofstream vely(path + problem + "-cpu-vely.csv");
    std::ofstream pressure(path + problem + "-cpu-pre.csv");

    std::ofstream time(path + problem + "-cpu-time.csv");

    // quadrant domain


    int nx = 400, ny = 400;
    double tStart = 0.0, tEnd = 0.3;
    double x0 = 0.0, x1 = 1;
    double y0 = 0.0, y1 = 1;


    // bubble domain

    /*
    int nx = 500, ny = 200;
    double tStart = 0.0, tEnd = 0.001;
    double x0 = 0.0, x1 = 0.225;
    double y0 = 0.0, y1 = 0.089;
    */

    double dx = (x1 - x0) / nx;
    double dy = (y1 - y0) / ny;
    double C = 0.7;

    int ghost = 2;

    vecu u(nx + 2 * ghost, ny + 2 * ghost);
    vecu uBar(nx + 2 * ghost, ny + 2 * ghost);
    vecu uPlus(nx + 2 * ghost, ny + 2 * ghost);

    vecu flux_x(nx + 2 * ghost, ny + 2 * ghost);
    vecu flux_y(nx + 2 * ghost, ny + 2 * ghost);


    if(problem == "quadrant"){
        quadrant(u, x0, y0, dx, dy);
    }
    else if (problem == "bubble"){
        bubble(u, x0, y0, dx, dy);
    }
    
    auto algo_start_time = std::chrono::high_resolution_clock::now();

    double t = tStart;
    while (t < tEnd) {
        time << t << ",";

        ubound(u.rho, ghost);
        ubound(u.momx, ghost);
        ubound(u.momy, ghost);
        ubound(u.ene, ghost);

        const double dt = C * (std::min(dx, dy) / wavespeed(u));
        t += dt;

        std::cout << "t: " << t << std::endl;

        flux_x = SLIC_x(u, dx, dt); // here

        for (int j = ghost; j < ny + ghost; j++) {
            for (int i = ghost; i < nx + ghost; i++) {
                uBar.rho[i][j]  = u.rho[i][j]  - (dt/dx) * (flux_x.rho[i][j]  - flux_x.rho[i-1][j]);
                uBar.momx[i][j] = u.momx[i][j] - (dt/dx) * (flux_x.momx[i][j] - flux_x.momx[i-1][j]);
                uBar.momy[i][j] = u.momy[i][j] - (dt/dx) * (flux_x.momy[i][j] - flux_x.momy[i-1][j]);
                uBar.ene[i][j]  = u.ene[i][j]  - (dt/dx) * (flux_x.ene[i][j]  - flux_x.ene[i-1][j]);
            }
        }

        ubound(uBar.rho, ghost);
        ubound(uBar.momx, ghost);
        ubound(uBar.momy, ghost);
        ubound(uBar.ene, ghost);

        flux_y = SLIC_y(uBar, dy, dt);

        for (int j = ghost; j < ny + ghost; j++) {
            for (int i = ghost; i < nx + ghost; i++) {
                uPlus.rho[i][j]  = uBar.rho[i][j]  - (dt/dy) * (flux_y.rho[i][j]  - flux_y.rho[i][j-1]);
                uPlus.momx[i][j] = uBar.momx[i][j] - (dt/dy) * (flux_y.momx[i][j] - flux_y.momx[i][j-1]);
                uPlus.momy[i][j] = uBar.momy[i][j] - (dt/dy) * (flux_y.momy[i][j] - flux_y.momy[i][j-1]);
                uPlus.ene[i][j]  = uBar.ene[i][j]  - (dt/dy) * (flux_y.ene[i][j]  - flux_y.ene[i][j-1]);
            }
        }

        ubound(uPlus.rho, ghost);
        ubound(uPlus.momx, ghost);
        ubound(uPlus.momy, ghost);
        ubound(uPlus.ene, ghost);


        for (int j = ghost; j < ny + ghost; j++) {
            for (int i = ghost; i < nx + ghost; i++) {
                //density << u.rho[i][j] << ",";
                //velx << u.momx[i][j] / u.rho[i][j] << ",";
                //vely << u.momy[i][j] / u.rho[i][j]<< ",";
                pressure << compute_pre(u.ene[i][j], u.rho[i][j], u.momx[i][j], u.momy[i][j]) << ",";
            }
        }

        //density << std::endl;
        //velx << std::endl;
        //vely << std::endl;
        pressure << std::endl;

        u = uPlus;
    }

    auto algo_end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> algo_time = algo_end_time - algo_start_time;
    std::cout << problem + " main loop time: " << algo_time.count() << " ms" << std::endl;

    // manually running one timestep more to measure performances

    double dt = C * (std::min(dx, dy) / wavespeed(u));

    auto x_fluxes_start_time = std::chrono::high_resolution_clock::now();

    flux_x = SLIC_x(u, dx, dt);

    auto x_fluxes_end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> x_fluxes_time = x_fluxes_end_time - x_fluxes_start_time;
    std::cout << problem + " x_fluxes time: " << x_fluxes_time.count() << " ms" << std::endl;

    auto x_fv_start_time = std::chrono::high_resolution_clock::now();

    for (int j = ghost; j < ny + ghost; j++) {
        for (int i = ghost; i < nx + ghost; i++) {
            uBar.rho[i][j]  = u.rho[i][j]  - (dt/dx) * (flux_x.rho[i][j]  - flux_x.rho[i-1][j]);
            uBar.momx[i][j] = u.momx[i][j] - (dt/dx) * (flux_x.momx[i][j] - flux_x.momx[i-1][j]);
            uBar.momy[i][j] = u.momy[i][j] - (dt/dx) * (flux_x.momy[i][j] - flux_x.momy[i-1][j]);
            uBar.ene[i][j]  = u.ene[i][j]  - (dt/dx) * (flux_x.ene[i][j]  - flux_x.ene[i-1][j]);
        }
    }

    auto x_fv_end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> x_fv_time = x_fv_end_time - x_fv_start_time;
    std::cout << problem + " x_fv time: " << x_fv_time.count() << " ms" << std::endl;

    auto y_fluxes_start_time = std::chrono::high_resolution_clock::now();

    flux_y = SLIC_y(uBar, dy, dt);

    auto y_fluxes_end_time = std::chrono::high_resolution_clock::now();


    std::chrono::duration<double, std::milli> y_fluxes_time = y_fluxes_end_time - y_fluxes_start_time;
    std::cout << problem + " y_fluxes time: " << y_fluxes_time.count() << " ms" << std::endl;

    auto y_fv_start_time = std::chrono::high_resolution_clock::now();

    for (int j = ghost; j < ny + ghost; j++) {
        for (int i = ghost; i < nx + ghost; i++) {
            uPlus.rho[i][j]  = uBar.rho[i][j]  - (dt/dy) * (flux_y.rho[i][j]  - flux_y.rho[i][j-1]);
            uPlus.momx[i][j] = uBar.momx[i][j] - (dt/dy) * (flux_y.momx[i][j] - flux_y.momx[i][j-1]);
            uPlus.momy[i][j] = uBar.momy[i][j] - (dt/dy) * (flux_y.momy[i][j] - flux_y.momy[i][j-1]);
            uPlus.ene[i][j]  = uBar.ene[i][j]  - (dt/dy) * (flux_y.ene[i][j]  - flux_y.ene[i][j-1]);
        }
    }

    auto y_fv_end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> y_fv_time = y_fv_end_time - y_fv_start_time;
    std::cout << problem + " y_fv time: " << y_fv_time.count() << " ms" << std::endl;

    return 0;
}
