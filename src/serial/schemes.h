#ifndef SCHEMES_H
#define SCHEMES_H

#define HALF 0.5

#include "struct.h"

inline double den_flux_x(const vecu& u, const int i, const int j) {
    const double velocity = u.momx[i][j] / u.rho[i][j];
    return u.rho[i][j] * velocity;
}

inline double momx_flux_x(const vecu& u, const int i, const int j) {
    const double velocity = u.momx[i][j] / u.rho[i][j];
    const double pressure = compute_pre(u.ene[i][j], u.rho[i][j], u.momx[i][j], u.momy[i][j]);
    return u.momx[i][j] * velocity + pressure;
}

inline double momy_flux_x(const vecu& u, const int i, const int j) {
    const double velocity = u.momy[i][j] / u.rho[i][j];
    return u.momx[i][j] * velocity;
}

inline double ene_flux_x(const vecu& u, const int i, const int j) {
    const double velocity = u.momx[i][j] / u.rho[i][j] ;
    const double pressure = compute_pre(u.ene[i][j], u.rho[i][j], u.momx[i][j], u.momy[i][j]);
    return (u.ene[i][j] + pressure) * velocity;
}

inline double den_flux_y(const vecu& u, const int i, const int j) {
    const double velocity = u.momy[i][j] / u.rho[i][j];
    return u.rho[i][j] * velocity;
}

inline double momx_flux_y(const vecu& u, const int i, const int j) {
    const double velocity = u.momy[i][j] / u.rho[i][j];
    return u.momx[i][j] * velocity;
}

inline double momy_flux_y(const vecu& u, const int i, const int j) {
    const double velocity = u.momy[i][j] / u.rho[i][j];
    const double pressure = compute_pre(u.ene[i][j], u.rho[i][j], u.momx[i][j], u.momy[i][j]);
    return u.momy[i][j] * velocity + pressure;
}

inline double ene_flux_y(const vecu& u, const int i, const int j) {
    const double velocity = u.momy[i][j] / u.rho[i][j];
    const double pressure = compute_pre(u.ene[i][j], u.rho[i][j], u.momx[i][j], u.momy[i][j]);
    return (u.ene[i][j] + pressure) * velocity;
}


inline double slope_measure(const double left, const double right, const double omega = 0) { //
    return HALF * (1 + omega) * left + HALF * (1 - omega) * right;;
}

inline double xi_minbee(const double r) {
    if (r <= 0) {
        return 0;
    }

    if (r > 0 && r <=1) {
        return r;
    }

    if (r > 1) {
        return std::min(1.0, 2 / (1 + r));
    }

    return 0;
}

inline double left_delta_x(const std::vector<std::vector<double>> &u, const long i, const long j) {
    const double delta_m = u[i][j] - u[i - 1][j];
    return delta_m;
}

inline double right_delta_x(const std::vector<std::vector<double>> &u, const long i, const long j) {
    const double delta_p = u[i + 1][j] - u[i][j];
    return delta_p;
}

inline double left_delta_y(const std::vector<std::vector<double>> &u, const long i, const long j) {
    const double delta_m = u[i][j] - u[i][j - 1];
    return delta_m;
}

inline double right_delta_y(const std::vector<std::vector<double>> &u, const long i, const long j) {
    const double delta_p = u[i][j + 1] - u[i][j];
    return delta_p;
}

inline void ubound(std::vector<std::vector<double>> &vec, const int ghost) {
    const int nx = vec.size();
    const int ny = vec[0].size();

    for (int i = 0; i < ghost; i++)
        for (int j = 0; j < ny; j++)
            vec[i][j] = vec[ghost][j];

    for (int i = nx - ghost; i < nx; i++)
        for (int j = 0; j < ny; j++)
            vec[i][j] = vec[nx - ghost - 1][j];

    for (int j = 0; j < ghost; j++)
        for (int i = 0; i < nx; i++)
            vec[i][j] = vec[i][ghost];

    for (int j = ny - ghost; j < ny; j++)
        for (int i = 0; i < nx; i++)
            vec[i][j] = vec[i][ny - ghost - 1];
}

inline vecu FORCE_x(const vecu& uL, const vecu& uR, const double dx, const double dt) {
    const int nx = uL.rho.size();
    const int ny = uL.rho[0].size();

    vecu flux(nx, ny);

    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {

            // LF
            const double lf_rho = HALF * (dx / dt) * (uR.rho[i][j] - uL.rho[i+1][j]) + HALF * (den_flux_x(uR, i, j) + den_flux_x(uL, i+1, j));
            const double lf_momx = HALF * (dx / dt) * (uR.momx[i][j] - uL.momx[i+1][j]) + HALF * (momx_flux_x(uR, i, j) + momx_flux_x(uL, i+1, j));
            const double lf_momy = HALF * (dx / dt) * (uR.momy[i][j] - uL.momy[i+1][j]) + HALF * (momy_flux_x(uR, i, j) + momy_flux_x(uL, i+1, j));
            const double lf_ene = HALF * (dx / dt) * (uR.ene[i][j] - uL.ene[i+1][j]) + HALF * (ene_flux_x(uR, i, j) + ene_flux_x(uL, i+1, j));

            // RI
            const double rho_half = HALF * (uR.rho[i][j] + uL.rho[i+1][j]) - HALF * (dt/dx) * (den_flux_x(uL, i+1, j) - den_flux_x(uR, i, j));
            const double momx_half = HALF * (uR.momx[i][j] + uL.momx[i+1][j]) - HALF * (dt/dx) * (momx_flux_x(uL, i+1, j) - momx_flux_x(uR, i, j));
            const double momy_half = HALF * (uR.momy[i][j] + uL.momy[i+1][j]) - HALF * (dt/dx) * (momy_flux_x(uL, i+1, j) - momy_flux_x(uR, i, j));
            const double ene_half = HALF * (uR.ene[i][j] + uL.ene[i+1][j]) - HALF * (dt/dx) * (ene_flux_x(uL, i+1, j) - ene_flux_x(uR, i, j));

            const double pre_half = compute_pre(ene_half, rho_half, momx_half, momy_half);
            const double vel_half = momx_half / rho_half;

            const double ri_rho = rho_half * vel_half;
            const double ri_momx = momx_half * vel_half + pre_half;
            const double ri_momy = momy_half * vel_half;
            const double ri_ene = (ene_half + pre_half) * vel_half;

            // FORCE
            flux.rho[i][j]  = HALF * (lf_rho  + ri_rho);
            flux.momx[i][j] = HALF * (lf_momx + ri_momx);
            flux.momy[i][j] = HALF * (lf_momy + ri_momy);
            flux.ene[i][j]  = HALF * (lf_ene  + ri_ene);
        }
    }
    return flux;
}


inline vecu FORCE_y(const vecu& uL, const vecu& uR, const double dy, const double dt) {
    const int nx = uL.rho.size();
    const int ny = uL.rho[0].size();

    vecu flux(nx, ny);

    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {

            // LF
            const double lf_rho = HALF * (dy / dt) * (uR.rho[i][j] - uL.rho[i][j+1]) + HALF * (den_flux_y(uR, i, j) + den_flux_y(uL, i, j+1));
            const double lf_momx = HALF * (dy / dt) * (uR.momx[i][j] - uL.momx[i][j+1]) + HALF * (momx_flux_y(uR, i, j) + momx_flux_y(uL, i, j+1));
            const double lf_momy = HALF * (dy / dt) * (uR.momy[i][j] - uL.momy[i][j+1]) + HALF * (momy_flux_y(uR, i, j) + momy_flux_y(uL, i, j+1));
            const double lf_ene = HALF * (dy / dt) * (uR.ene[i][j] - uL.ene[i][j+1]) + HALF * (ene_flux_y(uR, i, j) + ene_flux_y(uL, i, j+1));

            // RI
            const double rho_half = HALF * (uR.rho[i][j] + uL.rho[i][j+1]) - HALF * (dt/dy) * (den_flux_y(uL, i, j+1) - den_flux_y(uR, i, j));
            const double momx_half = HALF * (uR.momx[i][j] + uL.momx[i][j+1]) - HALF * (dt/dy) * (momx_flux_y(uL, i, j+1) - momx_flux_y(uR, i, j));
            const double momy_half = HALF * (uR.momy[i][j] + uL.momy[i][j+1]) - HALF * (dt/dy) * (momy_flux_y(uL, i, j+1) - momy_flux_y(uR, i, j));
            const double ene_half = HALF * (uR.ene[i][j] + uL.ene[i][j+1]) - HALF * (dt/dy) * (ene_flux_y(uL, i, j+1) - ene_flux_y(uR, i, j));

            const double pre_half = compute_pre(ene_half, rho_half, momx_half, momy_half);
            const double vel_half = momy_half / rho_half;

            const double ri_rho = rho_half * vel_half;
            const double ri_momx = momx_half * vel_half;
            const double ri_momy = momy_half * vel_half + pre_half;
            const double ri_ene = (ene_half + pre_half) * vel_half;

            // FORCE
            flux.rho[i][j]  = HALF * (lf_rho  + ri_rho);
            flux.momx[i][j] = HALF * (lf_momx + ri_momx);
            flux.momy[i][j] = HALF * (lf_momy + ri_momy);
            flux.ene[i][j]  = HALF * (lf_ene  + ri_ene);
        }
    }
    return flux;
}


inline vecu SLIC_x(const vecu& u, const double dx, const double dt) {
    const int nx = u.rho.size();
    const int ny = u.rho[0].size();

    vecu flux(nx, ny);
    vecu uL_bar(nx, ny), uR_bar(nx, ny);

    vecu uL_bar_half(nx, ny), uR_bar_half(nx, ny);

    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
            double den_slope = slope_measure(left_delta_x(u.rho, i, j), right_delta_x(u.rho, i, j));
            double momx_slope = slope_measure(left_delta_x(u.momx, i, j), right_delta_x(u.momx, i, j));
            double momy_slope = slope_measure(left_delta_x(u.momy, i, j), right_delta_x(u.momy, i, j));
            double ene_slope = slope_measure(left_delta_x(u.ene, i, j), right_delta_x(u.ene, i, j));

            double r_rho = left_delta_x(u.rho, i, j) / right_delta_x(u.rho, i, j);
            double r_momx = left_delta_x(u.momx, i, j) / right_delta_x(u.momx, i, j);
            double r_momy  = left_delta_x(u.momy, i, j) / right_delta_x(u.momy, i, j);
            double r_ene = left_delta_x(u.ene, i, j) / right_delta_x(u.ene, i, j);

            uL_bar.rho[i][j] = u.rho[i][j] - HALF * xi_minbee(r_rho) * den_slope;
            uR_bar.rho[i][j] = u.rho[i][j] + HALF * xi_minbee(r_rho) * den_slope;

            uL_bar.momx[i][j] = u.momx[i][j] - HALF * xi_minbee(r_momx) * momx_slope;
            uR_bar.momx[i][j] = u.momx[i][j] + HALF * xi_minbee(r_momx) * momx_slope;

            uL_bar.momy[i][j] = u.momy[i][j] - HALF * xi_minbee(r_momy) * momy_slope;
            uR_bar.momy[i][j] = u.momy[i][j] + HALF * xi_minbee(r_momy) * momy_slope;

            uL_bar.ene[i][j] = u.ene[i][j] - HALF * xi_minbee(r_ene) * ene_slope;
            uR_bar.ene[i][j] = u.ene[i][j] + HALF * xi_minbee(r_ene) * ene_slope;
        }
    }

    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
            uL_bar_half.rho[i][j] = uL_bar.rho[i][j] - HALF * (dt / dx) * (den_flux_x(uR_bar, i, j) - den_flux_x(uL_bar, i, j));
            uR_bar_half.rho[i][j] = uR_bar.rho[i][j] - HALF * (dt / dx) * (den_flux_x(uR_bar, i, j) - den_flux_x(uL_bar, i, j));

            uL_bar_half.momx[i][j] = uL_bar.momx[i][j] - HALF * (dt / dx) * (momx_flux_x(uR_bar, i, j) - momx_flux_x(uL_bar, i, j));
            uR_bar_half.momx[i][j] = uR_bar.momx[i][j] - HALF * (dt / dx) * (momx_flux_x(uR_bar, i, j) - momx_flux_x(uL_bar, i, j));

            uL_bar_half.momy[i][j] = uL_bar.momy[i][j] - HALF * (dt / dx) * (momy_flux_x(uR_bar, i, j) - momy_flux_x(uL_bar, i, j));
            uR_bar_half.momy[i][j] = uR_bar.momy[i][j] - HALF * (dt / dx) * (momy_flux_x(uR_bar, i, j) - momy_flux_x(uL_bar, i, j));

            uL_bar_half.ene[i][j] = uL_bar.ene[i][j] - HALF * (dt / dx) * (ene_flux_x(uR_bar, i, j) - ene_flux_x(uL_bar, i, j));
            uR_bar_half.ene[i][j] = uR_bar.ene[i][j] - HALF * (dt / dx) * (ene_flux_x(uR_bar, i, j) - ene_flux_x(uL_bar, i, j));
        }
    }

    flux = FORCE_x(uL_bar_half, uR_bar_half, dx, dt);

    return flux;
}


inline vecu SLIC_y(const vecu& u, const double dy, const double dt) {
    const int nx = u.rho.size();
    const int ny = u.rho[0].size();

    vecu flux(nx, ny);
    vecu uL_bar(nx, ny), uR_bar(nx, ny);

    vecu uL_bar_half(nx, ny), uR_bar_half(nx, ny);

    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
            double den_slope = slope_measure(left_delta_y(u.rho, i, j), right_delta_y(u.rho, i, j));
            double momx_slope = slope_measure(left_delta_y(u.momx, i, j), right_delta_y(u.momx, i, j));
            double momy_slope = slope_measure(left_delta_y(u.momy, i, j), right_delta_y(u.momy, i, j));
            double ene_slope = slope_measure(left_delta_y(u.ene, i, j), right_delta_y(u.ene, i, j));

            double r_rho = left_delta_y(u.rho, i, j) / right_delta_y(u.rho, i, j);
            double r_momx = left_delta_y(u.momx, i, j) / right_delta_y(u.momx, i, j);
            double r_momy  = left_delta_y(u.momy, i, j) / right_delta_y(u.momy, i, j);
            double r_ene = left_delta_y(u.ene, i, j) / right_delta_y(u.ene, i, j);


            uL_bar.rho[i][j] = u.rho[i][j] - HALF * xi_minbee(r_rho) * den_slope;
            uR_bar.rho[i][j] = u.rho[i][j] + HALF * xi_minbee(r_rho) * den_slope;

            uL_bar.momx[i][j] = u.momx[i][j] - HALF * xi_minbee(r_momx) * momx_slope;
            uR_bar.momx[i][j] = u.momx[i][j] + HALF * xi_minbee(r_momx) * momx_slope;

            uL_bar.momy[i][j] = u.momy[i][j] - HALF * xi_minbee(r_momy) * momy_slope;
            uR_bar.momy[i][j] = u.momy[i][j] + HALF * xi_minbee(r_momy) * momy_slope;

            uL_bar.ene[i][j] = u.ene[i][j] - HALF * xi_minbee(r_ene) * ene_slope;
            uR_bar.ene[i][j] = u.ene[i][j] + HALF * xi_minbee(r_ene) * ene_slope;
        }
    }

    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
            uL_bar_half.rho[i][j] = uL_bar.rho[i][j] - HALF * (dt / dy) * (den_flux_y(uR_bar, i, j) - den_flux_y(uL_bar, i, j));
            uR_bar_half.rho[i][j] = uR_bar.rho[i][j] - HALF * (dt / dy) * (den_flux_y(uR_bar, i, j) - den_flux_y(uL_bar, i, j));

            uL_bar_half.momx[i][j] = uL_bar.momx[i][j] - HALF * (dt / dy) * (momx_flux_y(uR_bar, i, j) - momx_flux_y(uL_bar, i, j));
            uR_bar_half.momx[i][j] = uR_bar.momx[i][j] - HALF * (dt / dy) * (momx_flux_y(uR_bar, i, j) - momx_flux_y(uL_bar, i, j));

            uL_bar_half.momy[i][j] = uL_bar.momy[i][j] - HALF * (dt / dy) * (momy_flux_y(uR_bar, i, j) - momy_flux_y(uL_bar, i, j));
            uR_bar_half.momy[i][j] = uR_bar.momy[i][j] - HALF * (dt / dy) * (momy_flux_y(uR_bar, i, j) - momy_flux_y(uL_bar, i, j));

            uL_bar_half.ene[i][j] = uL_bar.ene[i][j] - HALF * (dt / dy) * (ene_flux_y(uR_bar, i, j) - ene_flux_y(uL_bar, i, j));
            uR_bar_half.ene[i][j] = uR_bar.ene[i][j] - HALF * (dt / dy) * (ene_flux_y(uR_bar, i, j) - ene_flux_y(uL_bar, i, j));
        }
    }

    flux = FORCE_y(uL_bar_half, uR_bar_half, dy, dt);

    return flux;
}

#endif
