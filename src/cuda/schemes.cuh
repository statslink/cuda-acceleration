#ifndef SCHEMES_CUH
#define SCHEMES_CUH

#include "struct.cuh"

#define HALF 0.5

__host__ __device__ inline double den_flux_x(const vecu& u, const int flat) {
    const double velocity = u.momx[flat] / u.rho[flat];
    return u.rho[flat] * velocity;
}

__host__ __device__ inline double momx_flux_x(const vecu u, const int flat) {
    double velocity = u.momx[flat] / u.rho[flat];
    double pressure = compute_pre(u.ene[flat], u.rho[flat], u.momx[flat], u.momy[flat]);
    return u.momx[flat] * velocity + pressure;
}

__host__ __device__ inline double momy_flux_x(const vecu u, const int flat) {
    double velocity = u.momx[flat] / u.rho[flat];
    return u.momy[flat] * velocity;
}

__host__ __device__ inline double ene_flux_x(const vecu u, const int flat) {
    double velocity = u.momx[flat] / u.rho[flat];
    double pressure = compute_pre(u.ene[flat], u.rho[flat], u.momx[flat], u.momy[flat]);
    return (u.ene[flat] + pressure) * velocity;
}


__host__ __device__ inline double den_flux_y(const vecu u, const int flat) {
    double velocity = u.momy[flat] / u.rho[flat];
    return u.rho[flat] * velocity;
}

__host__ __device__ inline double momx_flux_y(const vecu u, const int flat) {
    double velocity = u.momy[flat] / u.rho[flat];
    return u.momx[flat] * velocity;
}

__host__ __device__ inline double momy_flux_y(const vecu u, const int flat) {
    double velocity = u.momy[flat] / u.rho[flat];
    double pressure = compute_pre(u.ene[flat], u.rho[flat], u.momx[flat], u.momy[flat]);
    return u.momy[flat] * velocity + pressure;
}

__host__ __device__ inline double ene_flux_y(const vecu u, const int flat) {
    double velocity = u.momy[flat] / u.rho[flat];
    double pressure = compute_pre(u.ene[flat], u.rho[flat], u.momx[flat], u.momy[flat]);
    return (u.ene[flat] + pressure) * velocity;
}


__host__ __device__ inline double left_delta_x(double* u, const long flat, const long flat_im) {
    const double delta_m = u[flat] - u[flat_im];
    return delta_m;
}

__host__ __device__ inline double right_delta_x(double* u, const long flat, const long flat_ip) {
        const double delta_p = u[flat_ip] - u[flat];
    return delta_p;
}

__host__ __device__ inline double left_delta_y(double* u, const long flat, const long flat_jm) {
    const double delta_m = u[flat] - u[flat_jm];
    return delta_m;
}

__host__ __device__ inline double right_delta_y(double* u, const long flat, const long flat_jp) {
    const double delta_p = u[flat_jp] - u[flat];
    return delta_p;
}

__host__ __device__ inline double slope_measure(const double left, const double right, const double omega = 0) {
    const double delta = HALF * (1 + omega) * left + HALF * (1 - omega) * right;

    return delta;
}

__host__ __device__ inline double xi_minbee(const double r) {
    if (r <= 0) {
        return 0;
    }

    if (r > 0 && r <=1) {
        return r;
    }

    if (r > 1) {
        return fmin(1.0, 2 / (1 + r));
    }

    return 0;
}

__global__ void ubound(double* vec, int nx, int ny, int ghost) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < nx && j < ny) {
        if (i < ghost){
            vec[FLATTEN(i,j,nx)] = vec[FLATTEN(ghost, j, nx)];
        }
           
        if (i >= nx - ghost){
            vec[FLATTEN(i,j,nx)] = vec[FLATTEN(nx - ghost - 1, j, nx)];
        }
          
        if (j < ghost){
            vec[FLATTEN(i,j,nx)] = vec[FLATTEN(i, ghost, nx)];
        }
           
        if (j >= ny - ghost){
            vec[FLATTEN(i,j,nx)] = vec[FLATTEN(i, ny - ghost - 1, nx)];
        }
    }
}

__global__ void fv_x(const vecu u, const vecu flux, vecu uBar, double dx, double dt, int nx, int ny, int ghost) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + ghost;
    int j = blockIdx.y * blockDim.y + threadIdx.y + ghost;

    if(i < nx - ghost && j < ny - ghost) {
        int flat = FLATTEN(i,j,nx);
        int flat_im = FLATTEN(i-1,j,nx);

        uBar.rho[flat]  = u.rho[flat]  - (dt/dx) * (flux.rho[flat]  - flux.rho[flat_im]);
        uBar.momx[flat] = u.momx[flat] - (dt/dx) * (flux.momx[flat] - flux.momx[flat_im]);
        uBar.momy[flat] = u.momy[flat] - (dt/dx) * (flux.momy[flat] - flux.momy[flat_im]);
        uBar.ene[flat]  = u.ene[flat]  - (dt/dx) * (flux.ene[flat]  - flux.ene[flat_im]);
    }
}

__global__ void fv_y(const vecu uBar, const vecu flux, vecu uPlus, double dy, double dt, int nx, int ny, int ghost) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + ghost;
    int j = blockIdx.y * blockDim.y + threadIdx.y + ghost;

    if(i < nx - ghost && j < ny - ghost) {
        int flat = FLATTEN(i,j,nx);
        int flat_jm = FLATTEN(i,j-1,nx);

        uPlus.rho[flat]  = uBar.rho[flat]  - (dt/dy) * (flux.rho[flat]  - flux.rho[flat_jm]);
        uPlus.momx[flat] = uBar.momx[flat] - (dt/dy) * (flux.momx[flat] - flux.momx[flat_jm]);
        uPlus.momy[flat] = uBar.momy[flat] - (dt/dy) * (flux.momy[flat] - flux.momy[flat_jm]);
        uPlus.ene[flat]  = uBar.ene[flat]  - (dt/dy) * (flux.ene[flat]  - flux.ene[flat_jm]);
    }
}


__global__ void FORCE_x(const vecu uL, const vecu uR, vecu flux, double dx, double dt, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if(i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        int flat    = FLATTEN(i, j, nx);
        int flat_ip = FLATTEN(i + 1, j, nx);

        // LF
        double lf_rho  = HALF * (dx / dt) * (uR.rho[flat] - uL.rho[flat_ip]) + HALF * (den_flux_x(uR, flat) + den_flux_x(uL, flat_ip));
        double lf_momx = HALF * (dx / dt) * (uR.momx[flat] - uL.momx[flat_ip]) + HALF * (momx_flux_x(uR, flat) + momx_flux_x(uL, flat_ip));
        double lf_momy = HALF * (dx / dt) * (uR.momy[flat] - uL.momy[flat_ip]) + HALF * (momy_flux_x(uR, flat) + momy_flux_x(uL, flat_ip));
        double lf_ene  = HALF * (dx / dt) * (uR.ene[flat] - uL.ene[flat_ip]) + HALF * (ene_flux_x(uR, flat) + ene_flux_x(uL, flat_ip));

        // RI
        double rho_half = HALF * (uR.rho[flat] + uL.rho[flat_ip]) - HALF * (dt/dx) * (den_flux_x(uL, flat_ip) - den_flux_x(uR, flat));
        double momx_half = HALF * (uR.momx[flat] + uL.momx[flat_ip]) - HALF * (dt/dx) * (momx_flux_x(uL, flat_ip) - momx_flux_x(uR, flat));
        double momy_half = HALF * (uR.momy[flat] + uL.momy[flat_ip]) - HALF * (dt/dx) * (momy_flux_x(uL, flat_ip) - momy_flux_x(uR, flat));
        double ene_half  = HALF * (uR.ene[flat] + uL.ene[flat_ip]) - HALF * (dt/dx) * (ene_flux_x(uL, flat_ip) - ene_flux_x(uR, flat));

        double pre_half = (GAMMA - 1.0) * (ene_half - (momx_half * momx_half + momy_half * momy_half) / (2.0 * rho_half));
        double vel_half = momx_half / rho_half;
        
        double ri_rho  = rho_half * vel_half;
        double ri_momx = momx_half * vel_half + pre_half;
        double ri_momy = momy_half * vel_half;
        double ri_ene  = (ene_half + pre_half) * vel_half;

        // FORCE
        flux.rho[flat]  = HALF * (lf_rho  + ri_rho);
        flux.momx[flat] = HALF * (lf_momx + ri_momx);
        flux.momy[flat] = HALF * (lf_momy + ri_momy);
        flux.ene[flat]  = HALF * (lf_ene  + ri_ene);
    }
}

__global__ void FORCE_y(const vecu uL, const vecu uR, vecu flux, double dx, double dt, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if(i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        int flat = FLATTEN(i, j, nx);
        int flat_jp = FLATTEN(i, j + 1, nx);

        // LF
        double lf_rho  = HALF * (dx / dt) * (uR.rho[flat] - uL.rho[flat_jp]) + HALF * (den_flux_y(uR, flat) + den_flux_y(uL, flat_jp));
        double lf_momx = HALF * (dx / dt) * (uR.momx[flat] - uL.momx[flat_jp]) + HALF * (momx_flux_y(uR, flat) + momx_flux_y(uL, flat_jp));
        double lf_momy = HALF * (dx / dt) * (uR.momy[flat] - uL.momy[flat_jp]) + HALF * (momy_flux_y(uR, flat) + momy_flux_y(uL, flat_jp));
        double lf_ene  = HALF * (dx / dt) * (uR.ene[flat] - uL.ene[flat_jp]) + HALF * (ene_flux_y(uR, flat) + ene_flux_y(uL, flat_jp));

        // RI
        double rho_half  = HALF * (uR.rho[flat] + uL.rho[flat_jp]) - HALF * (dt/dx) * (den_flux_y(uL, flat_jp) - den_flux_y(uR, flat));
        double momx_half = HALF * (uR.momx[flat] + uL.momx[flat_jp]) - HALF * (dt/dx) * (momx_flux_y(uL, flat_jp) - momx_flux_y(uR, flat));
        double momy_half = HALF * (uR.momy[flat] + uL.momy[flat_jp]) - HALF * (dt/dx)* (momy_flux_y(uL, flat_jp) - momy_flux_y(uR, flat));
        double ene_half  = HALF * (uR.ene[flat] + uL.ene[flat_jp]) - HALF * (dt/dx) * (ene_flux_y(uL, flat_jp) - ene_flux_y(uR, flat));


        double pre_half = (GAMMA - 1.0) * (ene_half - (momx_half * momx_half + momy_half * momy_half) / (2.0 * rho_half));
        double vel_half = uR.momy[flat] / uR.rho[flat];
        
        double ri_rho  = rho_half * vel_half;
        double ri_momx = momx_half * vel_half;
        double ri_momy = momy_half * vel_half + pre_half;
        double ri_ene  = (ene_half + pre_half) * vel_half;

        // FORCE
        flux.rho[flat]  = HALF * (lf_rho  + ri_rho);
        flux.momx[flat] = HALF * (lf_momx + ri_momx);
        flux.momy[flat] = HALF * (lf_momy + ri_momy);
        flux.ene[flat]  = HALF * (lf_ene  + ri_ene);
    }
}

__global__ void SLIC_x_reconstruciton(vecu u, vecu uL_bar, vecu uR_bar, double dx, double dt, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; 
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if(i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        int flat = FLATTEN(i, j, nx);
        int flat_ip = FLATTEN(i + 1, j, nx);
        int flat_im = FLATTEN(i - 1, j, nx);

        double den_slope = slope_measure(left_delta_x(u.rho, flat, flat_im), right_delta_x(u.rho, flat, flat_ip));
        double momx_slope = slope_measure(left_delta_x(u.momx, flat, flat_im), right_delta_x(u.momx, flat, flat_ip));
        double momy_slope = slope_measure(left_delta_x(u.momy, flat, flat_im), right_delta_x(u.momy, flat, flat_ip));
        double ene_slope = slope_measure(left_delta_x(u.ene, flat, flat_im), right_delta_x(u.ene, flat, flat_ip));

        double r_rho = left_delta_x(u.rho, flat, flat_im) / right_delta_x(u.rho, flat, flat_ip);
        double r_momx = left_delta_x(u.momx, flat, flat_im) / right_delta_x(u.momx, flat, flat_ip);
        double r_momy  = left_delta_x(u.momy, flat, flat_im) / right_delta_x(u.momy, flat, flat_ip);
        double r_ene = left_delta_x(u.ene, flat, flat_im) / right_delta_x(u.ene, flat, flat_ip);

        uL_bar.rho[flat] = u.rho[flat] - HALF * xi_minbee(r_rho) * den_slope;
        uR_bar.rho[flat] = u.rho[flat] + HALF * xi_minbee(r_rho) * den_slope;

        uL_bar.momx[flat] = u.momx[flat] - HALF * xi_minbee(r_momx) * momx_slope;
        uR_bar.momx[flat] = u.momx[flat] + HALF * xi_minbee(r_momx) * momx_slope;

        uL_bar.momy[flat] = u.momy[flat] - HALF * xi_minbee(r_momy) * momy_slope;
        uR_bar.momy[flat] = u.momy[flat] + HALF * xi_minbee(r_momy) * momy_slope;

        uL_bar.ene[flat] = u.ene[flat] - HALF * xi_minbee(r_ene) * ene_slope;
        uR_bar.ene[flat] = u.ene[flat] + HALF * xi_minbee(r_ene) * ene_slope;
    }
}

__global__ void SLIC_x_halfstep(vecu uL_bar, vecu uR_bar, vecu uL_bar_half, vecu uR_bar_half, double dx, double dt, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if(i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        int flat = FLATTEN(i, j, nx);

        uL_bar_half.rho[flat] = uL_bar.rho[flat] - HALF * (dt / dx) * (den_flux_x(uR_bar, flat) - den_flux_x(uL_bar, flat));
        uR_bar_half.rho[flat] = uR_bar.rho[flat] - HALF * (dt / dx) * (den_flux_x(uR_bar, flat) - den_flux_x(uL_bar, flat));

        uL_bar_half.momx[flat] = uL_bar.momx[flat] - HALF * (dt / dx) * (momx_flux_x(uR_bar, flat) - momx_flux_x(uL_bar, flat));
        uR_bar_half.momx[flat] = uR_bar.momx[flat] - HALF * (dt / dx) * (momx_flux_x(uR_bar, flat) - momx_flux_x(uL_bar, flat));

        uL_bar_half.momy[flat] = uL_bar.momy[flat] - HALF * (dt / dx) * (momy_flux_x(uR_bar, flat) - momy_flux_x(uL_bar, flat));
        uR_bar_half.momy[flat] = uR_bar.momy[flat] - HALF * (dt / dx) * (momy_flux_x(uR_bar, flat) - momy_flux_x(uL_bar, flat));

        uL_bar_half.ene[flat] = uL_bar.ene[flat] - HALF * (dt / dx) * (ene_flux_x(uR_bar, flat) - ene_flux_x(uL_bar, flat));
        uR_bar_half.ene[flat] = uR_bar.ene[flat] - HALF * (dt / dx) * (ene_flux_x(uR_bar, flat) - ene_flux_x(uL_bar, flat));
    }
}

__global__ void SLIC_y_reconstruction(vecu u, vecu uL_bar, vecu uR_bar, double dx, double dt, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if(i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        int flat = FLATTEN(i, j, nx);

        int flat_jp = FLATTEN(i, j + 1, nx);
        int flat_jm = FLATTEN(i, j - 1, nx);

        double den_slope = slope_measure(left_delta_y(u.rho, flat, flat_jm), right_delta_y(u.rho, flat, flat_jp));
        double momx_slope = slope_measure(left_delta_y(u.momx, flat, flat_jm), right_delta_y(u.momx, flat, flat_jp));
        double momy_slope = slope_measure(left_delta_y(u.momy, flat, flat_jm), right_delta_y(u.momy, flat, flat_jp));
        double ene_slope = slope_measure(left_delta_y(u.ene, flat, flat_jm), right_delta_y(u.ene, flat, flat_jp));

        double r_rho = left_delta_y(u.rho, flat, flat_jm) / right_delta_y(u.rho, flat, flat_jp);
        double r_momx = left_delta_y(u.momx, flat, flat_jm) / right_delta_y(u.momx, flat, flat_jp);
        double r_momy  = left_delta_y(u.momy, flat, flat_jm) / right_delta_y(u.momy, flat, flat_jp);
        double r_ene = left_delta_y(u.ene, flat, flat_jm) / right_delta_y(u.ene, flat, flat_jp);

        uL_bar.rho[flat] = u.rho[flat] - HALF * xi_minbee(r_rho) * den_slope;
        uR_bar.rho[flat] = u.rho[flat] + HALF * xi_minbee(r_rho) * den_slope;

        uL_bar.momx[flat] = u.momx[flat] - HALF * xi_minbee(r_momx) * momx_slope;
        uR_bar.momx[flat] = u.momx[flat] + HALF * xi_minbee(r_momx) * momx_slope;

        uL_bar.momy[flat] = u.momy[flat] - HALF * xi_minbee(r_momy) * momy_slope;
        uR_bar.momy[flat] = u.momy[flat] + HALF * xi_minbee(r_momy) * momy_slope;

        uL_bar.ene[flat] = u.ene[flat] - HALF * xi_minbee(r_ene) * ene_slope;
        uR_bar.ene[flat] = u.ene[flat] + HALF * xi_minbee(r_ene) * ene_slope;

    }
}

__global__ void SLIC_y_halfstep(vecu uL_bar, vecu uR_bar, vecu uL_bar_half, vecu uR_bar_half, double dx, double dt, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if(i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        int flat = FLATTEN(i, j, nx);

        uL_bar_half.rho[flat] = uL_bar.rho[flat] - HALF * (dt / dx) * (den_flux_y(uR_bar, flat) - den_flux_y(uL_bar, flat));
        uR_bar_half.rho[flat] = uR_bar.rho[flat] - HALF * (dt / dx) * (den_flux_y(uR_bar, flat) - den_flux_y(uL_bar, flat));

        uL_bar_half.momx[flat] = uL_bar.momx[flat] - HALF * (dt / dx) * (momx_flux_y(uR_bar, flat) - momx_flux_y(uL_bar, flat));
        uR_bar_half.momx[flat] = uR_bar.momx[flat] - HALF * (dt / dx) * (momx_flux_y(uR_bar, flat) - momx_flux_y(uL_bar, flat));

        uL_bar_half.momy[flat] = uL_bar.momy[flat] - HALF * (dt / dx) * (momy_flux_y(uR_bar, flat) - momy_flux_y(uL_bar, flat));
        uR_bar_half.momy[flat] = uR_bar.momy[flat] - HALF * (dt / dx) * (momy_flux_y(uR_bar, flat) - momy_flux_y(uL_bar, flat));

        uL_bar_half.ene[flat] = uL_bar.ene[flat] - HALF * (dt / dx) * (ene_flux_y(uR_bar, flat) - ene_flux_y(uL_bar, flat));
        uR_bar_half.ene[flat] = uR_bar.ene[flat] - HALF * (dt / dx) * (ene_flux_y(uR_bar, flat) - ene_flux_y(uL_bar, flat));
    }
}

#endif