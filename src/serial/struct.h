#ifndef STRUCT_H
#define STRUCT_H

#include <vector>

#define GAMMA 1.4

struct vecu {
    std::vector<std::vector<double>> rho, momx, momy, ene;
    explicit vecu(const int x_cells, const int y_cells) {
        rho.resize(x_cells, std::vector<double>(y_cells));
        momx.resize(x_cells, std::vector<double>(y_cells));
        momy.resize(x_cells, std::vector<double>(y_cells));
        ene.resize(x_cells, std::vector<double>(y_cells));
    }
};

inline double compute_pre(const double energy, const double density, const double momentum_x, const double momentum_y) {
    return (GAMMA - 1.0) * (energy - (momentum_x * momentum_x + momentum_y * momentum_y) / (2 * density));
}

inline double compute_ene(const double pressure, const double density, const double velocity_x, const double velocity_y) {
    const double momx = velocity_x * density;
    const double momy = velocity_y * density;
    return pressure / (GAMMA - 1.0) + (momx * momx + momy * momy) / (2 * density);
}


#endif
