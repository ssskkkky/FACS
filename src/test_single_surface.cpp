#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../include/equilibrium.h"
#include "../include/gFileRawData.h"
#include "../include/two_dof_continuum.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <gfile> <psi>\n";
        return 1;
    }

    std::string gfile_path = argv[1];
    double target_psi = std::stod(argv[2]);

    std::ifstream gfile(gfile_path);
    if (!gfile.is_open()) {
        std::cerr << "Cannot open g-file: " << gfile_path << "\n";
        return 1;
    }

    GFileRawData gfile_data;
    gfile >> gfile_data;
    gfile.close();

    if (!gfile_data.is_complete()) {
        std::cerr << "Failed to parse g-file\n";
        return 1;
    }

    std::cout << "Loaded g-file: " << gfile_path << "\n";
    std::cout << "Target psi: " << target_psi << "\n\n";

    std::size_t radial_grid = 1000;
    std::size_t poloidal_grid = 300;
    double psi_ratio = 0.96;

    NumericEquilibrium<double> eq(gfile_data, radial_grid, poloidal_grid,
                                  psi_ratio);
    auto psi_range = eq.psi_range();

    if (target_psi < psi_range.first || target_psi > psi_range.second) {
        std::cerr << "Psi " << target_psi << " is out of range ["
                  << psi_range.first << ", " << psi_range.second << "]\n";
        return 1;
    }

    double q = eq.safety_factor(target_psi);
    double r_minor = eq.minor_radius(target_psi);

    std::cout << "=== EQUILIBRIUM AT PSI = " << target_psi << " ===\n";
    std::cout << "Safety factor q: " << q << "\n";
    std::cout << "Minor radius r/a: " << r_minor << "\n\n";

    TwoDofParams params;
    params.eq = &eq;
    params.use_geo = true;
    params.psi = target_psi;
    params.beta_val = 0.206494;
    params.Gamma_val = 1.6667;

    bool withSoundGap = true;
    std::size_t steps = 1000;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Computing continuum for ω = 0.001 → 1.0\n\n";

    std::ofstream out("continuum_psi_" + std::to_string(target_psi) + ".csv");
    out << "omega,floquet_exp_sound_1,floquet_exp_sound_2,"
        << "floquet_exp_alfven_1,floquet_exp_alfven_2,"
        << "alfvenicity_sound_1,alfvenicity_sound_2,"
        << "alfvenicity_alfven_1,alfvenicity_alfven_2\n";

    std::cout << std::setw(10) << "Omega" << std::setw(20) << "Sound Floquet"
              << std::setw(20) << "Alfven Floquet" << "\n";
    std::cout << std::string(55, '-') << "\n";

    for (double omega = 0.001; omega <= 1.0; omega += 0.005) {
        std::vector<double> omega_vals = {omega};
        EigenwaveAnalyzer analyzer(omega_vals, params, withSoundGap, steps);

        const auto& sound_waves = analyzer.soundWaveTable();
        const auto& alfven_waves = analyzer.alfvenWaveTable();

        std::cout << std::setw(10) << omega;

        double sound_floquet_1 =
            sound_waves.size() > 0 ? sound_waves[0].floquet_exponent.imag() : 0;
        double sound_floquet_2 =
            sound_waves.size() > 1 ? sound_waves[1].floquet_exponent.imag() : 0;
        double alfven_floquet_1 = alfven_waves.size() > 0
                                      ? alfven_waves[0].floquet_exponent.imag()
                                      : 0;
        double alfven_floquet_2 = alfven_waves.size() > 1
                                      ? alfven_waves[1].floquet_exponent.imag()
                                      : 0;

        std::cout << std::setw(20) << sound_floquet_1;
        if (sound_waves.size() > 1) { std::cout << ", " << sound_floquet_2; }
        std::cout << std::setw(20) << alfven_floquet_1;
        if (alfven_waves.size() > 1) { std::cout << ", " << alfven_floquet_2; }
        std::cout << "\n";

        double sound_alfv_1 =
            sound_waves.size() > 0 ? sound_waves[0].alfvenicity_val : 0;
        double sound_alfv_2 =
            sound_waves.size() > 1 ? sound_waves[1].alfvenicity_val : 0;
        double alfven_alfv_1 =
            alfven_waves.size() > 0 ? alfven_waves[0].alfvenicity_val : 0;
        double alfven_alfv_2 =
            alfven_waves.size() > 1 ? alfven_waves[1].alfvenicity_val : 0;

        out << omega << "," << sound_floquet_1 << "," << sound_floquet_2 << ","
            << alfven_floquet_1 << "," << alfven_floquet_2 << ","
            << sound_alfv_1 << "," << sound_alfv_2 << "," << alfven_alfv_1
            << "," << alfven_alfv_2 << "\n";
    }

    out.close();

    std::cout << "\n=== Results saved ===\n";
    std::cout << "CSV: continuum_psi_" << target_psi << ".csv\n";
    std::cout
        << "Use python3 to plot: omega (x-axis) vs Floquet exponent (y-axis)\n";
    std::cout
        << "\nNote: Check if Floquet exponents vary smoothly with omega.\n";
    std::cout << "Strange behavior may indicate issues with getMGeo or RK4 "
                 "integration.\n";

    return 0;
}
