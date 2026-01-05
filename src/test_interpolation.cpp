#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../include/Floquet_two_dof.h"
#include "../include/equilibrium.h"
#include "../include/gFileRawData.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <gfile> [psi]\n";
        std::cerr << "Default psi = 0.02\n";
        return 1;
    }

    std::string gfile_path = argv[1];
    double target_psi = 0.02;
    if (argc >= 3) { target_psi = std::stod(argv[2]); }

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

    std::cout << "=== GENERATING INTERPOLATION TEST DATA ===\n";
    std::cout << "G-file: " << gfile_path << "\n";
    std::cout << "Target psi: " << target_psi << "\n\n";

    std::size_t radial_grid = 1000;
    std::size_t poloidal_grid = 300;
    double psi_ratio = 0.96;

    NumericEquilibrium<double> eq(gfile_data, radial_grid, poloidal_grid);
    auto psi_range = eq.psi_range();

    if (target_psi < psi_range.first || target_psi > psi_range.second) {
        std::cerr << "Psi " << target_psi << " is out of range ["
                  << psi_range.first << ", " << psi_range.second << "]\n";
        return 1;
    }

    double q = eq.safety_factor(target_psi);
    double r_minor = eq.minor_radius(target_psi);

    std::cout << "=== EQUILIBRIUM INFO ===\n";
    std::cout << "Safety factor q: " << q << "\n";
    std::cout << "Minor radius r/a: " << r_minor << "\n\n";

    TwoDofParams params;
    params.eq = &eq;
    params.use_geo = true;
    params.psi = target_psi;
    params.beta_val = 0.0206494;
    params.Gamma_val = 1.6667;

    bool withSoundGap = false;
    std::size_t steps = 500;

    std::cout << "=== PARAMETERS ===\n";
    std::cout << "getMGeo enabled: " << (params.use_geo ? "yes" : "no") << "\n";
    std::cout << "withSoundGap: " << (withSoundGap ? "yes" : "no") << "\n";
    std::cout << "Integration steps: " << steps << "\n\n";

    std::cout << "Scanning omega = 0.001 to 1.0 (step 0.001)...\n";

    std::vector<double> omega_values;
    for (double w = 0.001; w <= 1.0; w += 0.001) { omega_values.push_back(w); }

    EigenwaveAnalyzer analyzer(omega_values, params, withSoundGap, steps);

    std::ofstream out("interpolation_test.csv");
    out << "floquet_exponent,wave_type,omega\n";

    auto sound_points = analyzer.soundWavePoints();
    auto alfven_points = analyzer.alfvenWavePoints();

    std::cout << "Found " << sound_points.size() << " sound wave points\n";
    std::cout << "Found " << alfven_points.size() << " Alfven wave points\n\n";

    int sound_count = 0;
    int alfven_count = 0;

    for (const auto& p : sound_points) {
        out << std::fixed << std::setprecision(3) << p.x << ",sound,"
            << std::setprecision(6) << p.y << "\n";
        sound_count++;
    }

    for (const auto& p : alfven_points) {
        out << std::fixed << std::setprecision(3) << p.x << ",alfven,"
            << std::setprecision(6) << p.y << "\n";
        alfven_count++;
    }

    out.close();

    std::cout << "=== OUTPUT SAVED ===\n";
    std::cout << "File: interpolation_test.csv\n";
    std::cout << "Sound waves: " << sound_count << " points\n";
    std::cout << "Alfven waves: " << alfven_count << " points\n";
    std::cout << "Total: " << (sound_count + alfven_count) << " points\n\n";

    std::cout << "Format: floquet_exponent,wave_type,omega\n";
    std::cout << "Plot: omega (y-axis) vs floquet_exponent (x-axis)\n";

    return 0;
}
