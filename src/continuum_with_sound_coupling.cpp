#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "../include/Floquet_two_dof.h"
#include "../include/equilibrium.h"
#include "../include/gFileRawData.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <path_to_gfile> <num_surfaces> <n_values>\n";
        std::cerr << "Example: " << argv[0] << " gfile 100 5,10,15\n";
        return 1;
    }

    std::string gfile_path = argv[1];
    std::size_t num_surfaces = std::stoul(argv[2]);
    double Gamma_val = 5.0 / 3.0;

    std::vector<int> n_values;
    std::string n_str = argv[3];
    std::stringstream ss(n_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        n_values.push_back(std::stoi(token));
    }

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

    std::size_t radial_grid = 1000;
    std::size_t poloidal_grid = 300;

    bool withSoundGap = false;
    std::size_t steps = 1000;

    std::cout << "\n=== CONFIGURATION ===\n";
    std::cout << "Number of surfaces: " << num_surfaces << "\n";
    std::cout << "n values: ";
    for (size_t i = 0; i < n_values.size(); ++i) {
        std::cout << n_values[i];
        if (i < n_values.size() - 1) std::cout << ", ";
    }
    std::cout << "\n";
    std::cout << "Gamma_val = " << Gamma_val << "\n";
    std::cout << "withSoundGap = " << (withSoundGap ? "true" : "false") << "\n";
    std::cout << "Integration steps: " << steps << "\n\n";

    NumericEquilibrium<double> eq(gfile_data, radial_grid, poloidal_grid);
    auto psi_range = eq.psi_range();

    std::cout << "=== EQUILIBRIUM INFO ===\n";
    std::cout << "Psi range: [" << psi_range.first << ", " << psi_range.second
              << "]\n\n";

    std::ofstream out("continuum_with_sound_coupling.csv");
    out << "n,index,minor_radius,psi,q,nq,sound_branches,alfven_branches,";
    out << "sound_omegas,sound_alfvenicities,alfven_omegas,alfven_"
           "alfvenicities\n";

    std::vector<double> omega_values;
    for (double w = 0.001; w <= 1.0; w += 0.005) { omega_values.push_back(w); }

    for (std::size_t i = 0; i < num_surfaces; ++i) {
        double target_r = (i + 0.5) / num_surfaces;

        double psi = psi_range.first;
        double r_min = eq.minor_radius(psi_range.first);
        double r_max = eq.minor_radius(psi_range.second);

        if (target_r < r_min || target_r > r_max) {
            std::cerr << "Warning: target_r = " << target_r
                      << " is outside range [" << r_min << ", " << r_max
                      << "]\n";
            continue;
        }

        double psi_low = psi_range.first;
        double psi_high = psi_range.second;
        const double psi_tol = 1e-8;
        double psi_mid;

        for (int iter = 0; iter < 100; ++iter) {
            psi_mid = 0.5 * (psi_low + psi_high);
            double r_mid = eq.minor_radius(psi_mid);

            if (std::abs(r_mid - target_r) < psi_tol) {
                psi = psi_mid;
                break;
            }

            if (r_mid < target_r) {
                psi_low = psi_mid;
            } else {
                psi_high = psi_mid;
            }
            psi = psi_mid;
        }

        double q = eq.safety_factor(psi);
        double r_minor = eq.minor_radius(psi);

        TwoDofParams params;
        params.eq = &eq;
        params.use_geo = true;
        params.psi = psi;
        params.beta_val = 0.0206494;
        params.Gamma_val = Gamma_val;

        std::size_t continuum_steps = 500;
        EigenwaveAnalyzer analyzer(omega_values, params, withSoundGap,
                                   continuum_steps);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\n=== FLUX SURFACE " << i << " (r/a=" << r_minor
                  << ") ===\n";
        std::cout << std::setw(5) << "n" << std::setw(10) << "n*q"
                  << std::setw(15) << "Sound" << std::setw(15) << "Alfven"
                  << "\n";
        std::cout << std::string(45, '-') << "\n";

        for (int n : n_values) {
            double nq = n * q;

            auto sound_oas = analyzer.getSoundOmegas(nq);
            auto alfven_oas = analyzer.getAlfvenOmegas(nq);

            std::cout << std::setw(5) << n << std::setw(10) << nq
                      << std::setw(15) << sound_oas.size() << std::setw(15)
                      << alfven_oas.size();

            if (abs(nq) < 0.05) { std::cout << "  <<< RESONANCE"; }
            std::cout << "\n";

            out << n << "," << i << "," << r_minor << "," << psi << "," << q
                << "," << nq << "," << sound_oas.size() << ","
                << alfven_oas.size() << ",\"[";

            for (size_t j = 0; j < sound_oas.size(); ++j) {
                out << sound_oas[j].omega;
                if (j < sound_oas.size() - 1) out << ";";
            }
            out << "]\",[";

            for (size_t j = 0; j < sound_oas.size(); ++j) {
                out << sound_oas[j].alfvenicity;
                if (j < sound_oas.size() - 1) out << ";";
            }
            out << "]\",[";

            for (size_t j = 0; j < alfven_oas.size(); ++j) {
                out << alfven_oas[j].omega;
                if (j < alfven_oas.size() - 1) out << ";";
            }
            out << "]\",[";

            for (size_t j = 0; j < alfven_oas.size(); ++j) {
                out << alfven_oas[j].alfvenicity;
                if (j < alfven_oas.size() - 1) out << ";";
            }
            out << "]\n";
        }

        std::cout << "Progress: " << (i + 1) << "/" << num_surfaces
                  << " flux surfaces processed\n";
        std::cout.flush();
    }

    out.close();

    std::cout << "\n=== RESULTS SAVED ===\n";
    std::cout << "Data saved to continuum_with_sound_coupling.csv\n";
    std::cout << "Columns: n, index, r/a, psi, q, n*q, sound_branches, "
                 "alfven_branches, sound_omegas, sound_alfvenicities, "
                 "alfven_omegas, alfven_alfvenicities\n\n";
    std::cout << "Use this CSV to create plots of omega vs minor_radius\n";
    std::cout << "where omega is interpolated from continuum using n*q as "
                 "Floquet exponent\n";
    std::cout << "alfvenicity indicates wave character (0=Sound, 1=Alfven)\n";

    return 0;
}
