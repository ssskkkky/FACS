#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../include/equilibrium.h"
#include "../include/gFileRawData.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <path_to_gfile> [num_surfaces]\n";
        return 1;
    }

    std::string gfile_path = argv[1];
    std::size_t num_surfaces = 50;
    if (argc >= 3) { num_surfaces = std::stoul(argv[2]); }

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
    double psi_ratio = 0.96;

    std::vector<NumericEquilibrium<double>*> equilibria;
    std::vector<double> psi_values;
    std::vector<double> q_values;
    std::vector<double> nqm_values;

    int n = 5;
    int m = 10;

    std::cout << "\n=== CREATING EQUILIBRIUM LIST ===\n";
    std::cout << "Number of surfaces: " << num_surfaces << "\n";
    std::cout << "n = " << n << ", m = " << m << "\n\n";

    std::cout << std::fixed << std::setprecision(6);
    std::cout << std::setw(10) << "Index" << std::setw(12) << "Psi"
              << std::setw(12) << "q" << std::setw(15) << "n*q - m" << "\n";
    std::cout << std::string(55, '-') << "\n";

    for (std::size_t i = 0; i < num_surfaces; ++i) {
        double psi_target = (i + 0.5) / num_surfaces;

        NumericEquilibrium<double>* eq = new NumericEquilibrium<double>(
            gfile_data, radial_grid, poloidal_grid);

        auto psi_range = eq->psi_range();
        double psi =
            psi_range.first + (psi_range.second - psi_range.first) * psi_target;

        double q = eq->safety_factor(psi);
        double nqm = n * q - m;

        equilibria.push_back(eq);
        psi_values.push_back(psi);
        q_values.push_back(q);
        nqm_values.push_back(nqm);

        std::cout << std::setw(10) << i << std::setw(12) << psi << std::setw(12)
                  << q << std::setw(15) << nqm;

        if (std::abs(nqm) < 0.01) { std::cout << "  <<< RESONANCE"; }
        std::cout << "\n";
    }

    std::cout << "\n=== RESONANCE ANALYSIS ===\n";
    std::cout
        << "Looking for surfaces where n*q - m ≈ 0 (rational surfaces)\n\n";

    for (std::size_t i = 0; i < num_surfaces; ++i) {
        if (std::abs(nqm_values[i]) < 0.01) {
            std::cout << "Resonant surface found at index " << i << ":\n";
            std::cout << "  Psi = " << psi_values[i] << "\n";
            std::cout << "  q = " << q_values[i] << "\n";
            std::cout << "  n*q - m = " << nqm_values[i] << "\n";
            std::cout << "\n";
        }
    }

    std::cout << "\n=== SAVING DATA TO CSV ===\n";
    std::ofstream out("equilibrium_surfaces.csv");
    out << "index,psi,q,nqm,minor_radius\n";
    for (std::size_t i = 0; i < num_surfaces; ++i) {
        double r_minor = equilibria[i]->minor_radius(psi_values[i]);
        out << i << "," << psi_values[i] << "," << q_values[i] << ","
            << nqm_values[i] << "," << r_minor << "\n";
    }
    out.close();
    std::cout << "Data saved to equilibrium_surfaces.csv\n";

    for (auto* eq : equilibria) { delete eq; }

    std::cout << "\n=== SUMMARY ===\n";
    std::cout << "Total equilibria created: " << num_surfaces << "\n";
    std::cout << "Psi range: [" << psi_values.front() << ", "
              << psi_values.back() << "]\n";
    std::cout << "q range: ["
              << *std::min_element(q_values.begin(), q_values.end()) << ", "
              << *std::max_element(q_values.begin(), q_values.end()) << "]\n";
    std::cout << "n*q - m range: ["
              << *std::min_element(nqm_values.begin(), nqm_values.end()) << ", "
              << *std::max_element(nqm_values.begin(), nqm_values.end())
              << "]\n";

    return 0;
}
