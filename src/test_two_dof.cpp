#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../include/equilibrium.h"
#include "../include/gFileRawData.h"
#include "../include/two_dof_continuum.h"

using namespace std::complex_literals;

int main(int argc, char** argv) {
    std::cout << "Starting main\n";

    TwoDofParams params;
    params.couplingTerm = 0.05;
    params.beta_val = 0.02;
    params.Gamma_val = 1.6667;

    bool withSoundGap = false;
    std::size_t steps = 1000;

    bool use_geo = false;
    NumericEquilibrium<double>* equilibrium = nullptr;

    if (argc >= 2) {
        std::string gfile_path = argv[1];
        std::ifstream gfile(gfile_path);
        if (gfile.is_open()) {
            GFileRawData gfile_data;
            gfile >> gfile_data;
            gfile.close();
            if (gfile_data.is_complete()) {
                equilibrium =
                    new NumericEquilibrium<double>(gfile_data, 1000, 300, 0.96);
                params.eq = equilibrium;
                params.use_geo = true;
                params.psi = (equilibrium->psi_range().first +
                              equilibrium->psi_range().second) /
                             2.0;
                use_geo = true;
                std::cout << "Loaded g-file: " << gfile_path << "\n";
            } else {
                std::cerr << "Failed to parse g-file, falling back to getM\n";
            }
        } else {
            std::cerr << "Cannot open g-file: " << gfile_path
                      << ", falling back to getM\n";
        }
    }

    std::cout << std::fixed << std::setprecision(6);

    std::vector<double> omega_values;
    for (double w = 0.001; w <= 1.0; w += 0.001) { omega_values.push_back(w); }
    EigenwaveAnalyzer analyzer(omega_values, params, withSoundGap, steps);

    const auto& soundWave = analyzer.soundWaveTable();
    const auto& alfvenWave = analyzer.alfvenWaveTable();

    std::cout << "\n=== SOUND WAVE LIST ===\n";
    for (size_t i = 0; i < soundWave.size(); ++i) {
        std::cout << "SoundWave " << i << ":\n";
        std::cout << "  EigenData: {\n";
        std::cout << "    floquet_exponent: "
                  << soundWave[i].floquet_exponent.real() << "+"
                  << soundWave[i].floquet_exponent.imag() << "i,\n";
        std::cout << "    eigenvector: [";
        for (int j = 0; j < 4; ++j) {
            std::cout << soundWave[i].eigenvector[j].real() << "+"
                      << soundWave[i].eigenvector[j].imag() << "i";
            if (j < 3) std::cout << ", ";
        }
        std::cout << "],\n";
        std::cout << "    alfvenicity_val: " << soundWave[i].alfvenicity_val
                  << ",\n";
        std::cout << "    omega: " << soundWave[i].omega << "\n";
        std::cout << "  }\n\n";
    }

    std::cout << "\n=== ALFVEN WAVE LIST ===\n";
    for (size_t i = 0; i < alfvenWave.size(); ++i) {
        std::cout << "AlfvenWave " << i << ":\n";
        std::cout << "  EigenData: {\n";
        std::cout << "    floquet_exponent: "
                  << alfvenWave[i].floquet_exponent.real() << "+"
                  << alfvenWave[i].floquet_exponent.imag() << "i,\n";
        std::cout << "    eigenvector: [";
        for (int j = 0; j < 4; ++j) {
            std::cout << alfvenWave[i].eigenvector[j].real() << "+"
                      << alfvenWave[i].eigenvector[j].imag() << "i";
            if (j < 3) std::cout << ", ";
        }
        std::cout << "],\n";
        std::cout << "    alfvenicity_val: " << alfvenWave[i].alfvenicity_val
                  << ",\n";
        std::cout << "    omega: " << alfvenWave[i].omega << "\n";
        std::cout << "  }\n\n";
    }

    auto soundPts = analyzer.soundWavePoints();
    auto alfvenPts = analyzer.alfvenWavePoints();

    std::ofstream sound_file("sound_wave_points.csv");
    sound_file << "x,y\n";
    for (const auto& p : soundPts) { sound_file << p.x << "," << p.y << "\n"; }
    sound_file.close();

    std::ofstream alfven_file("alfven_wave_points.csv");
    alfven_file << "x,y\n";
    for (const auto& p : alfvenPts) {
        alfven_file << p.x << "," << p.y << "\n";
    }
    alfven_file.close();

    std::cout << "\n=== SOUND WAVE POINTS (x=imag(eigenvalue), y=omega) ===\n";
    for (size_t i = 0; i < soundPts.size(); ++i) {
        std::cout << "Point " << i << ": x = " << soundPts[i].x
                  << ", y = " << soundPts[i].y << "\n";
    }

    std::cout << "\n=== ALFVEN WAVE POINTS (x=imag(eigenvalue), y=omega) ===\n";
    for (size_t i = 0; i < alfvenPts.size(); ++i) {
        std::cout << "Point " << i << ": x = " << alfvenPts[i].x
                  << ", y = " << alfvenPts[i].y << "\n";
    }

    std::cout << "\n=== TESTING INTERPOLATION FUNCTIONS ===\n";

    std::ofstream interp_file("interpolation_test.csv");
    interp_file << "floquet_exponent,wave_type,omega\n";

    std::vector<double> test_exponents;
    for (double exp = -0.5; exp <= 0.5; exp += 0.05) {
        test_exponents.push_back(exp);
    }

    for (double exp : test_exponents) {
        std::cout << "\nTesting floquet exponent: " << exp << "\n";

        std::cout << "  getSoundOmegas: [";
        auto soundOmegas = analyzer.getSoundOmegas(exp);
        for (size_t i = 0; i < soundOmegas.size(); ++i) {
            std::cout << soundOmegas[i];
            interp_file << exp << ",sound," << soundOmegas[i] << "\n";
            if (i < soundOmegas.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        std::cout << "  getAlfvenOmegas: [";
        auto alfvenOmegas = analyzer.getAlfvenOmegas(exp);
        for (size_t i = 0; i < alfvenOmegas.size(); ++i) {
            std::cout << alfvenOmegas[i];
            interp_file << exp << ",alfven," << alfvenOmegas[i] << "\n";
            if (i < alfvenOmegas.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    interp_file.close();

    if (equilibrium) { delete equilibrium; }

    return 0;
}
