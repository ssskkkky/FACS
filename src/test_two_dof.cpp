#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../include/equilibrium.h"
#include "../include/gFileRawData.h"
#include "../include/two_dof_continuum.h"

using namespace std::complex_literals;

int main() {
    std::cout << "Starting main\n";

    TwoDofParams params;
    params.couplingTerm = 0.05;

    bool withSoundGap = false;
    std::size_t steps = 1000;

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "\n=== ALFVENICITY BENCHMARK (omega=0.1) ===\n";
    double test_omega = 0.1;
    FloquetMatrix4D<double> floquet(params, test_omega, withSoundGap, steps);
    auto mults = floquetMultipliers(floquet.monodromy);
    auto evals = eigenvalues(floquet.monodromy);
    auto evecs = eigenvectors(floquet.monodromy);

    std::cout << "Floquet Multipliers:\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "  μ" << i << " = " << mults[i].real() << "+"
                  << mults[i].imag() << "i\n";
    }
    std::cout << "\n";

    double theta = 0.0;
    auto M = getM(theta, test_omega, params, withSoundGap);
    double M11_val = M[0][0];
    double M22_val = M[1][1];

    std::vector<double> alfvenicity_times;
    const int bench_iterations = 10000;

    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < bench_iterations; ++j) {
            volatile double alf =
                alfvenicity(M11_val, M22_val, test_omega, evecs[i]);
            (void)alf;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto dur =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        double avg_ns = static_cast<double>(dur.count()) / bench_iterations;
        alfvenicity_times.push_back(avg_ns);

        double alf = alfvenicity(M11_val, M22_val, test_omega, evecs[i]);
        std::cout << "Eigenpair " << i << ":\n";
        std::cout << "  Eigenvalue: " << evals[i].real() << "+"
                  << evals[i].imag() << "i\n";
        std::cout << "  Eigenvector: [";
        for (int j = 0; j < 4; ++j) {
            std::cout << evecs[i][j].real() << "+" << evecs[i][j].imag() << "i";
            if (j < 3) std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "  Alfvenicity: " << alf << "\n";
        std::cout << "  Average time: " << avg_ns << " ns\n\n";
    }

    std::cout << "=== ALFVENICITY SUMMARY ===\n";
    double alf_avg = 0, alf_min = 1e9, alf_max = 0;
    for (size_t i = 0; i < alfvenicity_times.size(); ++i) {
        alf_avg += alfvenicity_times[i];
        alf_min = std::min(alf_min, alfvenicity_times[i]);
        alf_max = std::max(alf_max, alfvenicity_times[i]);
    }
    alf_avg /= alfvenicity_times.size();
    std::cout << "Average time: " << alf_avg << " ns\n";
    std::cout << "Min time: " << alf_min << " ns\n";
    std::cout << "Max time: " << alf_max << " ns\n";

    return 0;
}
