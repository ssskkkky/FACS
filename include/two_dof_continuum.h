#ifndef TWO_DOF_CONTINUUM_H
#define TWO_DOF_CONTINUUM_H

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include "equilibrium.h"
#include "integrator.h"

const double PI = 3.141592653589793;

using Vec4 = std::array<std::complex<double>, 4>;
using Mat4 = std::array<std::array<std::complex<double>, 4>, 4>;

inline Vec4 operator*(double a, const Vec4& v) {
    Vec4 res;
    for (int i = 0; i < 4; ++i) res[i] = a * v[i];
    return res;
}

inline Vec4 operator*(std::complex<double> a, const Vec4& v) {
    Vec4 res;
    for (int i = 0; i < 4; ++i) res[i] = a * v[i];
    return res;
}

inline Vec4 operator+(const Vec4& a, const Vec4& b) {
    Vec4 res;
    for (int i = 0; i < 4; ++i) res[i] = a[i] + b[i];
    return res;
}

inline Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 res{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k) res[i][j] += a[i][k] * b[k][j];
    return res;
}

inline std::array<std::complex<double>, 4> solve_quartic(double a,
                                                         double b,
                                                         double c,
                                                         double d) {
    // Durand-Kerner method for simultaneous root finding
    auto poly = [&](std::complex<double> x) {
        std::complex<double> x2 = x * x;
        std::complex<double> x3 = x2 * x;
        std::complex<double> x4 = x3 * x;
        return x4 + a * x3 + b * x2 + c * x + d;
    };
    std::array<std::complex<double>, 4> roots = {
        1.0, -1.0, std::complex<double>(0, 1), std::complex<double>(0, -1)};
    const int max_iter = 100;
    const double tol = 1e-12;
    for (int iter = 0; iter < max_iter; ++iter) {
        bool converged = true;
        for (int k = 0; k < 4; ++k) {
            std::complex<double> prod = 1.0;
            for (int j = 0; j < 4; ++j) {
                if (j != k) prod *= (roots[k] - roots[j]);
            }
            std::complex<double> pval = poly(roots[k]);
            std::complex<double> delta = pval / prod;
            roots[k] -= delta;
            if (std::abs(delta) > tol) converged = false;
        }
        if (converged) break;
    }
    return roots;
}

inline std::complex<double> det(const Mat4& m) {
    auto a = m[0][0], b = m[0][1], c = m[0][2], d = m[0][3];
    auto e = m[1][0], f = m[1][1], g = m[1][2], h = m[1][3];
    auto ii = m[2][0], jj = m[2][1], kk = m[2][2], ll = m[2][3];
    auto mm = m[3][0], nn = m[3][1], oo = m[3][2], pp = m[3][3];
    return a * (f * (kk * pp - ll * oo) - g * (jj * pp - ll * nn) +
                h * (jj * oo - kk * nn)) -
           b * (e * (kk * pp - ll * oo) - g * (ii * pp - ll * mm) +
                h * (ii * oo - kk * mm)) +
           c * (e * (jj * pp - ll * nn) - f * (ii * pp - ll * mm) +
                h * (ii * nn - jj * mm)) -
           d * (e * (jj * oo - kk * nn) - f * (ii * oo - kk * mm) +
                g * (ii * nn - jj * mm));
}

struct TwoDofParams {
    double deltaPrime = 0.2;
    double epsilon = 0.2;
    double couplingTerm = 0.0;
    double omegaA = 1.0;
    double omegaS = 0.2;
    double J = 1.0;
    double B0 = 1.0;
    double q_val = 1.0;
    double R0 = 1.0;
    double Gamma_val = 1.0;
    double beta_val = 1.0;
    double psi = 0.0;
    double deriv_term = 0.0;
    double kappa_g = 0.0;
    NumericEquilibrium<double>* eq = nullptr;
    bool use_geo = false;
};

inline std::array<std::array<double, 2>, 2>
getMGeo(double theta, double omega, const TwoDofParams& params) {
    // Compute theta-dependent quantities
    double J = params.eq->safety_factor(params.psi) *
               std::sqrt(params.eq->j_func(params.psi, theta));
    double B0 = params.eq->intp_data().intp_2d[0](params.psi, theta);
    double deriv_term = params.eq->radial_func(params.psi, theta);

    // Compute kappa_g with theta dependence
    double eps = 0.01;  // This is absolutely wrong, I will deal with this later
    double F_psi = params.eq->intp_data().intp_1d[1](params.psi);
    double R = params.eq->intp_data().intp_2d[1](params.psi, theta);
    double B_phi = F_psi / R;
    double B_p = std::sqrt(B0 * B0 - B_phi * B_phi);
    double grad_psi =
        J / (params.eq->intp_data().intp_1d[0](params.psi) * R * B_p);
    double dB_dtheta =
        (params.eq->intp_data().intp_2d[0](params.psi, theta + eps) -
         params.eq->intp_data().intp_2d[0](params.psi, theta - eps)) /
        (2 * eps);
    double kappa_g = (F_psi / grad_psi) * (dB_dtheta / (J * B0 * B0));
    double omegaA_q_R0 = 1.0;
    double omegaS_q_R0 = std::sqrt(params.beta_val / 2.0) * omegaA_q_R0;

    // Compute matrix elements
    double M11 =
        (omega * omega * J * J * B0 * B0) / (omegaA_q_R0 * omegaA_q_R0) -
        deriv_term;
    double M22 =
        (omega * omega / (omegaS_q_R0 * omegaS_q_R0)) * (J * J * B0 * B0);
    double off_diag = std::sqrt(2 * params.Gamma_val * params.beta_val) *
                      (J * J * B0 * B0) * kappa_g * (omega / omegaS_q_R0);
    double M12 = off_diag;
    double M21 = off_diag;
    return {{{M11, M12}, {M21, M22}}};
}

inline std::array<std::array<double, 2>, 2> getM(double theta,
                                                 double omega,
                                                 const TwoDofParams& params,
                                                 bool withSoundGap) {
    if (params.use_geo) {
        return getMGeo(theta, omega, params);
    } else {
        double M11 = params.deltaPrime * std::cos(theta) +
                     (omega * omega / (params.omegaA * params.omegaA)) *
                         (1 + 4 * params.epsilon * std::cos(theta));
        double M12 =
            -params.couplingTerm * std::sin(theta) * (omega / params.omegaS);
        double M21 = M12;
        double M22 = (omega * omega) / (params.omegaS * params.omegaS);
        if (withSoundGap) M22 *= (1 + 4 * params.epsilon * std::cos(theta));
        return {{{M11, M12}, {M21, M22}}};
    }
}

inline Vec4 rhs(const Vec4& Y,
                double theta,
                double omega,
                const TwoDofParams& params,
                bool withSoundGap) {
    auto M = getM(theta, omega, params, withSoundGap);
    Vec4 dY;
    dY[0] = Y[2];
    dY[1] = Y[3];
    dY[2] = -M[0][0] * Y[0] - M[0][1] * Y[1];
    dY[3] = -M[1][0] * Y[0] - M[1][1] * Y[1];
    return dY;
}

struct State4D {
    using value_type = double;
    using velocity_type = Vec4;

    State4D(double t_,
            const Vec4& y_,
            double omega_,
            const TwoDofParams& p_,
            bool wsg)
        : t(t_), y(y_), omega(omega_), params(p_), withSoundGap(wsg) {}

    void put_velocity(velocity_type& v) {
        v = rhs(y, t, omega, params, withSoundGap);
    }

    void update(velocity_type v, value_type dt) {
        for (int i = 0; i < 4; ++i) y[i] += v[i] * dt;
        t += dt;
    }

    auto get_update_err(velocity_type v, value_type dt) {
        value_type err = 0;
        for (auto& dv : v) err = std::max(err, std::abs(dv) * dt);
        return err;
    }

    auto initial_velocity_storage() { return Vec4{}; }

    double t;
    Vec4 y;
    double omega;
    TwoDofParams params;
    bool withSoundGap;
};

template <typename T>
struct FloquetMatrix4D {
    using value_type = T;

    FloquetMatrix4D(const TwoDofParams& params,
                    value_type omega,
                    bool withSoundGap,
                    std::size_t steps) {
        value_type period = 2 * PI;
        std::array<Vec4, 4> columns;
        for (int i = 0; i < 4; ++i) {
            Vec4 y{0.0, 0.0, 0.0, 0.0};
            y[i] = 1.0;
            double t = 0.0;
            value_type dt = period / steps;
            for (std::size_t j = 0; j < steps; ++j) {
                // RK4 step
                Vec4 k1 = rhs(y, t, omega, params, withSoundGap);
                Vec4 y2 = y + (0.5 * dt) * k1;
                Vec4 k2 = rhs(y2, t + 0.5 * dt, omega, params, withSoundGap);
                Vec4 y3 = y + (0.5 * dt) * k2;
                Vec4 k3 = rhs(y3, t + 0.5 * dt, omega, params, withSoundGap);
                Vec4 y4 = y + dt * k3;
                Vec4 k4 = rhs(y4, t + dt, omega, params, withSoundGap);
                y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
                t += dt;
            }
            columns[i] = y;
        }
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) monodromy[i][j] = columns[j][i];
    }

    Mat4 monodromy;
};

// Hessenberg reduction for QR algorithm
inline void hessenberg(Mat4& A) {
    for (int k = 0; k < 3; ++k) {  // for 4x4
        int len = 4 - k - 1;
        if (len <= 0) continue;

        // Extract subvector
        std::array<std::complex<double>, 4> x{};
        for (int i = 0; i < len; ++i) { x[i] = A[k + 1 + i][k]; }

        std::complex<double> alpha = 0.0;
        for (int i = 0; i < len; ++i) { alpha += x[i] * std::conj(x[i]); }
        alpha = std::sqrt(alpha);

        if (std::abs(alpha) < 1e-12) continue;

        if (std::real(x[0]) >= 0) alpha = -alpha;

        x[0] -= alpha;

        std::complex<double> norm = 0.0;
        for (int i = 0; i < len; ++i) { norm += x[i] * std::conj(x[i]); }
        norm = std::sqrt(norm);

        if (std::abs(norm) < 1e-12) continue;

        for (int i = 0; i < len; ++i) { x[i] /= norm; }

        // Apply Householder to columns
        for (int j = k; j < 4; ++j) {
            std::complex<double> sum = 0.0;
            for (int i = 0; i < len; ++i) {
                sum += std::conj(x[i]) * A[k + 1 + i][j];
            }
            for (int i = 0; i < len; ++i) {
                A[k + 1 + i][j] -= 2.0 * x[i] * sum;
            }
        }

        // Apply to rows
        for (int i = 0; i < 4; ++i) {
            std::complex<double> sum = 0.0;
            for (int j = 0; j < len; ++j) { sum += A[i][k + 1 + j] * x[j]; }
            for (int j = 0; j < len; ++j) {
                A[i][k + 1 + j] -= 2.0 * std::conj(x[j]) * sum;
            }
        }
    }
}

// QR decomposition for complex matrices using modified Gram-Schmidt
inline void qr_decomposition(const Mat4& A, Mat4& Q, Mat4& R) {
    // Initialize R to zero
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) { R[i][j] = 0.0; }
    }

    // Start with columns of A
    std::array<Vec4, 4> cols;
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) { cols[j][i] = A[i][j]; }
    }

    // Modified Gram-Schmidt
    for (int i = 0; i < 4; ++i) {
        // Compute norm of column i
        std::complex<double> norm = 0.0;
        for (int k = 0; k < 4; ++k) {
            norm += cols[i][k] * std::conj(cols[i][k]);
        }
        norm = std::sqrt(norm);

        if (std::abs(norm) < 1e-14) {
            for (int k = 0; k < 4; ++k) { Q[k][i] = 0.0; }
            R[i][i] = 0.0;
            continue;
        }

        // Normalize column i to get column of Q
        for (int k = 0; k < 4; ++k) { Q[k][i] = cols[i][k] / norm; }
        R[i][i] = norm;

        // Orthogonalize remaining columns against column i
        for (int j = i + 1; j < 4; ++j) {
            // R[i][j] = q_i^H * a_j
            std::complex<double> dot = 0.0;
            for (int k = 0; k < 4; ++k) {
                dot += std::conj(Q[k][i]) * cols[j][k];
            }
            R[i][j] = dot;

            // a_j = a_j - q_i * R[i][j]
            for (int k = 0; k < 4; ++k) { cols[j][k] -= Q[k][i] * R[i][j]; }
        }
    }
}

// QR algorithm for eigenvalues with Wilkinson shift
inline std::array<std::complex<double>, 4> qr_eigenvalues(const Mat4& A) {
    Mat4 H = A;
    hessenberg(H);

    const double eps = 1e-10;
    std::vector<std::complex<double>> evals;
    int n = 4;

    while (n > 0) {
        // Check for deflation at bottom
        if (n == 1) {
            evals.push_back(H[0][0]);
            break;
        }

        // Check if subdiagonal element is negligible
        if (std::abs(H[n - 1][n - 2]) < eps) {
            evals.push_back(H[n - 1][n - 1]);
            n--;
            continue;
        }

        // Check for 2x2 block at bottom
        if (n == 2) {
            std::complex<double> a = H[0][0];
            std::complex<double> b = H[0][1];
            std::complex<double> c = H[1][0];
            std::complex<double> d = H[1][1];
            std::complex<double> trace = a + d;
            std::complex<double> det = a * d - b * c;
            std::complex<double> disc = std::sqrt(trace * trace - 4.0 * det);
            evals.push_back((trace + disc) / 2.0);
            evals.push_back((trace - disc) / 2.0);
            break;
        }

        // Wilkinson shift: compute eigenvalue of bottom 2x2 block closest to
        // H[n-1][n-1]
        std::complex<double> a = H[n - 2][n - 2];
        std::complex<double> b = H[n - 2][n - 1];
        std::complex<double> c = H[n - 1][n - 2];
        std::complex<double> d = H[n - 1][n - 1];
        std::complex<double> trace = a + d;
        std::complex<double> det = a * d - b * c;
        std::complex<double> disc = std::sqrt(trace * trace - 4.0 * det);
        std::complex<double> e1 = (trace + disc) / 2.0;
        std::complex<double> e2 = (trace - disc) / 2.0;
        std::complex<double> shift =
            (std::abs(e1 - d) < std::abs(e2 - d)) ? e1 : e2;

        // Apply shift: H - shift*I
        for (int i = 0; i < n; ++i) H[i][i] -= shift;

        // QR decomposition of top-left n×n block
        Mat4 Q{}, R{};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                Q[i][j] = 0.0;
                R[i][j] = 0.0;
            }
        }

        // Modified Gram-Schmidt for n×n block
        std::array<Vec4, 4> cols;
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) cols[j][i] = H[i][j];
        }

        for (int i = 0; i < n; ++i) {
            std::complex<double> norm = 0.0;
            for (int k = 0; k < n; ++k) {
                norm += cols[i][k] * std::conj(cols[i][k]);
            }
            norm = std::sqrt(norm);

            if (std::abs(norm) < 1e-14) {
                for (int k = 0; k < 4; ++k) Q[k][i] = 0.0;
                R[i][i] = 0.0;
                continue;
            }

            for (int k = 0; k < n; ++k) Q[k][i] = cols[i][k] / norm;
            R[i][i] = norm;

            for (int j = i + 1; j < n; ++j) {
                std::complex<double> dot = 0.0;
                for (int k = 0; k < n; ++k) {
                    dot += std::conj(Q[k][i]) * cols[j][k];
                }
                R[i][j] = dot;

                for (int k = 0; k < n; ++k) cols[j][k] -= Q[k][i] * R[i][j];
            }
        }

        // H = R*Q + shift*I (only for n×n block)
        Mat4 new_H{};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) new_H[i][j] = 0.0;
        }

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    new_H[i][j] += R[i][k] * Q[k][j];
                }
            }
        }

        // Copy converged part from bottom right
        for (int i = n; i < 4; ++i) {
            for (int j = n; j < 4; ++j) { new_H[i][j] = H[i][j]; }
        }

        // Add shift back
        for (int i = 0; i < n; ++i) new_H[i][i] += shift;

        H = new_H;

        // Check for convergence of last subdiagonal
        if (std::abs(H[n - 1][n - 2]) < eps) {
            evals.push_back(H[n - 1][n - 1]);
            n--;
        }
    }

    std::array<std::complex<double>, 4> result;
    for (size_t i = 0; i < 4 && i < evals.size(); ++i) result[i] = evals[i];
    return result;
}

// Eigenvalue computation using QR algorithm with Wilkinson shift
inline std::array<std::complex<double>, 4> eigenvalues(const Mat4& m) {
    return qr_eigenvalues(m);
}

inline double aPara(const Mat4& monodromy) {
    std::complex<double> tr = 0;
    for (int i = 0; i < 4; ++i) tr += monodromy[i][i];
    return tr.real();
}

inline double bPara(const Mat4& monodromy) {
    double a = aPara(monodromy);
    double tr_sq = 0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            tr_sq += (monodromy[i][j] * monodromy[j][i]).real();
    return (a * a - tr_sq) / 2;
}

// Durand-Kerner based multipliers for comparison
inline std::array<std::complex<double>, 4> floquetMultipliers_dk(
    const Mat4& monodromy) {
    std::complex<double> tr = 0;
    for (int i = 0; i < 4; ++i) tr += monodromy[i][i];
    Mat4 A2 = multiply(monodromy, monodromy);
    std::complex<double> trA2 = 0;
    for (int i = 0; i < 4; ++i) trA2 += A2[i][i];
    std::complex<double> c2 = (tr * tr - trA2) / 2.0;
    Mat4 A3 = multiply(A2, monodromy);
    std::complex<double> trA3 = 0;
    for (int i = 0; i < 4; ++i) trA3 += A3[i][i];
    std::complex<double> c1 =
        (tr * tr * tr - 3.0 * tr * trA2 + 2.0 * trA3) / 6.0;
    std::complex<double> c0 = det(monodromy);
    // Solve x^4 - tr x^3 + c2 x^2 - c1 x + c0 = 0
    return solve_quartic(-tr.real(), c2.real(), -c1.real(), c0.real());
}

// Alternative floquet exponents using Durand-Kerner eigenvalues
inline std::array<std::complex<double>, 4> floquetExponents_dk(
    const Mat4& monodromy) {
    auto mult = floquetMultipliers_dk(monodromy);
    for (auto& m : mult) m = std::log(m) / (2 * PI);
    return mult;
}

inline std::array<std::complex<double>, 4> floquetMultipliers(
    const Mat4& monodromy) {
    return qr_eigenvalues(monodromy);
}

inline std::array<std::complex<double>, 4> floquetExponents(
    const Mat4& monodromy) {
    auto mult = floquetMultipliers(monodromy);
    for (auto& m : mult) m = std::log(m) / (2 * PI);
    return mult;
}

// Compute eigenvector for a given eigenvalue using inverse iteration
inline Vec4 eigenvector(const Mat4& m, std::complex<double> lambda) {
    Vec4 v = {1.0, 0.1, 0.1, 0.1};

    // Do inverse iteration: solve (M - λI) * v_new = v, then normalize
    for (int iter = 0; iter < 10; ++iter) {
        Mat4 A = m;
        for (int i = 0; i < 4; ++i) A[i][i] -= lambda;

        Vec4 b = v;
        // Gaussian elimination to solve A * x = b
        for (int col = 0; col < 4; ++col) {
            int pivot = col;
            for (int row = col + 1; row < 4; ++row) {
                if (std::abs(A[row][col]) > std::abs(A[pivot][col]))
                    pivot = row;
            }
            std::swap(A[col], A[pivot]);
            std::swap(b[col], b[pivot]);

            for (int row = col + 1; row < 4; ++row) {
                std::complex<double> factor = A[row][col] / A[col][col];
                for (int j = col; j < 4; ++j) A[row][j] -= factor * A[col][j];
                b[row] -= factor * b[col];
            }
        }

        Vec4 v_new{0.0, 0.0, 0.0, 0.0};
        for (int row = 3; row >= 0; --row) {
            std::complex<double> sum = b[row];
            for (int col = row + 1; col < 4; ++col)
                sum -= A[row][col] * v_new[col];
            v_new[row] = sum / A[row][row];
        }

        v = v_new;
        // Normalize to unit norm
        std::complex<double> norm = 0.0;
        for (auto& val : v) norm += val * std::conj(val);
        norm = std::sqrt(norm);
        for (auto& val : v) val /= norm;
    }

    // Make largest element pure real
    int max_idx = 0;
    double max_abs = 0.0;
    for (int i = 0; i < 4; ++i) {
        double abs_val = std::abs(v[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
            max_idx = i;
        }
    }
    std::complex<double> phase = std::conj(v[max_idx]) / std::abs(v[max_idx]);
    for (auto& val : v) val *= phase;

    return v;
}

// Compute all eigenvectors
inline std::array<Vec4, 4> eigenvectors(const Mat4& m) {
    auto evals = eigenvalues(m);
    std::array<Vec4, 4> evecs;
    for (int i = 0; i < 4; ++i) { evecs[i] = eigenvector(m, evals[i]); }
    return evecs;
}

// Verify eigenvector: compute M*v and compare with λ*v
inline void verify_eigenvector(const Mat4& m,
                               std::complex<double> lambda,
                               const Vec4& v) {
    Vec4 mv{0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) { mv[i] += m[i][j] * v[j]; }
    }
    Vec4 lambda_v = lambda * v;

    std::cout << "Verification: M*v - λ*v = ";
    double max_diff = 0.0;
    for (int i = 0; i < 4; ++i) {
        std::complex<double> diff = mv[i] - lambda_v[i];
        max_diff = std::max(max_diff, std::abs(diff));
    }
    std::cout << "max diff = " << max_diff << "\n";
}

inline double alfvenicity(double M11,
                          double M22,
                          double omega,
                          const Vec4& eigenVector) {
    double abs_sq[4];
    for (int i = 0; i < 4; ++i) {
        abs_sq[i] = std::abs(eigenVector[i]) * std::abs(eigenVector[i]);
    }

    double term1 = M11 * abs_sq[0] / 2.0;
    double term2 = abs_sq[2] / 2.0;
    double term3 = M22 * abs_sq[1] / 2.0;
    double term4 = abs_sq[3] / 2.0;

    double numerator = term1 + term2;
    double denominator = numerator + term3 + term4;

    return numerator / denominator;
}

#endif  // TWO_DOF_CONTINUUM_H
