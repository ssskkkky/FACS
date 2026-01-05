# Alfvenicity Implementation

## Overview

Implemented Alfvenicity calculation for 2-DOF continuum system based on Mathematica formula.

## Formula

```
alfvenicity = (M11 * (|eigenvector[0]|^2/2 + |eigenvector[2]|^2/2)) /
              (M11 * (|eigenvector[0]|^2/2 + |eigenvector[2]|^2/2) + 
               M22 * (|eigenvector[1]|^2/2 + |eigenvector[3]|^2/2))
```

Where:
- `M11, M22`: Matrix elements from `getM(theta=0, omega, ...)` at indices [0][0] and [1][1]
- `eigenvector`: 4D eigenvector of monodromy matrix
- `Abs[...]`: Magnitude `|z| = sqrt(Re(z)² + Im(z)²)`
- Formula interpretation: `^2/2` is `(z)^2` raised to power `1/2` = `|z|` **(NOT squaring)**

## Key: Eigenvector Normalization

**CRITICAL**: Eigenvectors MUST be normalized to unit norm before computing alfvenicity. Without normalization, different eigenvectors with different magnitudes would give incorrect alfvenicity values.

```cpp
// eigenvector function returns unit-normed eigenvectors
Vec4 eigenvector(const Mat4& m, std::complex<double> lambda) {
    Vec4 v = {1.0, 0.1, 0.1, 0.1};
    
    // Inverse iteration
    for (int iter = 0; iter < 10; ++iter) {
        Mat4 A = m;
        for (int i = 0; i < 4; ++i) A[i][i] -= lambda;
        
        Vec4 b = v;
        // Gaussian elimination
        for (int col = 0; col < 4; ++col) {
            int pivot = col;
            for (int row = col + 1; row < 4; ++row)
                if (std::abs(A[row][col]) > std::abs(A[pivot][col]))
                    pivot = row;
            std::swap(A[col], A[pivot]);
            std::swap(b[col], b[pivot]);
            
            for (int row = col + 1; row < 4; ++row) {
                std::complex<double> factor = A[row][col] / A[col][col];
                for (int j = col; j < 4; ++j) A[row][j] -= factor * A[col][j];
                b[row] -= factor * b[col];
            }
        }
        
        v = b;
    }
    
    // Normalize to unit norm
    std::complex<double> norm = 0.0;
    for (auto& val : v) norm += val * std::conj(val);
    norm = std::sqrt(norm);
    for (auto& val : v) val /= norm;
    
    return v;
}
```

## Implementation

### Header File (include/Floquet_two_dof.h)

```cpp
inline double alfvenicity(double M11, double M22, double omega, const Vec4& eigenVector) {
    // Mapping from Mathematica 1-based [[n]] to C++ 0-based [n-1]:
    // [[1]] -> [0], [[2]] -> [1], [[3]] -> [2], [[4]] -> [3]
    
    // alfvenicity = M11 * (|v0|²/2 + |v2|²/2) /
    //              (M11 * (|v0|²/2 + |v2|²/2) + M22 * (|v1|²/2 + |v3|²/2))
    
    double M11_weight = (std::abs(eigenVector[0]) * std::abs(eigenVector[0]) +
                          std::abs(eigenVector[2]) * std::abs(eigenVector[2])) / 2.0;
    
    double M22_weight = (std::abs(eigenVector[1]) * std::abs(eigenVector[1]) +
                          std::abs(eigenVector[3]) * std::abs(eigenVector[3])) / 2.0;
    
    double numerator = M11 * M11_weight;
    double denominator = numerator + M22 * M22_weight;
    
    return numerator / denominator;
}
```

## Test Results

For `omega = 0.1`, `couplingTerm = 0.05`:

| Eigenvector | Alfvenicity | Interpretation |
|-------------|-------------|----------------|
| 0 | 0.001005 | Primarily slow-wave mode (magnetic-dominated) |
| 1 | 0.023547 | Mixed mode with magnetic coupling |
| 2 | 0.036700 | Mixed mode, Alfvenic character increasing |
| 3 | 0.925605 | Primarily Alfvenic mode |

### Physical Meaning

The alfvenicity measure quantifies **relative contribution of Alfvénic vs slow-wave dynamics**:

- **alfvenicity ≈ 0**: Pure slow-wave behavior
- **alfvenicity ≈ 1**: Pure Alfvénic behavior  
- **0 < alfvenicity < 1**: Mixed character

This helps identify which continuum modes are:
- **Alfvénic modes**: Good magnetic reconnection sites
- **Slow modes**: Compressional/acoustic modes
- **Mixed modes**: Coupling between different wave types

## Notes

1. **Eigenvector normalization is essential** - ensures fair comparison between modes
2. **Formula interpretation**: `Abs[...]^2/2 = |...|` NOT `(Abs[...])²` / 2`
3. **M11, M22 extracted at theta=0** - corresponds to symmetric point in periodic orbit
4. **All 4 eigenvectors analyzed** - each Floquet mode gets alfvenicity value
5. **Range**: [0, 1] as it's a normalized ratio

## Files Modified

- `include/Floquet_two_dof.h`:
  - Added `alfvenicity()` function (line ~550)
  - Modified `eigenvector()` to return unit-normed vectors
- `src/test_two_dof.cpp`:
  - Added alfvenicity test cases
  - Integrated into omega scan

## Date: 2025

