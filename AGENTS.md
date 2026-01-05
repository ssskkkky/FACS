# FACS - Floquet Alfven Continuum Solver

Version: 0.4

This is a C++ application for computing Alfven continuum in plasma physics using Floquet theory, based on the work by Falessi 2019.

## Project Structure

- `src/main.cpp`: Main entry point, reads g-file, computes equilibrium, calculates Floquet exponents, and outputs continuum data.
- `src/gFileRawData.cpp`: Handles parsing of g-file data.
- `src/continuum_with_sound_coupling.cpp`: Continuum solver with sound coupling and multiple n values.
- `src/test_interpolation.cpp`: Interpolation test program.
- `include/`: Header files for classes like Equilibrium, Floquet, Integrator, TwoDof, etc.
- `CMakeLists.txt`: Build configuration with C++20, ASan in debug mode.

## Build Instructions

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

Executables: `facs`, `continuum_with_sound_coupling`, `test_interpolation`

Usage:
```bash
./facs <path_to_gfile>
./continuum_with_sound_coupling <path_to_gfile> <num_surfaces> <n_values>
./test_interpolation <path_to_gfile>
```

where `<n_values>` is a comma-separated list of n values (e.g., "5,10,15"). Gamma is fixed to 5/3.

## Dependencies

- C++20 compatible compiler (gcc, clang)
- CMake 3.15+
- Standard libraries: filesystem, complex, etc.

## Coding Patterns

- Modern C++: uses namespaces, auto, lambdas, ranges.
- Error handling with errno codes.
- Timer class for profiling.
- Complex<double> for Floquet math.
- Vectors and maps for data structures.
- Constants like EPSILON for floating point comparisons.
- Header functions marked `inline` to avoid ODR violations.

## Eigenvalue Solver

**Primary: QR Algorithm**
- Robust for all matrix types including degenerate cases
- Uses Hessenberg reduction and QR iterations with Wilkinson shift
- Handles near-degenerate and ill-conditioned matrices correctly
- Performance: ~94 μs average

**Reference: Durand-Kerner**
- Polynomial root finding method
- ~12% faster (84 μs average) but fails on degenerate cases
- Used for unit testing and comparison only
- See `docs/degenerate_eigenvalue_analysis.md` for details

## Agents/Commands

- Lint: Not specified (suggest clang-tidy if desired)
- Typecheck: Build with cmake to check compilation

## Changelog

### v0.4 (2026)
- Added continuum_with_sound_coupling solver with multiple n support
- Removed m variable from continuum_with_sound_coupling
- Gamma fixed to 5/3 in continuum_with_sound_coupling
- Optimized EigenwaveAnalyzer computation (once per flux surface)
- Updated plotting to support color-coding by n value
- CSV output now includes n column and nq (was nqm)

### v0.3 (2025)
- Switched to QR Algorithm for Floquet multipliers (robust)
- Added accuracy verification using characteristic polynomial residuals
- Fixed ODR violations by marking header functions as `inline`
- Added documentation on degenerate eigenvalue handling
- Removed unused headers and cleaned up code

### v0.2
- Initial two-DOF continuum solver implementation
- Durand-Kerner eigenvalue computation
- RK4 integration for monodromy matrix