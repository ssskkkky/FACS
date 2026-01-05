# Build Instructions for FACS Project

## Quick Build

```bash
# Configure with cmake
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build all targets
cd build
make -j$(nproc)
```

## Built Executables

After building, the following executables are created in the `build/` directory:

| Executable | Description | Usage |
|-----------|-------------|---------|
| `facs` | Main FACS solver | `./build/facs <gfile>` |
| `continuum_with_sound_coupling` | Continuum with sound coupling | `./build/continuum_with_sound_coupling <gfile> <num_surfaces> <n_values>` |
| `test_interpolation` | Interpolation test | `./build/test_interpolation <gfile>` |

## Examples

```bash
# Run main FACS solver
./build/facs gfile_131041

# Run continuum with sound coupling (10 surfaces, n=5,10)
./build/continuum_with_sound_coupling gfile_131041 10 5,10

# Run interpolation test
./build/test_interpolation gfile_131041

# Plot continuum with sound coupling results
python3 plot_continuum.py
```

## Clean Build

```bash
# Remove build directory
rm -rf build

# Clean cmake cache
rm -rf CMakeCache.txt CMakeFiles/
```

## Build Configuration

- **C++ Standard:** C++20
- **CMake Version:** 3.15+
- **Default Build Type:** Release
- **Parallel Build:** Enabled (use `make -j$(nproc)`)

## Dependencies

- C++20 compatible compiler (gcc, clang)
- CMake 3.15+
- Standard C++ libraries (filesystem, complex, etc.)
