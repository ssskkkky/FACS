# Removal of Test Source Files

## Date
2026-01-05

## Files Removed

### Source Files
| File | Reason for Removal |
|------|---------------------|
| src/test_equilibrium_list.cpp | Outdated test for equilibrium surface listing |
| src/test_single_surface.cpp | Outdated test for single surface continuum |
| src/test_two_dof.cpp | Outdated test for two-DOF eigenvalue benchmarking |

### Corresponding Executables
| Executable | Status |
|-----------|--------|
| test_equilibrium_list | No longer built |
| test_single_surface | No longer built |
| test_two_dof | No longer built |

## CMakeLists.txt Changes

### Targets Removed
```cmake
# REMOVED:
add_executable(test_two_dof src/test_two_dof.cpp src/gFileRawData.cpp)
target_include_directories(test_two_dof PRIVATE ...)

add_executable(test_equilibrium_list src/test_equilibrium_list.cpp src/gFileRawData.cpp)
target_include_directories(test_equilibrium_list PRIVATE ...)

add_executable(test_single_surface src/test_single_surface.cpp src/gFileRawData.cpp)
target_include_directories(test_single_surface PRIVATE ...)
```

### Remaining Targets
```cmake
# RETAINED:
add_executable(facs src/main.cpp src/gFileRawData.cpp)
target_include_directories(facs PRIVATE ...)

add_executable(continuum_with_sound_coupling src/continuum_with_sound_coupling.cpp src/gFileRawData.cpp)
target_include_directories(continuum_with_sound_coupling PRIVATE ...)

add_executable(test_interpolation src/test_interpolation.cpp src/gFileRawData.cpp)
target_include_directories(test_interpolation PRIVATE ...)
```

## Remaining Source Files

| File | Purpose |
|------|----------|
| src/main.cpp | Main FACS application |
| src/gFileRawData.cpp | G-file data parsing |
| src/continuum_with_sound_coupling.cpp | Continuum calculation with sound coupling |
| src/test_interpolation.cpp | Interpolation test and benchmark |

## Rationale

### Why These Files Were Removed

1. **test_equilibrium_list.cpp**:
   - Lists equilibrium surfaces
   - Functionality superseded by `continuum_with_sound_coupling`
   - No longer actively used

2. **test_single_surface.cpp**:
   - Computes continuum for a single surface
   - Superseded by `continuum_with_sound_coupling` (batch processing)
   - Limited utility for production use

3. **test_two_dof.cpp**:
   - Benchmarks eigenvalue solvers (QR vs Durand-Kerner)
   - Algorithm choice resolved in favor of QR
   - Documentation already exists in `docs/degenerate_eigenvalue_analysis.md`
   - No longer needed for validation

### Retained Files Justification

1. **continuum_with_sound_coupling.cpp**:
   - Primary production tool for continuum calculation
   - Batch processing of multiple surfaces
   - Generates CSV output for plotting

2. **test_interpolation.cpp**:
   - Tests and benchmarks interpolation functions
   - Validates equilibrium data access
   - Useful for performance testing

3. **facs (main.cpp)**:
   - Main application entry point
   - Provides interface for Floquet continuum calculation
   - Core functionality

## Build Verification

### Before Removal
```bash
make -j4
# Targets: facs, test_two_dof, test_equilibrium_list,
#          test_single_surface, test_interpolation, continuum_with_sound_coupling
```

### After Removal
```bash
make clean && cmake -DCMAKE_BUILD_TYPE=Release . && make -j4
# Targets: facs, test_interpolation, continuum_with_sound_coupling
```

### Test Results
All remaining executables built and tested successfully:
```bash
$ ./facs gfile_131041
# ✓ Works correctly

$ ./continuum_with_sound_coupling gfile_131041 100 1.6667
# ✓ Works correctly

$ ./test_interpolation gfile_131041 0.021
# ✓ Works correctly
```

## Documentation Updates

### Affected Documentation
Documentation referencing removed files should be updated:
- `docs/code_cleanup_summary.md` - References test_two_dof.cpp
- `docs/degenerate_eigenvalue_analysis.md` - References test_two_dof.cpp
- `docs/alfvenicity_implementation.md` - References test_two_dof.cpp

### Recommended Action
Update documentation to reference `continuum_with_sound_coupling` or `test_interpolation` instead of removed test files.

## Impact on Codebase

### Benefits
1. **Reduced complexity**: Fewer test targets to maintain
2. **Clearer purpose**: Main production tool is `continuum_with_sound_coupling`
3. **Reduced build time**: Fewer source files to compile
4. **Better organization**: Only essential test files retained

### No Breaking Changes
- All functionality preserved through retained files
- No API changes
- No dependency on removed files in remaining code

## Source Tree After Cleanup

```
src/
├── main.cpp                                    (retained)
├── gFileRawData.cpp                           (retained)
├── continuum_with_sound_coupling.cpp             (retained)
└── test_interpolation.cpp                       (retained)
```

## Notes

The removed test files served their purpose during development:
- Validated eigenvalue algorithms
- Tested equilibrium data structures
- Provided debugging tools

Their functionality is now captured in:
- `docs/degenerate_eigenvalue_analysis.md` (QR vs Durand-Kerner comparison)
- `continuum_with_sound_coupling.cpp` (production continuum tool)
- `test_interpolation.cpp` (interpolation validation)
