# Header File Renaming: two_dof_continuum.h → Floquet_two_dof.h

## Date
2026-01-05

## Changes
Renamed `include/two_dof_continuum.h` to `include/Floquet_two_dof.h` to better reflect its purpose: Floquet theory for two-degree-of-freedom continuum calculation.

## Files Renamed

| Old Name | New Name |
|-----------|-----------|
| include/two_dof_continuum.h | include/Floquet_two_dof.h |

## Files Updated (Include Statements)

All files that included `two_dof_continuum.h` updated to use `Floquet_two_dof.h`:

### Source Files
| File | Old Include | New Include |
|------|-------------|-------------|
| src/continuum_with_sound_coupling.cpp | `#include "../include/two_dof_continuum.h"` | `#include "../include/Floquet_two_dof.h"` |
| src/test_two_dof.cpp | `#include "../include/two_dof_continuum.h"` | `#include "../include/Floquet_two_dof.h"` |
| src/test_single_surface.cpp | `#include "../include/two_dof_continuum.h"` | `#include "../include/Floquet_two_dof.h"` |
| src/test_interpolation.cpp | `#include "../include/two_dof_continuum.h"` | `#include "../include/Floquet_two_dof.h"` |

### Test Files
| File | Old Include | New Include |
|------|-------------|-------------|
| test_degenerate.cpp | `#include "../include/two_dof_continuum.h"` | `#include "include/Floquet_two_dof.h"` |
| test_filtering.cpp | `#include "include/two_dof_continuum.h"` | `#include "include/Floquet_two_dof.h"` |

### Documentation Files
| File | Line Updated |
|------|---------------|
| docs/degenerate_eigenvalue_analysis.md | Line 145 |
| docs/alfvenicity_implementation.md | Line 67, 124 |
| docs/code_cleanup_summary.md | Line 11, 82 |
| docs/gp_rt_normalization.md | Line 54 |

## Rationale

The new name `Floquet_two_dof.h` better describes the header file's purpose:

- **Floquet**: Uses Floquet theory for periodic systems
- **two_dof**: Two-degree-of-freedom model
- **Consistent naming**: Aligns with other Floquet-related code (e.g., Floquet_two_dof.h is now the standard name)

## Verification

All executables built successfully after renaming:
```bash
make clean && cmake -DCMAKE_BUILD_TYPE=Release . && make -j4
# Result: All targets built successfully
```

### Test Results
- ✓ `continuum_with_sound_coupling` compiles and runs correctly
- ✓ `test_two_dof` compiles and runs correctly
- ✓ `test_single_surface` compiles and runs correctly
- ✓ `test_interpolation` compiles and runs correctly
- ✓ `test_equilibrium_list` compiles and runs correctly
- ✓ `facs` compiles and runs correctly

## No Breaking Changes

- All include paths updated consistently
- No changes to header file content
- CMakeLists.txt does not reference header file directly (uses source files only)
- All functionality preserved

## Build Status

```bash
$ make -j4
[ 11%] Building CXX object CMakeFiles/test_two_dof.dir/src/test_two_dof.cpp.o
[ 11%] Building CXX object CMakeFiles/facs.dir/src/main.cpp.o
[ 22%] Building CXX object CMakeFiles/continuum_with_sound_coupling.dir/src/continuum_with_sound_coupling.cpp.o
[ 22%] Building CXX object CMakeFiles/test_equilibrium_list.dir/src/test_equilibrium_list.cpp.o
...
[100%] Linking CXX executable test_interpolation
[100%] Built target test_interpolation
```

All 6 executables built successfully.

## Notes

The old name `two_dof_continuum.h` was descriptive but could be confused with:
- Generic continuum solvers
- Other multi-degree-of-freedom systems
- Not emphasizing the Floquet method

The new name `Floquet_two_dof.h` is more specific and informative.
