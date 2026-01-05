# File Renaming Summary

## Date
2026-01-04

## Changes
Renamed all files containing "resonance" and removed "test" prefix from executables.

## Files Renamed

### Main Directory
| Old Name | New Name |
|-----------|-----------|
| plot_resonance_continuum.py | plot_continuum.py |
| resonance_continuum_gamma5over3.csv | continuum_gamma5over3.csv |
| resonance_continuum.csv | continuum.csv |
| src/test_resonance_continuum.cpp | src/continuum.cpp |
| test_resonance_continuum | continuum |
| resonance_continuum.log | continuum.log |
| resonance_continuum.csv | resonance_continuum_old.csv.bak (backup) |

### test_gamma Directory
| Old Name | New Name |
|-----------|-----------|
| plot_resonance_continuum.py | plot_continuum.py |
| resonance_continuum.csv | continuum.csv |
| test_resonance_continuum | continuum |
| gamma100/resonance_continuum.csv | gamma100/continuum.csv |
| gamma5Over3/resonance_continuum.csv | gamma5Over3/continuum.csv |

### Backup Files
- plot_continuum_old.py (previous version of plot script)
- resonance_continuum_old.csv.bak (old CSV backup)

## CMakeLists.txt Changes

Updated target name from `test_resonance_continuum` to `continuum`:
```cmake
# BEFORE:
add_executable(test_resonance_continuum src/test_resonance_continuum.cpp src/gFileRawData.cpp)
target_include_directories(test_resonance_continuum PRIVATE ...)

# AFTER:
add_executable(continuum src/continuum.cpp src/gFileRawData.cpp)
target_include_directories(continuum PRIVATE ...)
```

## plot_continuum.py Changes

Updated to reference new filenames:
```python
# Line 3: Updated docstring
"Plot continuum results from continuum"

# Line 168: Updated CSV filename
csv_file = "continuum.csv"

# Line 173: Updated error message
print("Please run continuum first to generate data.")
```

## Usage Updates

### Old Usage
```bash
./test_resonance_continuum gfile_131041 100 1.6667
python3 plot_resonance_continuum.py
```

### New Usage
```bash
./continuum gfile_131041 100 1.6667
python3 plot_continuum.py
```

## Verification

All executables built and tested successfully:
- ✓ `continuum` executable works correctly
- ✓ `plot_continuum.py` generates plots from `continuum.csv`
- ✓ All CMake targets build without errors

## Not Changed
Reference data directories kept unchanged:
- gyweiEQ/
- gyweiEQGamma3/
- gyweiEQGamma5Over3/
- gyweiEQGamma10/
- gyweiEQGamma100/

These contain reference data from GY Wei and were not modified.
