# continuum_with_sound_coupling Renaming Summary

## Date
2026-01-05

## Changes
Renamed `continuum` to `continuum_with_sound_coupling` with corresponding data and plot file updates.

## Files Renamed

### Source Files
| Old Name | New Name |
|----------|-----------|
| src/continuum.cpp | src/continuum_with_sound_coupling.cpp |

### Executables
| Old Name | New Name |
|----------|-----------|
| continuum | continuum_with_sound_coupling |

### Data Files
| Old Name | New Name |
|----------|-----------|
| continuum.csv | continuum_with_sound_coupling.csv |

### Plot Files
| Old Name | New Name |
|----------|-----------|
| continuum_vs_psi.png | continuum_with_sound_coupling_vs_psi.png |
| continuum.log | continuum_with_sound_coupling.log |

## Code Changes

### CMakeLists.txt
```cmake
# BEFORE:
add_executable(continuum src/continuum.cpp src/gFileRawData.cpp)
target_include_directories(continuum PRIVATE ...)

# AFTER:
add_executable(continuum_with_sound_coupling src/continuum_with_sound_coupling.cpp src/gFileRawData.cpp)
target_include_directories(continuum_with_sound_coupling PRIVATE ...)
```

### src/continuum_with_sound_coupling.cpp
- Line 62: Output CSV filename
  ```cpp
  // BEFORE:
  std::ofstream out("continuum.csv");

  // AFTER:
  std::ofstream out("continuum_with_sound_coupling.csv");
  ```

- Line 176: Console message
  ```cpp
  // BEFORE:
  std::cout << "Data saved to continuum.csv\n";

  // AFTER:
  std::cout << "Data saved to continuum_with_sound_coupling.csv\n";
  ```

### plot_continuum.py
- Line 3: Updated docstring
  ```python
  # BEFORE:
  "Plot continuum results from continuum"

  # AFTER:
  "Plot continuum results from continuum_with_sound_coupling"
  ```

- Line 168: CSV filename
  ```python
  # BEFORE:
  csv_file = "continuum.csv"

  # AFTER:
  csv_file = "continuum_with_sound_coupling.csv"
  ```

- Line 55: Output plot filename
  ```python
  # BEFORE:
  output_file="continuum_vs_psi.png"

  # AFTER:
  output_file="continuum_with_sound_coupling_vs_psi.png"
  ```

- Line 173: Error message
  ```python
  # BEFORE:
  print("Please run continuum first to generate data.")

  # AFTER:
  print("Please run continuum_with_sound_coupling first to generate data.")
  ```

## Usage Updates

### Old Usage
```bash
./continuum gfile_131041 100 1.6667
python3 plot_continuum.py
```

### New Usage
```bash
./continuum_with_sound_coupling gfile_131041 100 1.6667
python3 plot_continuum.py
```

## Files Generated

After running `./continuum_with_sound_coupling`:
- `continuum_with_sound_coupling.csv` (data file)

After running `python3 plot_continuum.py`:
- `continuum_with_sound_coupling_vs_psi.png` (plot file)

## Verification

Tested with 2 surfaces:
```bash
$ ./continuum_with_sound_coupling gfile_131041 2 1.6667
Loaded g-file: gfile_131041
...
=== RESULTS SAVED ===
Data saved to continuum_with_sound_coupling.csv

$ python3 plot_continuum.py
...
Saved: continuum_with_sound_coupling_vs_psi.png
```

✓ All executables built successfully
✓ CSV file generated correctly
✓ Plot file generated correctly
✓ No references to old filenames remain

## Backup Files

Old versions kept for reference:
- `continuum.csv` (previous data)
- `continuum_vs_psi.png` (previous plot)

## Notes

The renaming clarifies that this tool computes continuum **with sound coupling**, distinguishing it from other possible continuum calculations (e.g., pure Alfven continuum, or continuum with different coupling models).

All other test executables remain unchanged:
- `test_two_dof`
- `test_equilibrium_list`
- `test_single_surface`
- `test_interpolation`
- `facs`
