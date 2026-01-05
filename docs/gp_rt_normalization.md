# gp_rt Normalization Update

## Summary

Successfully normalized `gp_rt` (intp_2d[5]) in `spdata.h` to use the same normalization as `F_psi` (intp_1d[1]).

## Changes Made

### 1. Code Change (spdata.h:386)
```cpp
// BEFORE:
gp_rt_boozer_grid.push_back(gp_rt_val);
gp_rt_boozer(ri, i) = gp_rt_val;

// AFTER:
gp_rt_boozer_grid.push_back(gp_rt_val / current_unit);
gp_rt_boozer(ri, i) = gp_rt_val / current_unit;
```

### 2. Comment Updates (spdata.h:39 and spdata.h:63)
Updated documentation to reflect that `gp_rt` is now normalized by `current_unit = R0 * B0`.

## Before vs After

| Quantity | Before | After | Units |
|----------|---------|--------|-------|
| F_psi (normalized) | 1.004 | 1.004 | [R0*B0] |
| gp_rt (raw) | 3.98 - 8.46 | - | [T·m] |
| gp_rt (normalized) | - | 0.12 - 0.26 | [R0*B0] |

**Ratio F_psi/gp_rt:**
- Before: 0.12 - 0.25 (weak coupling)
- After: 3.85 - 8.19 (corrected coupling)

## Physical Interpretation

Both `F_psi` and `gp_rt` now use consistent normalization:

```cpp
current_unit = R0 * B0  (units: [T·m])

Normalized F_psi = F_psi_physical / (R0 * B0)
Normalized gp_rt = |∇ψ|_physical / (R0 * B0)
```

This ensures:
1. ✓ Same physical dimensions [T·m]
2. ✓ Consistent normalization scheme
3. ✓ Both values are O(1) in normalized units
4. ✓ Corrected coupling strength in continuum equation

## Impact on Continuum Equation

Coupling term in `Floquet_two_dof.h:150`:
```cpp
off_diag = 2 * J * F_psi / gp_rt * dB_dtheta_over_B * omega
```

**Physical effect:**
- Ratio `F_psi/gp_rt` now correctly represents physical coupling
- Previously underestimated (ratio too small)
- Now properly scales with R0 and B0

## Verification

Tested with `test_units.cpp` at ψ = 0.02128:

| Theta | F_psi | gp_rt | F_psi/gp_rt |
|--------|---------|---------|--------------|
| 0.0 | 1.004 | 0.261 | 3.85 |
| π/4 | 1.004 | 0.168 | 5.97 |
| π/2 | 1.004 | 0.123 | 8.19 |
| π | 1.004 | 0.163 | 6.15 |

All executables tested and working correctly:
- ✓ facs
- ✓ test_resonance_continuum
- ✓ test_two_dof
- ✓ test_equilibrium_list
- ✓ test_single_surface
- ✓ test_interpolation

## Date
2026-01-04
