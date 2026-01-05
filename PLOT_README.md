# Plotting Scripts

## plot_continuum.py - Continuum with Sound Coupling

Plots results from `continuum_with_sound_coupling` program.

### Usage

```bash
python3 plot_continuum.py
```

### Prerequisites

1. Run `continuum_with_sound_coupling` first to generate `continuum_with_sound_coupling.csv`:
   ```bash
   ./continuum_with_sound_coupling <gfile> <num_surfaces> <n_values>
   ```

2. Required Python packages:
   - numpy
   - matplotlib
   - csv (standard library)
   - ast (standard library)

### Generated Plots

| File | Description |
|------|-------------|
| `continuum_with_sound_coupling_vs_psi.png` | 3-panel plot: Sound continuum, Alfven continuum, Combined (color by n) |

### Plot Contents

**continuum_with_sound_coupling_vs_psi.png (3 panels):**

- **Top Left:** Sound wave continuum - Omega vs minor_radius (color-coded by n value)
- **Top Right:** Alfvén wave continuum - Omega vs minor_radius (color-coded by n value)
- **Bottom:** Combined Sound + Alfvén continuum - Omega vs minor_radius (color by n, grayscale by alfvenicity)

When multiple n values are provided, each n is assigned a different color and a legend is displayed.

### Output Statistics

The script prints:
- Configuration (n values, number of surfaces)
- Ranges of minor radius
- Branch statistics (sound and Alfvén points plotted)

### Example Output

```
================================================================================
                    CONTINUUM ANALYSIS
================================================================================

[CONFIGURATION]
  Mode numbers: n = 5, 10
  Total surfaces analyzed: 20
  Total n values: 2

[X-AXIS: Minor radius r/a (equally spaced)]
  Range: [0.0500, 0.9500]

[GENERATING PLOTS]
Debug: Plotted 121 sound points, 59 alfven points in combined plot
Saved: continuum_with_sound_coupling_vs_psi.png

================================================================================
All plots generated successfully!
X-axis: minor_radius (equally spaced)
================================================================================
```
