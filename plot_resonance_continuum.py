#!/usr/bin/env python3
"""
Plot resonance continuum results from test_resonance_continuum
X-axis: minor_radius (r/a) - equally spaced
"""

import csv
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def read_resonance_data(filename):
    """Read resonance continuum data from CSV file"""
    data = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sound_str = row["sound_omegas"].replace(";", ",")
            sound_alfv_str = row["sound_alfvenicities"].replace(";", ",")
            alfven_str = row["alfven_omegas"].replace(";", ",")
            alfven_alfv_str = row["alfven_alfvenicities"].replace(";", ",")

            try:
                sound_omegas = ast.literal_eval(sound_str.strip('"'))
                sound_alfvenicities = ast.literal_eval(sound_alfv_str.strip('"'))
                alfven_omegas = ast.literal_eval(alfven_str.strip('"'))
                alfven_alfvenicities = ast.literal_eval(alfven_alfv_str.strip('"'))
            except Exception as e:
                print(f"Error parsing row: {e}")
                sound_omegas = []
                sound_alfvenicities = []
                alfven_omegas = []
                alfven_alfvenicities = []

            data.append(
                {
                    "index": int(row["index"]),
                    "minor_radius": float(row["minor_radius"]),
                    "psi": float(row["psi"]),
                    "q": float(row["q"]),
                    "nqm": float(row["nqm"]),
                    "sound_branches": int(row["sound_branches"]),
                    "alfven_branches": int(row["alfven_branches"]),
                    "sound_omegas": sound_omegas,
                    "sound_alfvenicities": sound_alfvenicities,
                    "alfven_omegas": alfven_omegas,
                    "alfven_alfvenicities": alfven_alfvenicities,
                }
            )
    return data


def plot_continuum_vs_minor_radius(data, output_file="continuum_vs_psi.png"):
    """Create Omega vs minor_radius plots"""

    r_vals = [d["minor_radius"] for d in data]
    nqm_vals = [d["nqm"] for d in data]

    # Find resonance surface (closest to n*q-m = 0)
    res_idx = min(range(len(nqm_vals)), key=lambda i: abs(nqm_vals[i]))
    res_r = r_vals[res_idx]
    res_psi = data[res_idx]["psi"]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: Sound wave continuum
    ax1 = fig.add_subplot(gs[0, 0])
    for d in data:
        for omega in d["sound_omegas"]:
            ax1.scatter(
                d["minor_radius"],
                omega,
                c="blue",
                s=50,
                alpha=0.6,
                edgecolors="white",
                linewidth=0.5,
                zorder=2,
            )

    ax1.axvline(
        x=res_r,
        color="red",
        linestyle="--",
        linewidth=3,
        label=f"Resonance r/a={res_r:.5f}",
        alpha=0.7,
        zorder=1,
    )
    ax1.set_xlabel("Minor radius r/a", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Omega (frequency)", fontsize=14, fontweight="bold")
    ax1.set_title("Sound Wave Continuum: Omega vs r/a", fontsize=15, fontweight="bold")
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Alfven wave continuum
    ax2 = fig.add_subplot(gs[0, 1])
    for d in data:
        for omega in d["alfven_omegas"]:
            ax2.scatter(
                d["minor_radius"],
                omega,
                c="orange",
                s=50,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
                zorder=2,
            )

    ax2.axvline(
        x=res_r,
        color="red",
        linestyle="--",
        linewidth=3,
        label="Resonance",
        alpha=0.7,
        zorder=1,
    )
    ax2.set_xlabel("Minor radius r/a", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Omega", fontsize=14, fontweight="bold")
    ax2.set_title("Alfvén Wave Continuum: Omega vs r/a", fontsize=15, fontweight="bold")
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Combined continuum, graylevel by alfvenicity (1=dark, 0=invisible)
    ax3 = fig.add_subplot(gs[1, 0])
    sound_count = 0
    alfven_count = 0
    all_r = []
    all_omega = []
    all_alpha = []

    for d in data:
        # Sound waves with alfvenicity
        for i, omega in enumerate(d["sound_omegas"]):
            if i < len(d.get("sound_alfvenicities", [])):
                alfvenicity = min(max(abs(d["sound_alfvenicities"][i]), 0.0), 1.0)
                all_r.append(d["minor_radius"])
                all_omega.append(omega)
                all_alpha.append(alfvenicity)
                sound_count += 1

        # Alfven waves with alfvenicity
        for i, omega in enumerate(d["alfven_omegas"]):
            if i < len(d.get("alfven_alfvenicities", [])):
                alfvenicity = min(max(abs(d["alfven_alfvenicities"][i]), 0.0), 1.0)
                all_r.append(d["minor_radius"])
                all_omega.append(omega)
                all_alpha.append(alfvenicity)
                alfven_count += 1

    print(
        f"Debug: Plotted {sound_count} sound points, {alfven_count} alfven points in combined plot"
    )

    ax3.scatter(
        all_r, all_omega, c="black", s=50, alpha=all_alpha, edgecolors="none", zorder=2
    )

    ax3.axvline(
        x=res_r,
        color="red",
        linestyle="--",
        linewidth=3,
        label="Resonance",
        alpha=0.7,
        zorder=1,
    )
    ax3.set_xlabel("Minor radius r/a", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Omega", fontsize=13, fontweight="bold")
    ax3.set_title(
        "Combined Continuum (grayscale by alfvenicity)", fontsize=14, fontweight="bold"
    )
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Plot 4: n*q-m vs r/a
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(r_vals, nqm_vals, "g-o", linewidth=3, markersize=10, label="n*q - m")
    ax4.axhline(
        y=0,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Resonance (n*q-m=0)",
        alpha=0.7,
    )
    ax4.axvline(
        x=res_r,
        color="purple",
        linestyle=":",
        linewidth=2,
        alpha=0.7,
        label="Resonance surface",
    )
    ax4.scatter(
        [res_r],
        [nqm_vals[res_idx]],
        s=500,
        c="red",
        marker="*",
        edgecolors="black",
        linewidth=2,
        zorder=3,
    )
    ax4.set_xlabel("Minor radius r/a", fontsize=13, fontweight="bold")
    ax4.set_ylabel("n*q - m (n=5, m=10)", fontsize=13, fontweight="bold")
    ax4.set_title("Resonance Parameter Profile", fontsize=14, fontweight="bold")
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")
    return fig


def main():
    csv_file = "resonance_continuum.csv"
    try:
        data = read_resonance_data(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        print("Please run test_resonance_continuum first to generate the data.")
        return

    if len(data) == 0:
        print(f"Error: No data found in {csv_file}")
        return

    r_vals = [d["minor_radius"] for d in data]

    print("=" * 80)
    print("                    RESONANCE CONTINUUM ANALYSIS                    ")
    print("=" * 80)
    print(f"\n[CONFIGURATION]")
    print(f"  Mode numbers: n=5, m=10")
    print(f"  Resonance condition: n*q - m = 0  →  q = m/n = 2.0")
    print(f"  Total surfaces analyzed: {len(data)}")
    print(f"\n[X-AXIS: Minor radius r/a (equally spaced)]")
    print(f"  Range: [{min(r_vals):.4f}, {max(r_vals):.4f}]")
    print(f"  Spacing: {r_vals[1] - r_vals[0]:.4f}")

    nqm_vals = [d["nqm"] for d in data]
    res_idx = min(range(len(nqm_vals)), key=lambda i: abs(nqm_vals[i]))
    res_data = data[res_idx]

    print(f"\n[RESONANCE SURFACE]")
    print(f"  Index:             {res_idx}")
    print(f"  r/a:              {res_data['minor_radius']:.6f}")
    print(f"  Psi:              {res_data['psi']:.6f}")
    print(f"  Safety factor q:   {res_data['q']:.6f}")
    print(f"  n*q - m:          {res_data['nqm']:+.6f}")

    print("\n[GENERATING PLOTS]")
    plot_continuum_vs_minor_radius(data)

    print("\n" + "=" * 80)
    print("All plots generated successfully!")
    print("X-axis: minor_radius (equally spaced)")
    print("=" * 80)


if __name__ == "__main__":
    main()
