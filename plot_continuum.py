#!/usr/bin/env python3
"""
Plot continuum results from continuum_with_sound_coupling
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
                    "n": int(row["n"]),
                    "index": int(row["index"]),
                    "minor_radius": float(row["minor_radius"]),
                    "psi": float(row["psi"]),
                    "q": float(row["q"]),
                    "nq": float(row["nq"]),
                    "sound_branches": int(row["sound_branches"]),
                    "alfven_branches": int(row["alfven_branches"]),
                    "sound_omegas": sound_omegas,
                    "sound_alfvenicities": sound_alfvenicities,
                    "alfven_omegas": alfven_omegas,
                    "alfven_alfvenicities": alfven_alfvenicities,
                }
            )
    return data


def plot_continuum_vs_minor_radius(
    data, output_file="continuum_with_sound_coupling_vs_psi.png"
):
    """Create Omega vs minor_radius plots"""

    n_vals = sorted(list(set([d["n"] for d in data])))
    colors = plt.cm.tab10(np.linspace(0, 1, len(n_vals)))

    r_vals = [d["minor_radius"] for d in data]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: Sound wave continuum
    ax1 = fig.add_subplot(gs[0, 0])
    for idx, n in enumerate(n_vals):
        sound_r = []
        sound_omega = []
        for d in data:
            if d["n"] == n:
                for omega in d["sound_omegas"]:
                    sound_r.append(d["minor_radius"])
                    sound_omega.append(omega)
        ax1.scatter(
            sound_r,
            sound_omega,
            c=[colors[idx]],
            s=10,
            alpha=0.6,
            edgecolors="none",
            linewidth=0,
            zorder=2,
            label=f"n={n}",
        )
    ax1.set_xlabel("Minor radius r/a", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Omega (frequency)", fontsize=14, fontweight="bold")
    ax1.set_title("Sound Wave Continuum: Omega vs r/a", fontsize=15, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    if len(n_vals) > 1:
        ax1.legend(fontsize=10)

    # Plot 2: Alfven wave continuum
    ax2 = fig.add_subplot(gs[0, 1])
    for idx, n in enumerate(n_vals):
        alfven_r = []
        alfven_omega = []
        for d in data:
            if d["n"] == n:
                for omega in d["alfven_omegas"]:
                    alfven_r.append(d["minor_radius"])
                    alfven_omega.append(omega)
        ax2.scatter(
            alfven_r,
            alfven_omega,
            c=[colors[idx]],
            s=10,
            alpha=0.7,
            edgecolors="none",
            linewidth=0,
            zorder=2,
            label=f"n={n}",
        )
    ax2.set_xlabel("Minor radius r/a", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Omega", fontsize=14, fontweight="bold")
    ax2.set_title("Alfvén Wave Continuum: Omega vs r/a", fontsize=15, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    if len(n_vals) > 1:
        ax2.legend(fontsize=10)

    # Plot 3: Combined continuum, color by n, graylevel by alfvenicity (1=dark, 0.1=light)
    ax3 = fig.add_subplot(gs[1, 0])
    sound_count = 0
    alfven_count = 0
    all_r = []
    all_omega = []
    all_alpha = []
    all_colors = []

    for idx, n in enumerate(n_vals):
        for d in data:
            if d["n"] == n:
                # Sound waves with alfvenicity
                for i, omega in enumerate(d["sound_omegas"]):
                    if i < len(d.get("sound_alfvenicities", [])):
                        alfvenicity = min(
                            max(abs(d["sound_alfvenicities"][i]), 0.0), 1.0
                        )
                        all_r.append(d["minor_radius"])
                        all_omega.append(omega)
                        all_alpha.append(alfvenicity)
                        all_colors.append(colors[idx])
                        sound_count += 1

                # Alfven waves with alfvenicity
                for i, omega in enumerate(d["alfven_omegas"]):
                    if i < len(d.get("alfven_alfvenicities", [])):
                        alfvenicity = min(
                            max(abs(d["alfven_alfvenicities"][i]), 0.0), 1.0
                        )
                        all_r.append(d["minor_radius"])
                        all_omega.append(omega)
                        all_alpha.append(alfvenicity)
                        all_colors.append(colors[idx])
                        alfven_count += 1

    print(
        f"Debug: Plotted {sound_count} sound points, {alfven_count} alfven points in combined plot"
    )

    # Rescale alpha to 0.1-1.0 range so sound waves are visible
    all_alpha_scaled = [0.1 + 0.9 * alpha for alpha in all_alpha]

    ax3.scatter(
        all_r,
        all_omega,
        c=all_colors,
        s=10,
        alpha=all_alpha_scaled,
        edgecolors="none",
        zorder=2,
    )

    ax3.set_xlabel("Minor radius r/a", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Omega", fontsize=13, fontweight="bold")
    ax3.set_title(
        "Combined Continuum (color by n, grayscale 0.1-1.0 by alfvenicity)",
        fontsize=14,
        fontweight="bold",
    )
    ax3.grid(True, alpha=0.3)
    if len(n_vals) > 1:
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[idx],
                markersize=8,
                label=f"n={n}",
            )
            for idx, n in enumerate(n_vals)
        ]
        ax3.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")
    return fig


def main():
    csv_file = "continuum_with_sound_coupling.csv"
    try:
        data = read_resonance_data(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        print("Please run continuum_with_sound_coupling first to generate data.")
        return

    if len(data) == 0:
        print(f"Error: No data found in {csv_file}")
        return

    n_vals = sorted(list(set([d["n"] for d in data])))
    r_vals = [d["minor_radius"] for d in data]

    print("=" * 80)
    print("                    CONTINUUM ANALYSIS                    ")
    print("=" * 80)
    print(f"\n[CONFIGURATION]")
    print(f"  Mode numbers: n = {', '.join(map(str, n_vals))}")
    print(f"  Total surfaces analyzed: {len(data)}")
    print(f"  Total n values: {len(n_vals)}")
    print(f"\n[X-AXIS: Minor radius r/a (equally spaced)]")
    print(f"  Range: [{min(r_vals):.4f}, {max(r_vals):.4f}]")
    print(f"  Spacing: {r_vals[1] - r_vals[0]:.4f}")

    print("\n[GENERATING PLOTS]")
    plot_continuum_vs_minor_radius(data)

    print("\n" + "=" * 80)
    print("All plots generated successfully!")
    print("X-axis: minor_radius (equally spaced)")
    print("=" * 80)


if __name__ == "__main__":
    main()
