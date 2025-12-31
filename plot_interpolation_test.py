#!/usr/bin/env python3
"""
Plot interpolation_test.csv showing Floquet continuum
"""

import csv
import matplotlib.pyplot as plt
import numpy as np


def read_interpolation_data(filename):
    """Read interpolation test data"""
    data = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(
                {
                    "floquet_exp": float(row["floquet_exponent"]),
                    "wave_type": row["wave_type"],
                    "omega": float(row["omega"]),
                }
            )
    return data


def plot_continuum(data, output_file="floquet_continuum_corrected.png"):
    """Create omega vs floquet_exponent plot"""

    sound_x = [d["floquet_exp"] for d in data if d["wave_type"] == "sound"]
    sound_y = [d["omega"] for d in data if d["wave_type"] == "sound"]
    alfven_x = [d["floquet_exp"] for d in data if d["wave_type"] == "alfven"]
    alfven_y = [d["omega"] for d in data if d["wave_type"] == "alfven"]

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(
        sound_x,
        sound_y,
        c="blue",
        s=30,
        alpha=0.6,
        label="Sound waves",
        edgecolors="white",
        linewidth=0.5,
    )
    ax.scatter(
        alfven_x,
        alfven_y,
        c="orange",
        s=30,
        alpha=0.7,
        label="Alfvén waves",
        edgecolors="white",
        linewidth=0.5,
    )

    ax.set_xlabel("Floquet exponent ν", fontsize=14, fontweight="bold")
    ax.set_ylabel("Omega ω", fontsize=14, fontweight="bold")
    ax.set_title(
        "Floquet Continuum at ψ=0.02 (using getMGeo)", fontsize=16, fontweight="bold"
    )
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")
    return fig


if __name__ == "__main__":
    data = read_interpolation_data("interpolation_test.csv")
    print(f"Total points: {len(data)}")
    sound_count = sum(1 for d in data if d["wave_type"] == "sound")
    alfven_count = sum(1 for d in data if d["wave_type"] == "alfven")
    print(f"Sound waves: {sound_count}")
    print(f"Alfvén waves: {alfven_count}")

    plot_continuum(data)
    plt.show()
