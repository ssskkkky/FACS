#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Read data
data = []
with open('continuum-gfile_131041', 'r') as f:
    for line in f:
        values = list(map(float, line.split()))
        n = int(values[0])
        m = int(values[1])
        coords = np.array(values[2:]).reshape(-1, 2)
        data.append((n, m, coords))

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each mode line
colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
for i, (n, m, coords) in enumerate(data):
    ax.plot(coords[:, 0], coords[:, 1], 'o-', 
            markersize=2, linewidth=0.5,
            color=colors[i], 
            label=f'n={n}, m={m}')

ax.set_xlabel('Minor radius (normalized)')
ax.set_ylabel('Frequency ω (normalized)')
ax.set_title('Alfvén Continuum (Single DOF, Hill Equation)')
ax.legend(loc='best', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.5)

plt.tight_layout()
plt.savefig('continuum_main.png', dpi=150)
print("Plot saved to continuum_main.png")
plt.close()
