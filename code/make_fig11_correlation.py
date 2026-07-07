"""Figure 11: N=3 vs N=5 sensitivity gain across six axis-network combinations.
Scatter with fitted line, R², p, and per-point labels. 
Point styling matches Figure 9's convention: tree axis = orange, markov axis = green;
network shape = Kanno square, Domain17 circle, ALARM triangle."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

FIGS_DIR = "/home/claude/figs"
RES_DIR  = "/home/claude/substrate_abm_project/results"
REL_DIR  = "/home/claude/substrate_abm_release/results"

TREE_COLOR   = "#B45309"
MARKOV_COLOR = "#166534"

def find_path(candidates):
    for p in candidates:
        if os.path.exists(p): return p
    raise FileNotFoundError(candidates)

def s(path):
    d = json.load(open(path))
    r = d['results']
    ma_c  = np.array([x['matched-aware__init_conflicts_mean']  for x in r])
    mia_c = np.array([x['mismatched-aware__init_conflicts_mean'] for x in r])
    return (mia_c - ma_c).mean()

cells = [('domain17','tree'), ('kanno','tree'), ('alarm','tree'),
         ('domain17','markov'), ('kanno','markov'), ('alarm','markov')]

n3_paths = {
    ('domain17','tree'):   f"{REL_DIR}/conflict_instrument_domain17_tree.json",
    ('domain17','markov'): f"{REL_DIR}/conflict_instrument_domain17_markov.json",
    ('kanno','tree'):      f"{REL_DIR}/conflict_instrument_kanno_tree.json",
    ('kanno','markov'):    f"{REL_DIR}/conflict_instrument_kanno_markov.json",
    ('alarm','tree'):      f"{RES_DIR}/conflict_instrument_alarm_tree.json",
    ('alarm','markov'):    f"{RES_DIR}/conflict_instrument_alarm_markov.json",
}
n5_paths = {c: f"{RES_DIR}/n5_instrument_{c[0]}_{c[1]}.json" for c in cells}

x = np.array([s(n3_paths[c]) for c in cells])
y = np.array([s(n5_paths[c]) for c in cells])

slope, intercept, r_val, p_val, se = stats.linregress(x, y)
r2 = r_val**2

fig, ax = plt.subplots(figsize=(10, 7.5))

# Draw fit line first
x_fit = np.linspace(0, max(x)*1.35, 100)
y_fit = slope * x_fit + intercept
ax.plot(x_fit, y_fit, color='#374151', linewidth=1.6, linestyle='--',
        label=f'Linear fit: y = {slope:.2f}x + {intercept:+.1f}', zorder=1)

# 1:1 reference for context
ax.plot([0, max(x)*1.35], [0, max(x)*1.35], color='gray', linewidth=0.7,
        linestyle=':', alpha=0.6, label='y = x (no scaling)', zorder=0)

# Point styling
NET_MARKER = {'kanno': 's', 'domain17': 'o', 'alarm': '^'}
NET_NAME   = {'kanno': 'Kanno', 'domain17': 'Domain17', 'alarm': 'ALARM'}
AXIS_COLOR = {'tree': TREE_COLOR, 'markov': MARKOV_COLOR}
AXIS_NAME  = {'tree': 'tree', 'markov': 'markov'}

# Scatter with individual labeling
for i, (net, axis_) in enumerate(cells):
    xi, yi = x[i], y[i]
    m = NET_MARKER[net]
    c = AXIS_COLOR[axis_]
    ax.scatter(xi, yi, marker=m, s=230, color=c, edgecolors='black',
               linewidth=1.4, zorder=4)
    # Label placement: offset to avoid overlap
    label = f"{NET_NAME[net]}-{AXIS_NAME[axis_]}"
    # Custom offsets per point to avoid crowding
    offsets = {
        ('domain17','tree'):   (-2, 5),
        ('kanno','tree'):      (3, -6),
        ('alarm','tree'):      (3, -3),
        ('domain17','markov'): (3, 2),
        ('kanno','markov'):    (3, -3),
        ('alarm','markov'):    (3, 3),
    }
    dx, dy = offsets.get((net, axis_), (3, 3))
    ax.annotate(label, xy=(xi, yi), xytext=(xi+dx, yi+dy), fontsize=10.5,
                weight='bold', color=c)

# R² and p annotation as text box
stats_text = (f"Pearson r = +{r_val:.3f}\n"
              f"R² = {r2:.3f}\n"
              f"p = {p_val:.3f}\n"
              f"n = 6 (axis × network) cells\n"
              f"slope = {slope:.2f}   intercept = {intercept:+.1f}")
ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.55', facecolor='#FEF3C7',
                  edgecolor='#B45309', linewidth=1.2, alpha=0.95))

# Explanatory note
note = ("Points above the y=x reference line indicate super-linear\n"
        "scaling of the sensitivity gain from N=3 to N=5.")
ax.text(0.97, 0.03, note, transform=ax.transAxes, fontsize=10,
        horizontalalignment='right', verticalalignment='bottom',
        color='#374151', style='italic')

ax.set_xlabel("Total sensitivity gain at N=3\n(initial conflicts, mm-aware − matched-aware)", fontsize=11)
ax.set_ylabel("Total sensitivity gain at N=5\n(initial conflicts, mm-aware − matched-aware)", fontsize=11)
ax.set_title("Cross-network scaling of the sensitivity gain: N=3 vs N=5",
             fontsize=13, weight='bold')
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.set_axisbelow(True)

# Custom legend
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#9CA3AF',
           markersize=11, markeredgecolor='black', label='Domain17'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#9CA3AF',
           markersize=11, markeredgecolor='black', label='Kanno'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#9CA3AF',
           markersize=11, markeredgecolor='black', label='ALARM'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=TREE_COLOR,
           markersize=11, markeredgecolor='black', label='Tree axis'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=MARKOV_COLOR,
           markersize=11, markeredgecolor='black', label='Markov axis'),
]
ax.legend(handles=legend_elems, loc='upper right', fontsize=10, framealpha=0.95,
          ncol=1, bbox_to_anchor=(1.0, 0.75))
# The fit-line legend
line_legend = ax.legend(handles=[Line2D([0], [0], color='#374151', linewidth=1.6,
                                         linestyle='--',
                                         label=f'Linear fit'),
                                  Line2D([0], [0], color='gray', linewidth=0.7,
                                         linestyle=':', label='y = x (no scaling)')],
                        loc='lower right', fontsize=9, framealpha=0.9,
                        bbox_to_anchor=(1.0, 0.19))
ax.add_artist(line_legend)
# Restore marker legend
ax.legend(handles=legend_elems, loc='upper right', fontsize=10, framealpha=0.95,
          bbox_to_anchor=(1.0, 0.72))

ax.set_xlim(-2, max(x) * 1.35)
ax.set_ylim(-2, max(y) * 1.15)

plt.tight_layout()
out = f"{FIGS_DIR}/fig11_scaling_correlation.png"
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved {out}")
print(f"r = {r_val:.3f}, R² = {r2:.3f}, p = {p_val:.4f}")
print(f"slope = {slope:.3f}, intercept = {intercept:.2f}")
plt.close()
