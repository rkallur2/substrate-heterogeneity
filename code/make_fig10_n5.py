"""Figure 10 v2: three-network N=3 vs N=5 team-size scalability."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGS_DIR = "/home/claude/figs"
RES_DIR  = "/home/claude/substrate_abm_project/results"
REL_DIR  = "/home/claude/substrate_abm_release/results"

TREE_COLOR = "#B45309"
MARKOV_COLOR = "#166534"

def find_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(candidates)

def stats(path):
    d = json.load(open(path))
    r = d['results']
    ma  = np.array([x['matched-aware__final_mean']  for x in r])
    mia = np.array([x['mismatched-aware__final_mean'] for x in r])
    ma_c  = np.array([x['matched-aware__init_conflicts_mean']  for x in r])
    mia_c = np.array([x['mismatched-aware__init_conflicts_mean'] for x in r])
    return dict(sub_gap=ma-mia, sens=mia_c-ma_c, n=len(r))

cells = [('domain17','tree'), ('kanno','tree'), ('alarm','tree'),
         ('domain17','markov'), ('kanno','markov'), ('alarm','markov')]

n3_paths = {
    ('domain17','tree'):   [f"{REL_DIR}/conflict_instrument_domain17_tree.json"],
    ('domain17','markov'): [f"{REL_DIR}/conflict_instrument_domain17_markov.json"],
    ('kanno','tree'):      [f"{REL_DIR}/conflict_instrument_kanno_tree.json"],
    ('kanno','markov'):    [f"{REL_DIR}/conflict_instrument_kanno_markov.json"],
    ('alarm','tree'):      [f"{RES_DIR}/conflict_instrument_alarm_tree.json"],
    ('alarm','markov'):    [f"{RES_DIR}/conflict_instrument_alarm_markov.json"],
}
n5_paths = {k: [f"{RES_DIR}/n5_instrument_{k[0]}_{k[1]}.json"] for k in cells}

n3 = {k: stats(find_path(v)) for k, v in n3_paths.items()}
n5 = {k: stats(find_path(v)) for k, v in n5_paths.items()}

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

BAR_W = 0.8
GAP_WITHIN = 0.15
GAP_ACROSS = 0.6
GAP_NET = 1.5

def positions_for_network(x0):
    x_n3_t = x0
    x_n5_t = x_n3_t + BAR_W + GAP_WITHIN
    x_n3_m = x_n5_t + BAR_W + GAP_ACROSS
    x_n5_m = x_n3_m + BAR_W + GAP_WITHIN
    return [x_n3_t, x_n5_t, x_n3_m, x_n5_m]

network_x0 = {'domain17': 0}
step = 4*BAR_W + 2*GAP_WITHIN + GAP_ACROSS + GAP_NET
network_x0['kanno'] = step
network_x0['alarm'] = 2*step

def render_panel(ax, key, ylabel, title, showratio=False):
    for net in ['domain17', 'kanno', 'alarm']:
        xs = positions_for_network(network_x0[net])
        data_bars = [
            (n3[(net,'tree')][key],   TREE_COLOR,   ''),
            (n5[(net,'tree')][key],   TREE_COLOR,   '///'),
            (n3[(net,'markov')][key], MARKOV_COLOR, ''),
            (n5[(net,'markov')][key], MARKOV_COLOR, '///'),
        ]
        for xi, (vals, color, hatch) in zip(xs, data_bars):
            m = vals.mean()
            se = vals.std(ddof=1)/np.sqrt(len(vals)) if len(vals) > 1 else 0
            ax.bar(xi, m, width=BAR_W, color=color, alpha=0.75,
                   edgecolor='black', linewidth=0.7, hatch=hatch)
            if se > 0:
                ax.errorbar(xi, m, yerr=se, color='black', capsize=3, zorder=4)
            rng = np.random.default_rng(abs(hash((net, xi, key))) % 2**32)
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(np.full_like(vals, xi) + jitter, vals, color='black',
                       alpha=0.35, s=13, zorder=3)
            if key == 'sub_gap':
                yoff = 0.008 if m >= 0 else -0.014
                ax.text(xi, m + yoff, f"{m:+.3f}", ha='center',
                        fontsize=7.5, weight='bold', color=color)
            else:
                ax.text(xi, m + max(se, 3) + 3, f"{m:+.1f}", ha='center',
                        fontsize=7.5, weight='bold', color=color)
        if showratio:
            for ps in [0, 2]:
                x0v = xs[ps] + BAR_W/2 - 0.05
                x1v = xs[ps+1] - BAR_W/2 + 0.05
                m0, m1 = data_bars[ps][0].mean(), data_bars[ps+1][0].mean()
                col = data_bars[ps][1]
                if m0 > 0.5:
                    ratio = m1/m0
                    ax.annotate('', xy=(x1v, m1), xytext=(x0v, m0),
                                arrowprops=dict(arrowstyle='->', color=col, lw=1.2, alpha=0.55))
                    ax.text((x0v+x1v)/2, (m0+m1)/2 + 4, f"{ratio:.1f}×",
                            fontsize=9, weight='bold', color=col,
                            ha='center', va='bottom', alpha=0.85)

    net_labels = ['Domain17', 'Kanno', 'ALARM']
    net_centers = []
    for net in ['domain17', 'kanno', 'alarm']:
        xs = positions_for_network(network_x0[net])
        net_centers.append((xs[0] + xs[-1]) / 2)
    ax.set_xticks(net_centers)
    ax.set_xticklabels(net_labels, fontsize=11, weight='bold')
    ax.axhline(0, color='gray', linewidth=0.7, linestyle='--')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, weight='bold')
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)

render_panel(axes[0], 'sub_gap',
             'Substrate gap (matched-aware − mismatched-aware)',
             'Substrate gap: N=3 vs N=5, three networks')
render_panel(axes[1], 'sens',
             'Total sensitivity gain (initial conflicts, mm-aware − matched-aware)',
             'Sensitivity gain: N=3 vs N=5, three networks',
             showratio=True)

legend_patches = [
    mpatches.Patch(facecolor='white', edgecolor='black', label='N=3 (Ext. 3 primary)'),
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='N=5 (scalability)'),
    mpatches.Patch(color=TREE_COLOR, alpha=0.75, label='Tree axis'),
    mpatches.Patch(color=MARKOV_COLOR, alpha=0.75, label='Markov axis'),
]
fig.legend(handles=legend_patches, loc='upper center', ncol=4,
           bbox_to_anchor=(0.5, 1.02), fontsize=10, framealpha=0.95)

plt.suptitle("Team-size scalability across three networks",
             fontsize=13, weight='bold', y=1.06)
plt.tight_layout()
out = f"{FIGS_DIR}/fig10_n5_scalability.png"
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved {out}")
plt.close()
