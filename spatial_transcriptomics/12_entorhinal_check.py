"""
12 - Spatial test of the entorhinal-cortex assignments

Selects beads whose MapMyCells subclass name contains "ENT", plots them on the
tissue coordinates next to the hippocampal arc, and tests whether they form a
compact region or are scattered.

Two spatial statistics, each compared against a null built from random draws of
the same number of beads from the same section:

  1. fraction of beads in the largest connected component (beads linked when
     closer than EPS)
  2. median distance to the 5th nearest neighbour within the set

If a set of beads marks a real anatomical territory it will be far more
clustered than a random draw of the same size. If the two are indistinguishable,
the assignments are scattered calls, not a region.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(r"path/to/slide_seq_project")
ANN = ROOT / "out" / "annotation" / "mapmycells_annotation_all_beads.csv"
OUT = ROOT / "out" / "entorhinal_check"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = ["J11", "J12", "J21", "J22", "J23"]
GROUP = {"J11": "SED", "J12": "SED", "J21": "RUN", "J22": "RUN", "J23": "RUN"}
HIPPO = ["DG", "CA1", "CA2", "CA3"]
EPS = 60.0        # linking distance, ~3x mean bead spacing
K = 5             # k-th nearest neighbour
N_NULL = 200      # random draws for the null
rng = np.random.default_rng(1234)

ann = pd.read_csv(ANN)
ann["is_ENT"] = ann.subclass_name.str.contains("ENT", regex=False)

print("subclasses matching ENT:")
print(ann[ann.is_ENT].subclass_name.value_counts().to_string())
print(f"\ntotal ENT beads: {ann.is_ENT.sum()} of {len(ann)} "
      f"({100*ann.is_ENT.mean():.2f}%)\n")


def largest_component_frac(xy, eps=EPS):
    if len(xy) < 2:
        return np.nan
    tree = cKDTree(xy)
    pairs = tree.query_pairs(eps, output_type="ndarray")
    if len(pairs) == 0:
        return 1.0 / len(xy)
    n = len(xy)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return np.bincount(lab).max() / n


def median_knn_dist(xy, k=K):
    if len(xy) <= k:
        return np.nan
    d, _ = cKDTree(xy).query(xy, k=k + 1)
    return float(np.median(d[:, k]))


rows = []
fig, axes = plt.subplots(1, 5, figsize=(25, 5.5))

for ax, s in zip(axes, SAMPLES):
    a = ann[ann["sample"] == s]
    pos = pd.read_csv(ROOT / s.lower() / f"{s}_MatchedBeadLocation.csv",
                      index_col=0)
    pos.columns = ["x", "y"]
    a = a.merge(pos, left_on="barcode", right_index=True, how="left")

    ent = a[a.is_ENT]
    hip = a[a.hippocampal_region.isin(HIPPO)]
    xy_ent = ent[["x", "y"]].to_numpy()
    xy_all = a[["x", "y"]].to_numpy()

    obs_frac = largest_component_frac(xy_ent)
    obs_knn = median_knn_dist(xy_ent)

    null_frac, null_knn = [], []
    for _ in range(N_NULL):
        idx = rng.choice(len(xy_all), size=len(xy_ent), replace=False)
        null_frac.append(largest_component_frac(xy_all[idx]))
        null_knn.append(median_knn_dist(xy_all[idx]))
    null_frac = np.array(null_frac); null_knn = np.array(null_knn)

    # Distance from each ENT bead to the nearest hippocampal bead, against the
    # same distance for random beads.
    hip_tree = cKDTree(hip[["x", "y"]].to_numpy())
    d_ent = np.median(hip_tree.query(xy_ent, k=1)[0])
    d_null = np.median([np.median(hip_tree.query(
        xy_all[rng.choice(len(xy_all), size=len(xy_ent), replace=False)], k=1)[0])
        for _ in range(N_NULL)])

    rows.append(dict(
        sample=s, group=GROUP[s],
        n_beads_total=len(a), n_ENT=len(ent),
        pct_ENT=round(100 * len(ent) / len(a), 3),
        corr_ENT=round(ent.subclass_correlation_coefficient.median(), 4),
        corr_global=round(a.subclass_correlation_coefficient.median(), 4),
        largest_comp_ENT=round(obs_frac, 3),
        largest_comp_null=round(null_frac.mean(), 3),
        p_compact=round(float(np.mean(null_frac >= obs_frac)), 3),
        knn5_ENT=round(obs_knn, 1),
        knn5_null=round(null_knn.mean(), 1),
        p_knn=round(float(np.mean(null_knn <= obs_knn)), 3),
        dist_to_hippo_ENT=round(float(d_ent), 1),
        dist_to_hippo_null=round(float(d_null), 1),
    ))

    ax.scatter(a.x, a.y, s=0.6, c="#e6e6e6", linewidths=0, rasterized=True)
    colors = {"DG": "#1b7837", "CA1": "#2166ac", "CA2": "#7fcdbb", "CA3": "#762a83"}
    for reg, col in colors.items():
        r = a[a.hippocampal_region == reg]
        ax.scatter(r.x, r.y, s=2.5, c=col, linewidths=0, label=reg, rasterized=True)
    ax.scatter(ent.x, ent.y, s=18, c="#d73027", edgecolors="black",
               linewidths=0.3, label=f"ENT (n={len(ent)})", zorder=5)
    ax.set_title(f"{s} ({GROUP[s]})\nENT n={len(ent)}  "
                 f"largest comp {obs_frac:.2f} vs null {null_frac.mean():.2f}",
                 fontsize=10)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7, markerscale=2, loc="upper right", framealpha=0.9)

plt.tight_layout()
plt.savefig(OUT / "ENT_spatial_distribution.png", dpi=200)

res = pd.DataFrame(rows)
res.to_csv(OUT / "ENT_spatial_stats.csv", index=False)
pd.set_option("display.width", 250)
print(res.to_string(index=False))
