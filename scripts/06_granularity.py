"""Separate graph construction from partition granularity, then choose both.

    python scripts/06_granularity.py

Step 05 swept the similarity threshold at a fixed Leiden resolution of 1.0 and
scored each partition against NPClassifier with AMI. That is confounded: the
threshold sets the graph, the resolution sets how finely it is cut, and AMI against
a coarse label set falls as the number of communities rises. The result was a
monotone trend with its optimum at the edge of the swept range, which is a property
of the metric rather than of the chemistry.

This runs a two-dimensional sweep over cached edge lists, so no similarity is
recomputed, and reports metrics with opposing biases side by side:

  AMI / V-measure   fall when the partition is much finer than the labels
  purity            rises when the partition is finer
  n_communities     compared against the number of label categories

A defensible setting is one where a partition of comparable granularity to the
labels recovers them, rather than one that maximises a single count-sensitive score.

Writes data/interim/granularity.json and granularity.csv.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seagap import config  # noqa: E402
from seagap.tables import load_table  # noqa: E402

try:
    import igraph as ig
    import leidenalg
except ImportError:
    sys.exit("igraph and leidenalg are required")

try:
    from sklearn.metrics import (adjusted_mutual_info_score,
                                 adjusted_rand_score,
                                 homogeneity_completeness_v_measure)
except ImportError:
    sys.exit("scikit-learn is required for this step")

pd.set_option("display.width", 220)
print("SCRIPT VERSION: 2026-10-14 granularity\n")

RESOLUTIONS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
MIN_FAMILY = 5

sub = load_table(config.INTERIM / "chemspace_members")
n = len(sub)
print(f"{n:,} compounds (scope {config.TIER})")

LABEL_COLS = ["np_classifier_pathway", "np_classifier_superclass",
              "np_classifier_class"]
labels = {}
for c in LABEL_COLS:
    if c in sub.columns and sub[c].notna().any():
        y = sub[c].fillna("(missing)").astype(str).values
        labels[c] = y
        k = len(set(y[y != "(missing)"]))
        print(f"  {c:28s} {k:>5,} categories, "
              f"{(y != '(missing)').mean() * 100:5.1f}% labelled")

scaf = sub["murcko_rdkit"] if "murcko_rdkit" in sub.columns else None

# ------------------------------------------------------------------ which graphs
avail = []
for thr in config.THRESHOLD_SWEEP:
    for suf in (".parquet", ".pkl"):
        p = config.INTERIM / f"edges_t{thr}{suf}"
        if p.exists():
            avail.append(thr)
            break
if not avail:
    sys.exit("No cached edge lists found. Run 05_chemspace.py first.")
print(f"\ncached graphs: {avail}")


def purity(memb, y, min_size=MIN_FAMILY):
    """Mean dominant-label share across communities of at least min_size,
    counting only labelled members."""
    d = pd.DataFrame({"c": memb, "y": y})
    d = d[d["y"] != "(missing)"]
    g = d.groupby("c")["y"]
    sizes = g.size()
    keep = sizes[sizes >= min_size].index
    if len(keep) == 0:
        return float("nan"), 0
    p = g.apply(lambda s: s.value_counts(normalize=True).iloc[0])
    return float(p.loc[keep].mean()), len(keep)


rows = []
for thr in avail:
    edges = load_table(config.INTERIM / f"edges_t{thr}")
    g = ig.Graph(n=n, edges=list(zip(edges["src"].tolist(), edges["dst"].tolist())),
                 directed=False)
    g.es["weight"] = edges["sim"].tolist()
    print(f"\nthreshold {thr}: {g.ecount():,} edges")

    for res in RESOLUTIONS:
        t0 = time.time()
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition, weights="weight",
            resolution_parameter=res, seed=config.LEIDEN_SEED)
        memb = np.asarray(part.membership)
        sizes = np.bincount(memb)

        row = {
            "threshold": thr,
            "resolution": res,
            "communities": int(len(sizes)),
            "largest_frac": round(float(sizes.max() / n), 4),
            "singletons": int((sizes == 1).sum()),
            f"fams_ge{MIN_FAMILY}": int((sizes >= MIN_FAMILY).sum()),
            "pct_in_fams": round(float(sizes[sizes >= MIN_FAMILY].sum() / n * 100), 1),
            "modularity": round(float(part.modularity), 4),
        }

        for c, y in labels.items():
            short = c.replace("np_classifier_", "")
            m = y != "(missing)"
            if m.sum() < 100:
                continue
            row[f"ami_{short}"] = round(
                adjusted_mutual_info_score(y[m], memb[m]), 4)
            h, comp, v = homogeneity_completeness_v_measure(y[m], memb[m])
            row[f"homog_{short}"] = round(h, 4)
            row[f"compl_{short}"] = round(comp, 4)
            row[f"v_{short}"] = round(v, 4)
            pu, nf = purity(memb, y)
            row[f"purity_{short}"] = round(pu, 4) if pu == pu else None

        if scaf is not None:
            d = pd.DataFrame({"c": memb, "s": scaf.values})
            gg = d.groupby("c")["s"]
            sz = gg.size()
            big = sz[sz >= MIN_FAMILY].index
            if len(big):
                nun = gg.nunique()
                row["scaf_per_cmpd"] = round(
                    float((nun.loc[big] / sz.loc[big]).mean()), 3)

        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"    res {res:<5} communities {row['communities']:>6,}  "
              f"largest {row['largest_frac']:.3f}  "
              f"homog {row.get('homog_superclass', float('nan'))}  "
              f"compl {row.get('compl_superclass', float('nan'))}  "
              f"V {row.get('v_superclass', float('nan'))}  "
              f"inFams {row['pct_in_fams']:.0f}%",
              flush=True)

sweep = pd.DataFrame(rows)

bar = "=" * 110
print(f"\n{bar}\nFull two-dimensional sweep\n{bar}")
cols = ["threshold", "resolution", "communities", "largest_frac",
        f"fams_ge{MIN_FAMILY}", "pct_in_fams", "modularity",
        "homog_superclass", "compl_superclass", "v_superclass",
        "purity_superclass", "ami_superclass", "v_class", "scaf_per_cmpd"]
cols = [c for c in cols if c in sweep.columns]
print(sweep[cols].to_string(index=False))

# ------------------------------------------------------------------ read it out
print(f"\n{bar}\nHow to read this\n{bar}")
n_sc = len(set(labels.get("np_classifier_superclass", ["x"]))) - 1
n_cl = len(set(labels.get("np_classifier_class", ["x"]))) - 1
print(f"NPClassifier has {n_sc} superclasses and {n_cl} classes.\n")
print("AMI and ARI fall as the community count rises even when no community mixes")
print("two labels, so neither can choose a granularity. Homogeneity can: it is")
print("exactly 1.0 for any partition that never mixes labels, whatever its size.")
print("Completeness is the opposite, falling as a label gets split across more")
print("communities. V-measure is their harmonic mean and has a real interior")
print("optimum, so it is the number to choose on.\n")
print("  homogeneity high, completeness low -> pure families, each superclass "
      "split\n                                        across many "
      "(over-partitioned)")
print("  homogeneity low, completeness high -> families merge distinct "
      "superclasses\n                                        (under-partitioned)")

print("\nBest V-measure against superclass, per threshold:")
if "v_superclass" in sweep.columns:
    best = sweep.loc[sweep.groupby("threshold")["v_superclass"].idxmax()]
    print(best[cols].to_string(index=False))
    b = sweep.loc[sweep["v_superclass"].idxmax()]
    print(f"\nHighest V overall: threshold {b['threshold']}, resolution "
          f"{b['resolution']}  ->  V={b['v_superclass']}, "
          f"{int(b['communities']):,} communities, "
          f"homogeneity {b['homog_superclass']}, completeness "
          f"{b['compl_superclass']}, {b['pct_in_fams']:.0f}% of compounds in "
          f"families of {MIN_FAMILY}+")

print("\nSettings whose community count is within 5x of the superclass count:")
if n_sc > 0:
    near = sweep[(sweep["communities"] >= n_sc / 5)
                 & (sweep["communities"] <= n_sc * 5)]
    print(near[cols].to_string(index=False) if len(near)
          else "  none in the swept range (try lower resolutions)")

print("\nUsable granularity for the taxon-by-family network:")
usable = sweep[(sweep["pct_in_fams"] >= 60) & (sweep["largest_frac"] <= 0.15)]
print(usable[cols].to_string(index=False) if len(usable)
      else "  no setting keeps 60%+ of compounds in families of 5+ without a "
           "giant component")
print("\nA bipartite network needs most compounds inside non-trivial families, "
      "so a setting\nthat leaves half the compounds as singletons is not usable "
      "whatever it scores.")

sweep.to_csv(config.INTERIM / "granularity.csv", index=False)
(config.INTERIM / "granularity.json").write_text(json.dumps({
    "tier": config.TIER,
    "n_compounds": n,
    "resolutions": RESOLUTIONS,
    "thresholds": avail,
    "min_family": MIN_FAMILY,
    "n_superclasses": n_sc,
    "n_classes": n_cl,
    "sweep": rows,
}, indent=2, default=float))
print(f"\nWrote {config.INTERIM / 'granularity.csv'}")
print(f"Wrote {config.INTERIM / 'granularity.json'}")
