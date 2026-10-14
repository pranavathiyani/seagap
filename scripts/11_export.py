"""Build the single-file viewer from the interim tables.

    python scripts/11_export.py

Reads viewer/template.html, substitutes a JSON payload, writes docs/index.html.
The result is one self-contained file: no build step, no CDN, no server. Serve it
from GitHub Pages by pointing Pages at /docs on the default branch.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seagap import config  # noqa: E402
from seagap.curve import fit_accumulation  # noqa: E402
from seagap.tables import load_table  # noqa: E402

ROOT = config.ROOT
TEMPLATE = ROOT / "viewer" / "template.html"
OUT = ROOT / "docs" / "index.html"
MIN_N = 10
MIN_ORDER_N = 30
MIN_ORDER_GENERA = 3

print("SCRIPT VERSION: 2026-10-14 export\n")
if not TEMPLATE.exists():
    sys.exit(f"missing {TEMPLATE}")


def jnum(v):
    """None for anything pandas calls missing, plain python numbers otherwise."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else round(float(v), 4)
    return v


# ------------------------------------------------------------------ rebuild
work = load_table(config.INTERIM / "families")
famp = load_table(config.INTERIM / "family_profile").set_index("family")
tax = load_table(config.INTERIM / "organism_taxonomy")
org = pd.read_csv(config.RELEASE / config.TABLES["organisms"], low_memory=False)
mol_org = pd.read_csv(config.RELEASE / config.TABLES["molecule_organism"],
                      low_memory=False)
scope = json.loads((config.INTERIM / "scope.json").read_text())
bias = json.loads((config.INTERIM / "bias.json").read_text())

link = (mol_org[["molecule_id", "organism_id"]].dropna().drop_duplicates()
        .merge(org[["id", "aphiaid_worms"]].rename(columns={"id": "organism_id"}),
               on="organism_id", how="left")
        .merge(tax[["aphiaid_worms", "genus", "order", "phylum"]],
               on="aphiaid_worms", how="left")
        .merge(work[["molecule_id", "family"]], on="molecule_id", how="inner")
        .dropna(subset=["genus"]))
spec = set(famp[~famp["is_primary"]].index)
sl = link[link["family"].isin(spec)].drop_duplicates(
    subset=["molecule_id", "genus", "family"])

obs = (sl.groupby("genus")
       .agg(n_compounds=("molecule_id", "nunique"),
            n_families=("family", "nunique")))
obs = obs[obs["n_compounds"] >= MIN_N].copy()
gf = fit_accumulation(obs["n_compounds"], obs["n_families"])
obs["z"] = np.round(gf["z"], 2)
obs["exp_families"] = np.round(gf["pred"], 1)
obs = obs.join(sl.groupby("genus").agg(
    phylum=("phylum", lambda s: s.mode().iloc[0] if s.notna().any() else None),
    order=("order", lambda s: s.mode().iloc[0] if s.notna().any() else None)))
if gf["n_floored"]:
    print(f"WARNING: {gf['n_floored']} genera hit the sd floor")
print(f"{len(obs):,} genera, curve n^{gf['b']:.3f}, R2 {gf['r2']:.3f}")

ordall = (sl.dropna(subset=["order"]).groupby("order")
          .agg(n_compounds=("molecule_id", "nunique"),
               n_families=("family", "nunique"),
               n_genera=("genus", "nunique"),
               phylum=("phylum", lambda s: s.mode().iloc[0] if s.notna().any() else None)))
ordobs = ordall[(ordall["n_compounds"] >= MIN_ORDER_N)
                & (ordall["n_genera"] >= MIN_ORDER_GENERA)].copy()
of = fit_accumulation(ordobs["n_compounds"], ordobs["n_families"])
ordobs["z"] = np.round(of["z"], 2)
print(f"{len(ordobs):,} orders, curve n^{of['b']:.3f}, R2 {of['r2']:.3f}")

# frontier: barely examined genera inside orders that are broad for their effort
prod = set(ordobs[ordobs["z"] > 0].index)
allg = link.groupby("genus")["molecule_id"].nunique()
gmeta = (link.dropna(subset=["order"]).groupby("genus")
         .agg(order=("order", lambda s: s.mode().iloc[0]),
              n_compounds=("molecule_id", "nunique")))
front_gen = gmeta[(gmeta["order"].isin(prod)) & (gmeta["n_compounds"] < MIN_N)]
fcount = front_gen.groupby("order").size().rename("unexamined")
frontier = (ordobs.loc[ordobs.index.isin(prod)]
            .join(fcount).fillna({"unexamined": 0}))
frontier["unexamined"] = frontier["unexamined"].astype(int)
frontier = frontier[frontier["unexamined"] > 0]


# ------------------------------------------------------------------ network
# Bipartite genus-family edges, as integer indices so the whole thing stays small
# enough to inline. Filtering happens in the browser, so any order is selectable
# without a round trip.
pairs = (sl.groupby(["genus", "family"]).size().reset_index()[["genus", "family"]])
gkeep = set(obs.index)
pairs = pairs[pairs["genus"].isin(gkeep)]

gen_order = sorted(gkeep)
gi = {g: i for i, g in enumerate(gen_order)}
fam_order = sorted(pairs["family"].unique())
fi = {f: i for i, f in enumerate(fam_order)}

order_names = sorted({v for v in obs["order"].dropna().unique()})
oi = {o: i for i, o in enumerate(order_names)}

fam_size = sl.groupby("family")["molecule_id"].nunique()
fam_span = sl.groupby("family")["genus"].nunique()

net = {
    "orders": order_names,
    "defaultOrder": (frontier.sort_values("unexamined", ascending=False).index[0]
                     if len(frontier) else (order_names[0] if order_names else "")),
    "genera": [[g,
                int(obs.at[g, "n_compounds"]),
                float(obs.at[g, "z"]),
                oi.get(obs.at[g, "order"], -1) if isinstance(obs.at[g, "order"], str) else -1]
               for g in gen_order],
    "families": [[int(fam_size.get(f, 0)), int(fam_span.get(f, 0))] for f in fam_order],
    "edges": [v for _, r in pairs.iterrows()
              for v in (gi[r["genus"]], fi[r["family"]])],
}
print(f"network: {len(net['genera']):,} genera, {len(net['families']):,} families, "
      f"{len(net['edges']) // 2:,} edges, {len(order_names):,} orders")

# ------------------------------------------------------------------ provenance
def binomial(s):
    import re
    if not isinstance(s, str) or not s.strip():
        return ""
    t = re.sub(r"\s*\([^)]*\)", " ", s)
    t = re.sub(r"\b(?:sp|spp|var|subsp|ssp|cf|aff|nov|ined)\.?\b", " ", t, flags=re.I)
    t = re.sub(r"[^A-Za-z\s\-]", " ", t)
    out = []
    for i, w in enumerate([w for w in t.split() if len(w) > 1]):
        if i == 0:
            out.append(w.lower())
        elif w[0].islower():
            out.append(w.lower()); break
        else:
            break
    return " ".join(out[:2])


o = (org[["id", "name", "aphiaid_worms"]].rename(columns={"id": "organism_id"})
     .merge(tax[["aphiaid_worms", "scientificname", "phylum", "kingdom"]],
            on="aphiaid_worms", how="left"))
o["rg"] = o["name"].map(binomial).str.split().str[0]
o["wg"] = o["scientificname"].map(binomial).str.split().str[0]
bad = o[o["scientificname"].notna() & (o["rg"] != "") & (o["wg"] != "")
        & (o["rg"] != o["wg"])]
bad_counts = (link[link["organism_id"].isin(set(bad["organism_id"]))]
              .groupby("organism_id")["molecule_id"].nunique().rename("n"))
prov = (bad.set_index("organism_id").join(bad_counts, how="inner")
        .sort_values("n", ascending=False).head(25))
print(f"{len(bad):,} organism records resolve to a different genus")

# ------------------------------------------------------------------ payload
zz = gf["z"]
edges = np.histogram_bin_edges(zz, bins=34)
counts, _ = np.histogram(zz, bins=edges)
hist = [{"x0": round(float(edges[i]), 3), "x1": round(float(edges[i + 1]), 3),
         "n": int(counts[i])} for i in range(len(counts))]

n_sat = int((obs["z"] < -1.96).sum())
fill = bias["bipartite"]["fill_pct"]
# the key is named for whatever MIN_FAMILY the bias step used
n_fam_used = next(v for k, v in bias.items() if k.startswith("n_families_ge"))
payload = {
    "tiles": [
        [f"{len(work):,}", "compounds in structural families"],
        [f"{n_fam_used:,}", "structural families"],
        [f"{len(obs):,}", "genera with 10+ compounds"],
        [f"{fill:.2f}%", "of the genus-by-family matrix filled"],
        [f"n<sup>{gf['b']:.2f}</sup>", "families scale with compounds"],
        [f"{n_sat}", "saturated genera"],
    ],
    "curve": {"a": round(gf["a"], 4), "b": round(gf["b"], 4),
              "r2": round(gf["r2"], 4)},
    "genera": [[i, int(r.n_compounds), int(r.n_families), float(r.z),
                r.phylum if isinstance(r.phylum, str) else None,
                r.order if isinstance(r.order, str) else None]
               for i, r in obs.iterrows()],
    "hist": hist,
    "net": net,
    "satSpec": [{"label": "genus", "sci": True}, {"label": "phylum"},
                {"label": "order"}, {"label": "compounds"},
                {"label": "families"}, {"label": "expected", "dp": 1},
                {"label": "z", "sort": True, "asc": True, "dp": 2}],
    "saturated": [[i, jnum(r.phylum), jnum(r.order), int(r.n_compounds),
                   int(r.n_families), float(r.exp_families), float(r.z)]
                  for i, r in obs[obs["z"] < -1.96]
                  .sort_values("z").iterrows()],
    "frontSpec": [{"label": "order"}, {"label": "phylum"}, {"label": "compounds"},
                  {"label": "families"}, {"label": "genera studied"},
                  {"label": "genera unexamined", "sort": True},
                  {"label": "z", "dp": 2}],
    "frontier": [[i, jnum(r.phylum), int(r.n_compounds), int(r.n_families),
                  int(r.n_genera), int(r.unexamined), float(r.z)]
                 for i, r in frontier.sort_values(
                     "unexamined", ascending=False).iterrows()],
    "provSpec": [{"label": "name in release", "sci": True},
                 {"label": "WoRMS record", "sci": True},
                 {"label": "resolves to phylum"}, {"label": "kingdom"},
                 {"label": "compounds", "sort": True}],
    "provenance": [[r["name"], r.scientificname, jnum(r.phylum), jnum(r.kingdom),
                    int(r.n)] for _, r in prov.iterrows()],
    "footer": (
        f"Built from MarineNP (Zenodo 15641716) with source taxonomy resolved "
        f"against WoRMS. Scope: {scope['tier1']:,} compounds whose source is a "
        "marine eukaryote or marine plant at species rank with a strictly marine "
        "WoRMS record and a verified name, of "
        f"{scope['n_release']:,} in the release. Structural families are Leiden "
        f"communities on a Tanimoto graph at threshold "
        f"{bias['threshold']}, chosen by agreement with NPClassifier class. "
        "Deviations are standardised against the sd expected at each sample size. "
        "Code and data: <a href=\"https://github.com/pranavathiyani/seagap\">"
        "github.com/pranavathiyani/seagap</a>."),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
html = TEMPLATE.read_text(encoding="utf-8").replace(
    "__DATA__", json.dumps(payload, separators=(",", ":")))
OUT.write_text(html, encoding="utf-8")
print(f"\nWrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
print("Open it directly, or enable GitHub Pages on /docs to publish.")
