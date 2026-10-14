"""Build the taxon-by-family network, quantify sampling bias, rank the gaps.

    python scripts/07_bias.py

The descriptive core of SEAGAP. Assigns structural families at the operating
point fixed in config, crosses them against source taxonomy and collection
geography, and asks which taxa look chemically promising relative to how hard
they have been worked.

On the null model. An earlier version compared each taxon's family count against
a random draw of equal size from the whole compound pool. That cannot work: real
taxa produce chemically related metabolites, so a genus with 13 compounds covers
far fewer families than 13 unrelated compounds would, and every real taxon scores
negative by construction. In simulation that null correlated -0.91 with sampling
effort and +0.13 with true breadth, i.e. it measured effort.

The null here is other taxa at comparable effort. Family count scales with
compound count as a power law (an accumulation curve), so the script fits
log(families) against log(compounds) across taxa and takes standardised
residuals. On the same simulation this recovers true breadth at +0.86 and is
independent of effort at -0.00.

Breadth is paired with a second, independent signal: how many of a taxon's
families are found in no other taxon. A group making chemistry nobody else makes
is interesting even when it makes little of it.

Writes families.parquet, taxon_profile.parquet, location_profile.parquet,
priority_taxa.csv and bias.json into data/interim.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seagap import config  # noqa: E402
from seagap.tables import load_table, save_table  # noqa: E402

try:
    import igraph as ig
    import leidenalg
except ImportError:
    sys.exit("igraph and leidenalg are required")

pd.set_option("display.width", 215)
pd.set_option("display.max_colwidth", 40)
print("SCRIPT VERSION: 2026-10-14 bias (curve null)\n")

THR = config.CHOSEN_THRESHOLD
RES = config.CHOSEN_RESOLUTION
MINF = config.MIN_FAMILY
RANK = config.TAXON_RANK
MIN_N = 10                   # below this, a taxon's breadth is too noisy to rank

bar = "=" * 94


def section(t):
    print(f"\n{bar}\n{t}\n{bar}")


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    k = len(x)
    if k == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, k + 1) - k - 1).dot(x) / (k * x.sum()))


def lorenz(x, qs=(50, 80, 90, 95, 99)):
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    tot = x.sum()
    return {f"top{100 - q}pct": round(float(x[:max(1, int(np.ceil(len(x) * (100 - q) / 100)))].sum() / tot * 100), 1)
            for q in qs}


def accumulation_fit(n_units, n_classes, label=""):
    """Fit n_classes = a * n_units^b in log space; return fitted values, z, R2.

    Residuals are standardised by their own sd. The exponent is reported because
    it is interpretable on its own: 0.5 means doubling the compounds from a taxon
    yields about 1.4x the families.
    """
    x = np.log10(np.asarray(n_units, dtype=float))
    y = np.log10(np.asarray(n_classes, dtype=float))
    b1, b0 = np.polyfit(x, y, 1)
    pred = b0 + b1 * x
    resid = y - pred
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    sd = resid.std(ddof=1)
    print(f"{label}accumulation curve: families = {10 ** b0:.2f} x "
          f"n^{b1:.3f}   R2 = {r2:.3f}   residual sd = {sd:.4f}")
    return 10 ** pred, resid / sd, r2, b1


def clean_location(s):
    """Trim the stray whitespace and trailing punctuation in geo_locations.name.

    No synonym merging: 'Qingdao' and 'Qingdao, China' stay distinct. Resolving
    those needs a gazetteer and belongs in its own step.
    """
    if not isinstance(s, str):
        return None
    t = re.sub(r"\s+", " ", s).strip().strip(".,;:").strip()
    return t or None


# ------------------------------------------------------------------ families
sub = load_table(config.INTERIM / "chemspace_members")
n = len(sub)
edges = load_table(config.INTERIM / f"edges_t{THR}")
print(f"{n:,} compounds, {len(edges):,} edges at threshold {THR}")

g = ig.Graph(n=n, edges=list(zip(edges["src"].tolist(), edges["dst"].tolist())),
             directed=False)
g.es["weight"] = edges["sim"].tolist()
part = leidenalg.find_partition(
    g, leidenalg.RBConfigurationVertexPartition, weights="weight",
    resolution_parameter=RES, seed=config.LEIDEN_SEED)
sub["family"] = np.asarray(part.membership)
sizes = sub["family"].value_counts()
sub["family_size"] = sub["family"].map(sizes)
keep = set(sizes[sizes >= MINF].index)
work = sub[sub["family"].isin(keep)].copy()
print(f"{len(sizes):,} structural families, {len(keep):,} with >={MINF} members, "
      f"modularity {part.modularity:.4f}")
print(f"{len(work):,} compounds ({len(work) / n * 100:.1f}%) inside those families")

# ------------------------------------------------------------------ link
tax = load_table(config.INTERIM / "organism_taxonomy")
org = pd.read_csv(config.RELEASE / config.TABLES["organisms"], low_memory=False)
mol_org = pd.read_csv(config.RELEASE / config.TABLES["molecule_organism"],
                      low_memory=False)
geo = pd.read_csv(config.RELEASE / config.TABLES["geo_locations"], low_memory=False)
geo_mol = pd.read_csv(config.RELEASE / config.TABLES["geo_location_molecule"],
                      low_memory=False)

link = (mol_org[["molecule_id", "organism_id"]].dropna().drop_duplicates()
        .merge(org[["id", "aphiaid_worms"]].rename(columns={"id": "organism_id"}),
               on="organism_id", how="left")
        .merge(tax[["aphiaid_worms", "genus", "family", "order", "class", "phylum"]]
               .rename(columns={"family": "tax_family"}),
               on="aphiaid_worms", how="left")
        .merge(work[["molecule_id", "family"]], on="molecule_id", how="inner")
        .dropna(subset=[RANK])
        .drop_duplicates(subset=["molecule_id", RANK, "family"]))
print(f"\n{len(link):,} (compound, {RANK}, family) rows across "
      f"{link[RANK].nunique():,} {RANK} values")

# ------------------------------------------------------------------ 1
section(f"1. The {RANK}-by-family network")
inc = link.groupby([RANK, "family"])["molecule_id"].nunique()
n_tax, n_fam = link[RANK].nunique(), link["family"].nunique()
print(f"dimensions      {n_tax:,} x {n_fam:,}")
print(f"occupied cells  {len(inc):,} of {n_tax * n_fam:,} "
      f"({len(inc) / (n_tax * n_fam) * 100:.3f}%)")
tdeg, fdeg = inc.groupby(level=0).size(), inc.groupby(level=1).size()
print(f"\nfamilies per {RANK}:  median {tdeg.median():.0f}  mean {tdeg.mean():.1f}"
      f"  max {tdeg.max():,}")
print(f"{RANK} values per family:  median {fdeg.median():.0f}  mean "
      f"{fdeg.mean():.1f}  max {fdeg.max():,}")
excl_fams = set(fdeg[fdeg == 1].index)
print(f"\n{len(excl_fams):,} families ({len(excl_fams) / len(fdeg) * 100:.1f}%) "
      f"occur in exactly one {RANK}")

# ------------------------------------------------------------------ 2
section("2. How concentrated is the effort")
conc = {}
for lab, s in {RANK: link.groupby(RANK)["molecule_id"].nunique(),
               "family": link.groupby("family")["molecule_id"].nunique()}.items():
    conc[lab] = {"n": int(len(s)), "gini": round(gini(s.values), 3), **lorenz(s.values)}
    print(f"\n{lab}: {len(s):,} units, Gini {conc[lab]['gini']:.3f}")
    print("  " + "  ".join(f"{k}={v}%" for k, v in lorenz(s.values).items()))
    if lab == RANK:
        print("  busiest: " + ", ".join(
            f"{i} ({v:,})" for i, v in s.sort_values(ascending=False).head(5).items()))

# ------------------------------------------------------------------ 3
section("3. Chemical breadth relative to other taxa at the same effort")
obs = (link.groupby(RANK)
       .agg(n_compounds=("molecule_id", "nunique"),
            n_families=("family", "nunique")))
obs = obs[obs["n_compounds"] >= MIN_N].copy()
print(f"{len(obs):,} {RANK} values with >={MIN_N} compounds\n")
fit, z, r2, expo = accumulation_fit(obs["n_compounds"], obs["n_families"])
obs["exp_families"] = np.round(fit, 1)
obs["breadth_z"] = np.round(z, 2)
obs["breadth_ratio"] = (obs["n_families"] / obs["exp_families"]).round(3)
print(f"\nAn exponent of {expo:.2f} means doubling the compounds reported from a "
      f"taxon yields\nabout {2 ** expo:.2f}x the structural families. "
      "Diminishing returns, quantified.")
print(f"\n{int((obs['breadth_z'] > 0).sum()):,} taxa above the curve, "
      f"{int((obs['breadth_z'] <= 0).sum()):,} below "
      "(roughly half each, as it must be)")

# exclusive families per taxon: chemistry found in no other taxon
ex = (link[link["family"].isin(excl_fams)].groupby(RANK)["family"].nunique()
      .rename("n_exclusive_families"))
obs = obs.join(ex).fillna({"n_exclusive_families": 0})
obs["n_exclusive_families"] = obs["n_exclusive_families"].astype(int)
obs["pct_exclusive"] = (obs["n_exclusive_families"] / obs["n_families"] * 100).round(1)

meta = (link.groupby(RANK)
        .agg(phylum=("phylum", lambda s: s.mode().iloc[0] if s.notna().any() else None),
             order=("order", lambda s: s.mode().iloc[0] if s.notna().any() else None)))
obs = obs.join(meta)

cols = ["phylum", "n_compounds", "n_families", "exp_families", "breadth_z",
        "breadth_ratio", "n_exclusive_families"]
print("\nBroadest for their effort (top 20):")
print(obs.sort_values("breadth_z", ascending=False).head(20)[cols].to_string())
print("\nNarrowest for their effort (bottom 15) -- heavily worked but "
      "chemically repetitive:")
print(obs.sort_values("breadth_z").head(15)[cols].to_string())

# ------------------------------------------------------------------ 4
section("4. Distinctive chemistry")
print(f"{int((obs['n_exclusive_families'] > 0).sum()):,} taxa produce at least one "
      f"family seen in no other {RANK}")
d = obs[obs["n_exclusive_families"] > 0].sort_values(
    ["n_exclusive_families", "pct_exclusive"], ascending=False)
print("\nTop 20 by count of exclusive families:")
print(d.head(20)[["phylum", "n_compounds", "n_families",
                  "n_exclusive_families", "pct_exclusive", "breadth_z"]].to_string())
if len(obs) > 2:
    cc = obs[["breadth_z", "pct_exclusive"]].corr().iloc[0, 1]
    print(f"\ncorrelation between breadth_z and pct_exclusive: {cc:+.3f}")
    print("Near zero means the two signals are close to independent, so combining "
          "them\nadds information rather than double-counting one trait.")

# ------------------------------------------------------------------ 5
section("5. Candidate priorities")
print("Three signals, combined as an explicit product of percentile ranks so the")
print("weighting stays arguable and recomputable from taxon_profile.parquet:")
print("  breadth    above the accumulation curve for its effort")
print("  exclusive  makes families no other taxon makes")
print("  headroom   little work done so far, so most of it is unexamined")

p = obs.copy()
p["breadth_pct"] = p["breadth_z"].rank(pct=True) * 100
p["exclusive_pct"] = p["pct_exclusive"].rank(pct=True) * 100
p["headroom_pct"] = (1 - p["n_compounds"].rank(pct=True)) * 100
p["priority"] = ((p["breadth_pct"] * p["exclusive_pct"] * p["headroom_pct"])
                 ** (1 / 3)).round(1)

pc = ["phylum", "n_compounds", "n_families", "breadth_z", "n_exclusive_families",
      "pct_exclusive", "headroom_pct", "priority"]
top = p.sort_values("priority", ascending=False)
print(f"\nTop 30 of {len(p):,}:")
print(top.head(30)[pc].to_string())
print("\nphylum spread of the top 100:")
print(top.head(100)["phylum"].value_counts().to_string())
print("\nIf one phylum dominates this list, the ranking may be tracking a "
      "taxonomic\nartefact rather than chemistry. Compare against the phylum "
      "spread of all ranked taxa:")
print(p["phylum"].value_counts().head(10).to_string())

# ------------------------------------------------------------------ 6
section("6. Geography")
gl = (geo_mol[["geo_location_id", "molecule_id"]].dropna().drop_duplicates()
      .merge(geo[["id", "name"]].rename(columns={"id": "geo_location_id"}),
             on="geo_location_id", how="left")
      .merge(work[["molecule_id", "family"]], on="molecule_id", how="inner"))
gl["location"] = gl["name"].map(clean_location)
raw_n = gl["name"].nunique()
gl = gl.dropna(subset=["location"])
print(f"location strings: {raw_n:,} raw -> {gl['location'].nunique():,} after "
      "trimming whitespace and trailing punctuation")
print(f"{gl['molecule_id'].nunique():,} compounds "
      f"({gl['molecule_id'].nunique() / len(work) * 100:.1f}% of those in families) "
      "carry one")
print("\nSynonyms are NOT merged: 'Qingdao' and 'Qingdao, China' remain separate,")
print("which inflates the location count and understates concentration. Resolving")
print("them needs a gazetteer.")

lobs = pd.DataFrame()
if len(gl):
    s = gl.groupby("location")["molecule_id"].nunique()
    conc["location"] = {"n": int(len(s)), "gini": round(gini(s.values), 3),
                        **lorenz(s.values)}
    print(f"\n{len(s):,} locations, Gini {conc['location']['gini']:.3f}")
    print("  " + "  ".join(f"{k}={v}%" for k, v in lorenz(s.values).items()))

    lobs = (gl.groupby("location")
            .agg(n_compounds=("molecule_id", "nunique"),
                 n_families=("family", "nunique")))
    lobs = lobs[lobs["n_compounds"] >= MIN_N].copy()
    if len(lobs) > 5:
        lfit, lz, lr2, lexpo = accumulation_fit(
            lobs["n_compounds"], lobs["n_families"], label="location ")
        lobs["exp_families"] = np.round(lfit, 1)
        lobs["breadth_z"] = np.round(lz, 2)
        print(f"\n{len(lobs):,} locations with >={MIN_N} compounds")
        print("\nBroadest chemistry for their effort (top 15):")
        print(lobs.sort_values("breadth_z", ascending=False).head(15).to_string())
        print("\nMost worked (top 10):")
        print(lobs.sort_values("n_compounds", ascending=False).head(10).to_string())

# ------------------------------------------------------------------ save
save_table(work.reset_index(drop=True), config.INTERIM / "families")
save_table(p.reset_index(), config.INTERIM / "taxon_profile")
if len(lobs):
    save_table(lobs.reset_index(), config.INTERIM / "location_profile")
top.reset_index().to_csv(config.INTERIM / "priority_taxa.csv", index=False)

(config.INTERIM / "bias.json").write_text(json.dumps({
    "tier": config.TIER, "threshold": THR, "resolution": RES,
    "min_family": MINF, "taxon_rank": RANK, "min_compounds_to_rank": MIN_N,
    "null_model": "log-log accumulation curve across taxa, standardised residuals",
    "accumulation_exponent": round(float(expo), 4),
    "accumulation_r2": round(float(r2), 4),
    "n_compounds": n, "n_in_families": int(len(work)),
    "n_families_total": int(len(sizes)), f"n_families_ge{MINF}": int(len(keep)),
    "modularity": round(float(part.modularity), 4),
    "bipartite": {"n_taxa": int(n_tax), "n_families": int(n_fam),
                  "occupied_cells": int(len(inc)),
                  "fill_pct": round(len(inc) / (n_tax * n_fam) * 100, 4),
                  "exclusive_families": int(len(excl_fams))},
    "concentration": conc,
    "n_ranked_taxa": int(len(obs)),
    "n_above_curve": int((obs["breadth_z"] > 0).sum()),
    "n_with_exclusive_families": int((obs["n_exclusive_families"] > 0).sum()),
}, indent=2, default=float))
print(f"\nWrote families, taxon_profile, location_profile, priority_taxa.csv, "
      f"bias.json\nto {config.INTERIM}")
