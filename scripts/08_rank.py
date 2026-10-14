"""Diagnose the study-type confound in breadth, then re-rank on chemical rarity.

    python scripts/08_rank.py

Step 07 ranked food fish and a Steller sea lion above every sponge. The reason is
not a coding error: a nutritional study of a fish reports a dozen unrelated
primary metabolites, one fatty acid, one sterol, one sphingolipid, and each falls
in a different structural family, so breadth is high. A natural products study of
a sponge reports fifty related diterpenes, which is one family. Breadth therefore
measures the kind of study performed as much as the chemistry available.

Section 1 tests that explanation rather than assuming it. Sections 2 and 3 replace
the two weak parts of the ranking:

  binary exclusivity -> continuous family rarity. 819 of 938 taxa had zero
  exclusive families, so percentile ranking gave them all the same mid value and
  the signal did no work. Mean -log10(prevalence) over a taxon's families has no
  tie mass and says directly whether a taxon occupies rare chemical space.

  unfiltered breadth -> breadth computed inside specialised-metabolite families
  only, with the primary-metabolite fraction reported so the filter can be
  checked rather than trusted.

Writes taxon_ranked.parquet, family_profile.parquet and rank.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seagap import config  # noqa: E402
from seagap.tables import load_table, save_table  # noqa: E402

pd.set_option("display.width", 215)
pd.set_option("display.max_colwidth", 38)
print("SCRIPT VERSION: 2026-10-14 rank\n")

RANK = config.TAXON_RANK
MIN_N = 10

# Pathways dominated by primary metabolism. Amino acids and Peptides stays on the
# specialised side on purpose: marine cyclic peptides and depsipeptides are among
# the most important natural products from this biome, so excluding that pathway
# would remove real targets to fix a different problem.
PRIMARY = {"Fatty acids", "Carbohydrates"}

bar = "=" * 96


def section(t):
    print(f"\n{bar}\n{t}\n{bar}")


def accumulation_fit(n_units, n_classes):
    x = np.log10(np.asarray(n_units, dtype=float))
    y = np.log10(np.asarray(n_classes, dtype=float))
    b1, b0 = np.polyfit(x, y, 1)
    resid = y - (b0 + b1 * x)
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return 10 ** (b0 + b1 * x), resid / resid.std(ddof=1), r2, b1


# ------------------------------------------------------------------ load
work = load_table(config.INTERIM / "families")
tax = load_table(config.INTERIM / "organism_taxonomy")
org = pd.read_csv(config.RELEASE / config.TABLES["organisms"], low_memory=False)
mol_org = pd.read_csv(config.RELEASE / config.TABLES["molecule_organism"],
                      low_memory=False)

link = (mol_org[["molecule_id", "organism_id"]].dropna().drop_duplicates()
        .merge(org[["id", "aphiaid_worms"]].rename(columns={"id": "organism_id"}),
               on="organism_id", how="left")
        .merge(tax[["aphiaid_worms", "genus", "order", "class", "phylum"]],
               on="aphiaid_worms", how="left")
        .merge(work[["molecule_id", "family", "np_classifier_pathway",
                     "np_classifier_superclass"]],
               on="molecule_id", how="inner")
        .dropna(subset=[RANK])
        .drop_duplicates(subset=["molecule_id", RANK, "family"]))
print(f"{len(link):,} (compound, {RANK}, family) rows, "
      f"{link[RANK].nunique():,} {RANK} values")

# ------------------------------------------------------------------ families
fam = (link.groupby("family")
       .agg(n_compounds=("molecule_id", "nunique"),
            n_taxa=("genus", "nunique"),
            pathway=("np_classifier_pathway",
                     lambda s: s.mode().iloc[0] if s.notna().any() else None),
            path_purity=("np_classifier_pathway",
                         lambda s: round(s.value_counts(normalize=True).iloc[0], 3)
                         if s.notna().any() else None),
            superclass=("np_classifier_superclass",
                        lambda s: s.mode().iloc[0] if s.notna().any() else None)))
n_taxa_total = link[RANK].nunique()
fam["prevalence"] = fam["n_taxa"] / n_taxa_total
fam["rarity"] = -np.log10(fam["prevalence"])
fam["is_primary"] = fam["pathway"].isin(PRIMARY)

print(f"\n{len(fam):,} families")
print(f"{int(fam['is_primary'].sum()):,} are dominated by a primary-metabolism "
      f"pathway ({', '.join(sorted(PRIMARY))})")
print("\nfamilies by dominant pathway:")
print(fam["pathway"].value_counts().to_string())

# ------------------------------------------------------------------ 1
section("1. Is breadth tracking study type?")
prof = load_table(config.INTERIM / "taxon_profile").set_index(RANK)

prim = (link.assign(primary=link["family"].map(fam["is_primary"]))
        .groupby(RANK)
        .agg(n_compounds=("molecule_id", "nunique"),
             primary_frac=("primary", "mean"),
             mean_rarity=("family", lambda s: fam.loc[s.unique(), "rarity"].mean()),
             mean_fam_size=("family", lambda s: fam.loc[s.unique(), "n_compounds"].mean())))
prim = prim[prim["n_compounds"] >= MIN_N]
prim = prim.join(prof[["n_families", "breadth_z", "phylum"]], how="inner")
prim["primary_frac"] = prim["primary_frac"].round(3)
prim["mean_rarity"] = prim["mean_rarity"].round(3)

r = prim[["breadth_z", "primary_frac"]].corr().iloc[0, 1]
print(f"correlation, breadth_z vs primary-metabolite fraction: {r:+.3f}")
r2_ = prim[["breadth_z", "mean_rarity"]].corr().iloc[0, 1]
print(f"correlation, breadth_z vs mean family rarity:          {r2_:+.3f}")
print("\nIf the first is clearly positive and the second is not, high breadth is "
      "a marker\nof primary-metabolite surveys rather than of rare chemistry.")

print("\nBy breadth decile:")
prim["decile"] = pd.qcut(prim["breadth_z"], 10, labels=False, duplicates="drop") + 1
print(prim.groupby("decile").agg(
    n_taxa=("n_families", "size"),
    median_compounds=("n_compounds", "median"),
    primary_frac=("primary_frac", "mean"),
    mean_rarity=("mean_rarity", "mean"),
    mean_fam_size=("mean_fam_size", "mean")).round(3).to_string())

FISH = ["Stenotomus", "Coregonus", "Brosme", "Molva", "Lophius", "Chanos",
        "Salmo", "Eumetopias", "Peprilus", "Ophiodon", "Lopholatilus",
        "Leiostomus", "Chlorurus"]
SPONGE = ["Theonella", "Plakortis", "Dysidea", "Agelas", "Xestospongia",
          "Callyspongia", "Axinyssa", "Amphimedon", "Melophlus"]
print("\nNamed comparison, the taxa that topped the old ranking against "
      "well-known\nnatural-products sponges:")
for lab, names in [("food fish / mammal", FISH), ("sponge", SPONGE)]:
    sel = prim[prim.index.isin(names)]
    if len(sel):
        print(f"  {lab:20s} n={len(sel):>3}  breadth_z {sel['breadth_z'].mean():+.2f}"
              f"  primary_frac {sel['primary_frac'].mean():.3f}"
              f"  mean_rarity {sel['mean_rarity'].mean():.3f}"
              f"  median compounds {sel['n_compounds'].median():.0f}")

# ------------------------------------------------------------------ 2
section("2. Breadth inside specialised-metabolite families only")
spec_fams = set(fam[~fam["is_primary"]].index)
sl = link[link["family"].isin(spec_fams)]
sobs = (sl.groupby(RANK)
        .agg(n_compounds=("molecule_id", "nunique"),
             n_families=("family", "nunique")))
sobs = sobs[sobs["n_compounds"] >= MIN_N].copy()
sfit, sz, sr2, sexpo = accumulation_fit(sobs["n_compounds"], sobs["n_families"])
sobs["exp_families"] = np.round(sfit, 1)
sobs["spec_breadth_z"] = np.round(sz, 2)
print(f"{len(sobs):,} {RANK} values with >={MIN_N} specialised-metabolite compounds")
print(f"accumulation curve: families = n^{sexpo:.3f}, R2 = {sr2:.3f}")

cmp_ = prof[["breadth_z", "phylum", "n_compounds"]].join(
    sobs[["spec_breadth_z"]], how="inner")
print(f"\ncorrelation between the two breadth measures: "
      f"{cmp_[['breadth_z', 'spec_breadth_z']].corr().iloc[0, 1]:+.3f}")
moved = cmp_.assign(shift=(cmp_["spec_breadth_z"] - cmp_["breadth_z"]).round(2))
print("\nLargest falls when primary-metabolite families are removed:")
print(moved.sort_values("shift").head(12).to_string())

# ------------------------------------------------------------------ 3
section("3. Re-ranked on rarity, specialised breadth and headroom")
R = sobs.join(prim[["primary_frac", "mean_rarity", "mean_fam_size", "phylum"]],
              how="inner")
R = R.dropna(subset=["mean_rarity"])

# Rarity recomputed over specialised families only, so the two axes agree on scope.
sr = (sl.groupby(RANK)["family"]
      .agg(lambda s: fam.loc[s.unique(), "rarity"].mean()).rename("spec_rarity"))
R = R.join(sr, how="left")
R["spec_rarity"] = R["spec_rarity"].round(3)

R["breadth_pct"] = R["spec_breadth_z"].rank(pct=True) * 100
R["rarity_pct"] = R["spec_rarity"].rank(pct=True) * 100
R["headroom_pct"] = (1 - R["n_compounds"].rank(pct=True)) * 100
R["priority"] = ((R["breadth_pct"] * R["rarity_pct"] * R["headroom_pct"])
                 ** (1 / 3)).round(1)

for c in ["breadth_pct", "rarity_pct", "headroom_pct"]:
    ties = R[c].value_counts().iloc[0]
    print(f"  {c:14s} largest tie group: {ties:,} of {len(R):,} "
          f"({ties / len(R) * 100:.1f}%)")
print("A large tie group means that axis is not discriminating. This was the "
      "defect in\nthe binary exclusivity score, where 87% of taxa shared one "
      "value.")

cols = ["phylum", "n_compounds", "n_families", "spec_breadth_z", "spec_rarity",
        "primary_frac", "priority"]
top = R.sort_values("priority", ascending=False)
print(f"\nTop 30 of {len(R):,}:")
print(top.head(30)[cols].to_string())

print("\nphylum spread, top 100 vs all ranked:")
spread = pd.DataFrame({
    "top100": top.head(100)["phylum"].value_counts(),
    "all": R["phylum"].value_counts()}).fillna(0).astype(int)
spread["enrichment"] = (spread["top100"] / 100
                        / (spread["all"] / len(R))).round(2)
print(spread.sort_values("top100", ascending=False).head(12).to_string())
print("\nEnrichment near 1 means a phylum appears in the top 100 at its "
      "background rate.")

print("\nHow many of the old top 30 survive:")
old = ["Placida", "Polymastia", "Comactinia", "Echinaster", "Watersipora",
       "Lithophyllum", "Chlorurus", "Echinometra", "Carinaria", "Verongula",
       "Polycitor", "Muricea", "Thermotoga", "Lanice", "Xylocarpus", "Brosme",
       "Chanos", "Eumetopias", "Ophiodon", "Lophius", "Molva", "Peprilus",
       "Pseudomonas", "Chalina", "Arelia", "Uca", "Svenzea", "Dasycladus",
       "Ruditapes", "Coscinoderma"]
new_top = set(top.head(30).index)
kept = [o for o in old if o in new_top]
print(f"  {len(kept)} of 30: {', '.join(kept) if kept else '(none)'}")

# ------------------------------------------------------------------ save
save_table(fam.reset_index(), config.INTERIM / "family_profile")
save_table(R.reset_index(), config.INTERIM / "taxon_ranked")
top.reset_index().to_csv(config.INTERIM / "priority_taxa.csv", index=False)
(config.INTERIM / "rank.json").write_text(json.dumps({
    "taxon_rank": RANK, "min_compounds": MIN_N,
    "primary_pathways": sorted(PRIMARY),
    "n_families": int(len(fam)),
    "n_primary_families": int(fam["is_primary"].sum()),
    "n_specialised_families": int(len(spec_fams)),
    "corr_breadth_primaryfrac": round(float(r), 4),
    "corr_breadth_rarity": round(float(r2_), 4),
    "spec_accumulation_exponent": round(float(sexpo), 4),
    "spec_accumulation_r2": round(float(sr2), 4),
    "n_ranked": int(len(R)),
}, indent=2, default=float))
print(f"\nWrote family_profile, taxon_ranked, priority_taxa.csv, rank.json\n"
      f"to {config.INTERIM}")
