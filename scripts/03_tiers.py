"""Attach WoRMS taxonomy to compounds, weight by compound, and assign scope tiers.

    python scripts/03_tiers.py

The release's own marine flag carries no information (every row is 't'), and its
environment_aphia_worms string is assigned at whatever rank the WoRMS record sits
at, so a genus containing some marine members marks every isolate as marine. This
step replaces that with real phylum-level and flag-level evidence, then cuts three
nested scopes so later results can be reported with sensitivity across them.

Writes data/interim/compounds_tiered.parquet and data/interim/tiers.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seagap import config  # noqa: E402
from seagap.tables import load_table, save_table  # noqa: E402

pd.set_option("display.width", 200)
print("SCRIPT VERSION: 2026-10-14 tiers\n")

df = load_table(config.INTERIM / "compounds")
tax = load_table(config.INTERIM / "organism_taxonomy")
org = pd.read_csv(config.RELEASE / config.TABLES["organisms"], low_memory=False)
mol_org = pd.read_csv(config.RELEASE / config.TABLES["molecule_organism"],
                      low_memory=False)

n = len(df)
print(f"{n:,} compounds, {len(tax):,} taxonomy records, {len(org):,} organism rows")

# ------------------------------------------------------------------ link
link = mol_org[["molecule_id", "organism_id"]].dropna().drop_duplicates()
link = link[link["molecule_id"].isin(df["molecule_id"])]
link = link.merge(
    org[["id", "name", "rank", "aphiaid_worms"]].rename(
        columns={"id": "organism_id", "name": "organism_name",
                 "rank": "release_rank"}),
    on="organism_id", how="left")

for c in ["isMarine", "isBrackish", "isFreshwater", "isTerrestrial"]:
    if c in tax:
        tax[c] = pd.to_numeric(tax[c], errors="coerce")

link = link.merge(tax, on="aphiaid_worms", how="left")
link["genus_name"] = link["organism_name"].fillna("").str.strip().str.split().str[0]

print(f"{len(link):,} molecule-organism links")
no_aphia = link["aphiaid_worms"].isna()
print(f"  {int(no_aphia.sum()):,} links have no AphiaID "
      f"({no_aphia.mean() * 100:.1f}%)")
unresolved = link["phylum"].isna() & ~no_aphia
print(f"  {int(unresolved.sum()):,} have an AphiaID that did not resolve")

# ------------------------------------------------------------------ by compound
print("\n" + "=" * 72)
print("Phylum, weighted by distinct compounds (not organisms)")
print("=" * 72)

byphy = (link.dropna(subset=["phylum"])
         .groupby("phylum")
         .agg(n_compounds=("molecule_id", "nunique"),
              n_organisms=("organism_id", "nunique"))
         .sort_values("n_compounds", ascending=False))
byphy["pct_compounds"] = (byphy["n_compounds"] / n * 100).round(2)
byphy["compounds_per_organism"] = (byphy["n_compounds"]
                                   / byphy["n_organisms"]).round(1)
print(byphy.head(25).to_string())

print("\nKingdom:")
print(link.dropna(subset=["kingdom"]).groupby("kingdom")["molecule_id"]
      .nunique().sort_values(ascending=False).to_string())

# ------------------------------------------------------------------ plants
print("\n" + "=" * 72)
print("Vascular plants (Tracheophyta): the clearest contamination")
print("=" * 72)
tr = link[link["phylum"] == "Tracheophyta"]
if len(tr):
    plant_mols = set(tr["molecule_id"])
    print(f"{len(plant_mols):,} compounds ({len(plant_mols) / n * 100:.2f}%) "
          f"from {tr['organism_id'].nunique():,} plant records")
    print("\nTop 15 plant sources, with the WoRMS environment the release inherited:")
    t = (tr.groupby("organism_name")
         .agg(n_compounds=("molecule_id", "nunique"),
              isMarine=("isMarine", "first"),
              isTerrestrial=("isTerrestrial", "first"),
              rank=("rank", "first"))
         .sort_values("n_compounds", ascending=False).head(15))
    print(t.to_string())
else:
    print("none found")

# ------------------------------------------------------------------ rank
print("\n" + "=" * 72)
print("Record rank, weighted by compound")
print("=" * 72)
rk = link["rank"].fillna("(unresolved)").str.strip().str.lower()
link["rank_clean"] = rk.str.replace(r"\s*\(?fuzzy\)?", "", regex=True).str.strip()
print(link.groupby("rank_clean")["molecule_id"].nunique()
      .sort_values(ascending=False).head(12).to_string())
print("\nA genus-rank record cannot tell a marine isolate from a soil one.")

# ------------------------------------------------------------------ tiers
CORE = {  # unambiguously marine macro-organisms and algae
    "Porifera", "Cnidaria", "Mollusca", "Echinodermata", "Bryozoa",
    "Annelida", "Chordata", "Arthropoda", "Rhodophyta", "Ochrophyta",
    "Chlorophyta", "Heterokontophyta", "Myzozoa", "Foraminifera",
    "Ciliophora", "Platyhelminthes", "Nemertea", "Brachiopoda",
    "Chaetognatha", "Hemichordata", "Porifera incertae sedis",
}
MICROBE = {  # marine-derived microbes: real field, weaker provenance
    "Actinobacteria", "Proteobacteria", "Cyanobacteria", "Ascomycota",
    "Basidiomycota", "Firmicutes", "Bacteroidetes", "Bacillariophyta",
    "Mycota", "Myxomycota", "Chytridiomycota", "Zygomycota",
}
EXCLUDE = {"Tracheophyta", "Bryophyta", "Marchantiophyta", "Anthophyta"}

link["grp"] = np.select(
    [link["phylum"].isin(CORE), link["phylum"].isin(MICROBE),
     link["phylum"].isin(EXCLUDE)],
    ["core", "microbe", "excluded"], default="other")

print("\n" + "=" * 72)
print("Phylum groups, by compound")
print("=" * 72)
g = (link.groupby("grp")
     .agg(n_compounds=("molecule_id", "nunique"),
          n_organisms=("organism_id", "nunique")))
g["pct"] = (g["n_compounds"] / n * 100).round(2)
print(g.to_string())
print("\n('other' is phyla in neither list plus links with no resolved phylum; "
      "the next block names them)")
oth = link[link["grp"] == "other"].dropna(subset=["phylum"])
if len(oth):
    print("\nUnclassified phyla present:")
    print(oth.groupby("phylum")["molecule_id"].nunique()
          .sort_values(ascending=False).head(20).to_string())

# strictly marine: isMarine=1 and the other three 0
f = {c: link[c].fillna(0) for c in
     ["isMarine", "isBrackish", "isFreshwater", "isTerrestrial"]}
link["strict_marine"] = ((f["isMarine"] == 1) & (f["isBrackish"] == 0)
                         & (f["isFreshwater"] == 0) & (f["isTerrestrial"] == 0))
link["any_marine"] = (f["isMarine"] == 1) | (f["isBrackish"] == 1)
link["species_rank"] = link["rank_clean"].eq("species")

# A compound qualifies for a tier if ANY of its organism links qualifies.
def mols(mask):
    return set(link.loc[mask, "molecule_id"])


tier_a = mols((link["grp"] == "core") & link["strict_marine"] & link["species_rank"])
tier_b = tier_a | mols(link["grp"].isin(["core", "microbe"]) & link["any_marine"])
tier_c = set(df["molecule_id"])

plant_only = mols(link["grp"] == "excluded") - mols(link["grp"] != "excluded")

print("\n" + "=" * 72)
print("Scope tiers")
print("=" * 72)
rows = [
    ("A  strict", "core marine phyla, strictly marine WoRMS flags, species rank",
     len(tier_a)),
    ("B  broad", "A, plus marine-derived microbes with a marine/brackish flag",
     len(tier_b)),
    ("C  as-shipped", "the full release, unfiltered", len(tier_c)),
]
for name, desc, k in rows:
    print(f"  {name:14s} {k:>7,}  ({k / n * 100:5.1f}%)   {desc}")
print(f"\n  compounds whose ONLY source is a vascular plant: {len(plant_only):,} "
      f"({len(plant_only) / n * 100:.2f}%)")

# ------------------------------------------------------------------ attach
df["tier_a"] = df["molecule_id"].isin(tier_a)
df["tier_b"] = df["molecule_id"].isin(tier_b)
df["plant_only"] = df["molecule_id"].isin(plant_only)

phy = (link.dropna(subset=["phylum"]).groupby("molecule_id")["phylum"]
       .agg(lambda s: sorted(set(s))))
df = df.merge(phy.rename("phyla"), left_on="molecule_id", right_index=True, how="left")
df["n_phyla"] = df["phyla"].apply(lambda v: len(v) if isinstance(v, list) else 0)

multi = int((df["n_phyla"] > 1).sum())
print(f"\n{multi:,} compounds ({multi / n * 100:.1f}%) are reported from more than "
      "one phylum")
print("  (symbiont-vs-host ambiguity, or genuine cross-taxon occurrence)")

# ------------------------------------------------------------------ what tier A keeps
print("\n" + "=" * 72)
print("Tier A composition")
print("=" * 72)
a = df[df["tier_a"]]
if len(a):
    ex = link[link["molecule_id"].isin(tier_a)]
    print("Phyla:")
    print(ex[ex["grp"] == "core"].groupby("phylum")["molecule_id"].nunique()
          .sort_values(ascending=False).head(15).to_string())
    print(f"\nCoverage within tier A ({len(a):,} compounds):")
    print(f"  with a location          {int((a['n_locations'] > 0).sum()):,} "
          f"({(a['n_locations'] > 0).mean() * 100:.1f}%)")
    print(f"  with NPClassifier path   "
          f"{int(a['np_classifier_pathway'].notna().sum()):,} "
          f"({a['np_classifier_pathway'].notna().mean() * 100:.1f}%)")
    print("\nNPClassifier pathway, tier A vs full release:")
    comp = pd.DataFrame({
        "tier_A_pct": (a["np_classifier_pathway"].value_counts(normalize=True) * 100),
        "full_pct": (df["np_classifier_pathway"].value_counts(normalize=True) * 100),
    }).fillna(0.0).round(1)
    comp["shift"] = (comp["tier_A_pct"] - comp["full_pct"]).round(1)
    print(comp.sort_values("full_pct", ascending=False).to_string())
    print("\nA large shift means the scope decision changes which chemistry you "
          "are mapping, not just how much.")

summary = {
    "n_compounds": n,
    "tier_a": len(tier_a),
    "tier_b": len(tier_b),
    "tier_c": len(tier_c),
    "plant_only": len(plant_only),
    "links_without_aphiaid": int(no_aphia.sum()),
    "multi_phylum_compounds": multi,
    "phylum_by_compound": byphy["n_compounds"].to_dict(),
    "core_phyla": sorted(CORE),
    "microbe_phyla": sorted(MICROBE),
    "excluded_phyla": sorted(EXCLUDE),
}
(config.INTERIM / "tiers.json").write_text(json.dumps(summary, indent=2, default=int))
out = save_table(df.reset_index(drop=True), config.INTERIM / "compounds_tiered")
print(f"\nWrote {out}")
print(f"Wrote {config.INTERIM / 'tiers.json'}")
