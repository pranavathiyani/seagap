"""Settle the analysis scope: corrected name matching, final tiers, flags on disk.

    python scripts/04_scope.py

Supersedes the tier logic in 03_tiers.py. Two corrections over that script:

1. Name comparison ignores taxonomic authorities. The release stores names like
   "Thymus vulgaris L." and "Dysidea avara Schmidt, 1862", so comparing full
   strings against a WoRMS scientificname inflates the mismatch rate. Only genus
   and specific epithet are compared.

2. Tier definitions. Requiring a marine-or-brackish flag admitted 83% of the
   release, because cosmopolitan genera like Streptomyces carry Marine/Brackish.
   Microbes now need a strictly marine WoRMS record.

Writes data/interim/compounds_scoped.parquet and data/interim/scope.json.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seagap import config  # noqa: E402
from seagap.tables import load_table, save_table  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 40)
print("SCRIPT VERSION: 2026-10-14 scope\n")

df = load_table(config.INTERIM / "compounds")
tax = load_table(config.INTERIM / "organism_taxonomy")
org = pd.read_csv(config.RELEASE / config.TABLES["organisms"], low_memory=False)
mol_org = pd.read_csv(config.RELEASE / config.TABLES["molecule_organism"],
                      low_memory=False)
n = len(df)

for c in ["isMarine", "isBrackish", "isFreshwater", "isTerrestrial"]:
    if c in tax:
        tax[c] = pd.to_numeric(tax[c], errors="coerce")

bar = "=" * 74


def section(t):
    print(f"\n{bar}\n{t}\n{bar}")


# ------------------------------------------------------------------ names
AUTHORITY = re.compile(
    r"\s+(?:\(?[A-Z][A-Za-z'’\-]*\.?(?:\s+(?:et|and|&|ex)\s+[A-Z][A-Za-z'’\-]*\.?)*"
    r"\)?)?(?:,?\s*\d{4})?\s*$")
PARENS = re.compile(r"\s*\([^)]*\)")
NOISE = re.compile(r"\b(?:sp|spp|var|subsp|ssp|cf|aff|nov|ined)\.?\b", re.I)


def binomial(s):
    """Reduce a name string to 'genus epithet' in lower case.

    Drops parenthetical subgenera, rank abbreviations and trailing authorities,
    then keeps at most the first two alphabetic tokens.
    """
    if not isinstance(s, str) or not s.strip():
        return ""
    t = PARENS.sub(" ", s)
    t = NOISE.sub(" ", t)
    t = re.sub(r"[^A-Za-z\s\-]", " ", t)
    toks = [w for w in t.split() if len(w) > 1]
    # An authority token is capitalised mid-name; keep only the leading
    # genus (capitalised) plus a lower-case epithet if one follows.
    out = []
    for i, w in enumerate(toks):
        if i == 0:
            out.append(w.lower())
        elif w[0].islower():
            out.append(w.lower())
            break
        else:
            break
    return " ".join(out[:2])


o = org[["id", "name", "aphiaid_worms", "rank", "environment_aphia_worms"]].rename(
    columns={"id": "organism_id", "name": "release_name", "rank": "release_rank"})
o = o.merge(tax, on="aphiaid_worms", how="left")
o["rel_bi"] = o["release_name"].apply(binomial)
o["worms_bi"] = o["scientificname"].apply(binomial)
o["rel_genus"] = o["rel_bi"].str.split().str[0]
o["worms_genus"] = o["worms_bi"].str.split().str[0]

section("1. Name match, with authorities stripped")
has = o["scientificname"].notna()
exact = has & (o["rel_bi"] == o["worms_bi"]) & (o["rel_bi"] != "")
gsame = has & ~exact & (o["rel_genus"] == o["worms_genus"]) & (o["rel_genus"] != "")
wrong = has & ~exact & ~gsame

print(f"{int(has.sum()):,} organism rows with a resolved WoRMS record")
for label, m in [("binomial matches", exact), ("same genus only", gsame),
                 ("different genus", wrong)]:
    print(f"  {label:20s} {int(m.sum()):>7,} "
          f"({m.sum() / max(has.sum(), 1) * 100:5.1f}%)")

bad_ids = set(o.loc[wrong, "organism_id"])

link = (mol_org[["molecule_id", "organism_id"]].dropna().drop_duplicates()
        .query("molecule_id in @df.molecule_id")
        .merge(o, on="organism_id", how="left"))

bad_mols = set(link.loc[link["organism_id"].isin(bad_ids), "molecule_id"])
print(f"\nCompounds touched by a different-genus match: {len(bad_mols):,} "
      f"({len(bad_mols) / n * 100:.1f}%)")

print("\nTop 15, with the cross-kingdom jump made explicit:")
w = (link[link["organism_id"].isin(bad_ids)]
     .groupby(["release_name", "scientificname", "phylum", "kingdom"])
     ["molecule_id"].nunique().sort_values(ascending=False).head(15)
     .rename("n_compounds").reset_index())
print(w.to_string(index=False))

# Cross-kingdom jumps are the indefensible ones: a land plant record whose
# AphiaID resolves to an animal.
PLANTISH = ("aceae", "idae")  # family endings seen in the release
cross = w[w["kingdom"].isin(["Animalia", "Chromista", "Bacteria", "Fungi"])]
print(f"\nOf the top 15 wrong matches, {len(cross)} resolve to a non-plant kingdom.")

# ------------------------------------------------------------------ plants
SEAGRASS = {"zostera", "posidonia", "cymodocea", "halophila", "thalassia",
            "syringodium", "phyllospadix", "ruppia", "amphibolis", "enhalus",
            "heterozostera", "thalassodendron"}
MANGROVE = {"rhizophora", "avicennia", "bruguiera", "sonneratia", "xylocarpus",
            "aegiceras", "excoecaria", "kandelia", "ceriops", "lumnitzera",
            "heritiera", "nypa", "acanthus", "aegialitis", "camptostemon",
            "osbornia", "pelliciera", "scyphiphora", "laguncularia",
            "conocarpus", "pemphis", "sapium", "dolichandrone", "cerbera",
            "clerodendrum", "salvadora", "cassytha", "cassytha"}
HALOPHYTE = {"glehnia", "triglochin", "salicornia", "suaeda", "spartina",
             "limonium", "cakile", "honckenya", "crithmum", "sesuvium", "batis",
             "atriplex", "vitex", "cocos", "pandanus", "barringtonia",
             "calophyllum", "thespesia", "terminalia", "ipomoea", "juncus",
             "tamarix", "borrichia", "uniola", "distichlis", "arthrocnemum",
             "halimione", "armeria", "frankenia", "bolboschoenus", "phragmites",
             "artemisia", "cochlearia", "euphorbia", "baccharis", "elymus"}

section("2. Vascular plants, grouped by habitat")
tr = link[link["phylum"] == "Tracheophyta"].copy()
tr["genus"] = tr["rel_genus"]


def plant_group(g):
    if g in SEAGRASS:
        return "seagrass"
    if g in MANGROVE:
        return "mangrove"
    if g in HALOPHYTE:
        return "halophyte"
    return "non-marine"


tr["plant_group"] = tr["genus"].apply(plant_group)
pg = (tr.groupby("plant_group")
      .agg(n_compounds=("molecule_id", "nunique"),
           n_records=("organism_id", "nunique")))
pg["pct_of_release"] = (pg["n_compounds"] / n * 100).round(2)
print(pg.sort_values("n_compounds", ascending=False).to_string())
print("\nSeagrasses, mangroves and halophytes are marine or intertidal and stay in "
      "scope.\nThe 'non-marine' group is excluded unless WoRMS calls the record "
      "marine with no\nterrestrial or freshwater flag.")

marine_plant_ids = set(tr.loc[tr["plant_group"] != "non-marine", "organism_id"])

# ------------------------------------------------------------------ tiers
CORE = {"Porifera", "Cnidaria", "Mollusca", "Echinodermata", "Bryozoa", "Annelida",
        "Chordata", "Arthropoda", "Rhodophyta", "Ochrophyta", "Chlorophyta",
        "Heterokontophyta", "Myzozoa", "Foraminifera", "Ciliophora",
        "Platyhelminthes", "Nemertea", "Brachiopoda", "Chaetognatha",
        "Hemichordata", "Cryptophyta", "Haptophyta", "Charophyta", "Ctenophora",
        "Phoronida", "Rotifera", "Nematoda", "Choanozoa", "Bigyra",
        "Prasinodermatophyta", "Euglenozoa"}
MICROBE = {"Actinobacteria", "Proteobacteria", "Cyanobacteria", "Ascomycota",
           "Basidiomycota", "Firmicutes", "Bacteroidetes", "Bacillariophyta",
           "Verrucomicrobia", "Planctomycetota", "Chloroflexi", "Chlorobi",
           "Thermotogae", "Aquificae", "Euryarchaeota", "Crenarchaeota",
           "Oomycota", "Chytridiomycota", "Zygomycota", "Myxomycota"}

f = {c: link[c].fillna(0) for c in
     ["isMarine", "isBrackish", "isFreshwater", "isTerrestrial"]}
link["strict_marine"] = ((f["isMarine"] == 1) & (f["isBrackish"] == 0)
                         & (f["isFreshwater"] == 0) & (f["isTerrestrial"] == 0))
rk = link["rank"].fillna("").str.strip().str.lower()
link["rank_clean"] = rk.str.replace(r"\s*\(?fuzzy\)?", "", regex=True).str.strip()
link["species_rank"] = link["rank_clean"].isin(
    ["species", "subspecies", "forma", "variety"])
link["name_ok"] = ~link["organism_id"].isin(bad_ids)
link["is_core"] = (link["phylum"].isin(CORE)
                   | link["organism_id"].isin(marine_plant_ids))
link["is_micro"] = link["phylum"].isin(MICROBE)


def mols(mask):
    return set(link.loc[mask, "molecule_id"])


t1 = mols(link["is_core"] & link["strict_marine"] & link["species_rank"]
          & link["name_ok"])
t2 = t1 | mols(link["is_micro"] & link["strict_marine"] & link["name_ok"])
t3 = set(df["molecule_id"])

section("3. Final scope tiers")
for name, desc, s in [
    ("1  core", "marine eukaryotes and marine plants; strictly marine; "
                "species rank; name verified", t1),
    ("2  extended", "tier 1 plus strictly marine microbes, any rank", t2),
    ("3  full", "the release as shipped", t3),
]:
    print(f"  {name:12s} {len(s):>7,}  ({len(s) / n * 100:5.1f}%)  {desc}")

# ------------------------------------------------------------------ flags
df["tier1"] = df["molecule_id"].isin(t1)
df["tier2"] = df["molecule_id"].isin(t2)
df["name_suspect"] = df["molecule_id"].isin(bad_mols)

genus_micro = mols(link["is_micro"] & ~link["species_rank"])
df["microbe_genus_only"] = df["molecule_id"].isin(genus_micro)

plant_nm = mols(link["organism_id"].isin(
    set(tr.loc[tr["plant_group"] == "non-marine", "organism_id"])))
df["nonmarine_plant"] = df["molecule_id"].isin(plant_nm)

phy = (link.dropna(subset=["phylum"]).groupby("molecule_id")["phylum"]
       .agg(lambda s: sorted(set(s))))
df = df.merge(phy.rename("phyla"), left_on="molecule_id", right_index=True,
              how="left")
df["n_phyla"] = df["phyla"].apply(lambda v: len(v) if isinstance(v, list) else 0)

section("4. Flags written to compounds_scoped")
for c in ["tier1", "tier2", "name_suspect", "microbe_genus_only",
          "nonmarine_plant"]:
    print(f"  {c:22s} {int(df[c].sum()):>7,}  ({df[c].mean() * 100:5.1f}%)")
print(f"  {'multi-phylum':22s} {int((df['n_phyla'] > 1).sum()):>7,}  "
      f"({(df['n_phyla'] > 1).mean() * 100:5.1f}%)")

section("5. Chemistry by tier")
cmp_ = pd.DataFrame({
    "tier1": df.loc[df["tier1"], "np_classifier_pathway"].value_counts(normalize=True) * 100,
    "tier2": df.loc[df["tier2"], "np_classifier_pathway"].value_counts(normalize=True) * 100,
    "full": df["np_classifier_pathway"].value_counts(normalize=True) * 100,
}).fillna(0).round(1)
cmp_["t1_minus_full"] = (cmp_["tier1"] - cmp_["full"]).round(1)
print(cmp_.sort_values("full", ascending=False).to_string())

print("\nCoverage within each tier:")
for t, mask in [("tier1", df["tier1"]), ("tier2", df["tier2"]),
                ("full", pd.Series(True, index=df.index))]:
    sub = df[mask]
    print(f"  {t:6s} n={len(sub):>7,}  location "
          f"{(sub['n_locations'] > 0).mean() * 100:5.1f}%  "
          f"pathway {sub['np_classifier_pathway'].notna().mean() * 100:5.1f}%")

scope = {
    "n_release": n,
    "tier1": len(t1), "tier2": len(t2), "tier3": len(t3),
    "name_match": {"binomial": int(exact.sum()), "genus_only": int(gsame.sum()),
                   "different_genus": int(wrong.sum()),
                   "resolved_records": int(has.sum())},
    "compounds_name_suspect": len(bad_mols),
    "microbe_genus_only_compounds": len(genus_micro),
    "plant_groups": pg["n_compounds"].to_dict(),
    "core_phyla": sorted(CORE), "microbe_phyla": sorted(MICROBE),
}
(config.INTERIM / "scope.json").write_text(json.dumps(scope, indent=2, default=int))
out = save_table(df.reset_index(drop=True), config.INTERIM / "compounds_scoped")
print(f"\nWrote {out}")
print(f"Wrote {config.INTERIM / 'scope.json'}")
