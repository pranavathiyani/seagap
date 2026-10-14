"""Audit WoRMS name matching, reclassify plants, and retune the scope tiers.

    python scripts/03b_review.py

Three jobs, all diagnostic. Writes nothing except a candidate-tier table.

1. Name matching. The release stores an AphiaID per organism. If the WoRMS record
   behind that ID has a different scientific name, the match is wrong and every
   downstream property inherited from it is wrong too. Thymus vulgaris carries 374
   compounds and a marine flag, yet does not appear under Tracheophyta, which can
   only mean its AphiaID points somewhere else.

2. Plants. Mangroves, seagrasses and coastal halophytes are legitimately marine or
   intertidal, so Tracheophyta is not uniformly contamination. Crops and freshwater
   plants are. This separates them.

3. Tier B. Requiring only a marine-or-brackish flag admits 96% of the release,
   which makes it indistinguishable from the unfiltered set. This tests stricter
   definitions and reports what each one costs.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seagap import config  # noqa: E402
from seagap.tables import load_table  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 44)
print("SCRIPT VERSION: 2026-10-14 review\n")

df = load_table(config.INTERIM / "compounds")
tax = load_table(config.INTERIM / "organism_taxonomy")
org = pd.read_csv(config.RELEASE / config.TABLES["organisms"], low_memory=False)
mol_org = pd.read_csv(config.RELEASE / config.TABLES["molecule_organism"],
                      low_memory=False)
n = len(df)

for c in ["isMarine", "isBrackish", "isFreshwater", "isTerrestrial"]:
    if c in tax:
        tax[c] = pd.to_numeric(tax[c], errors="coerce")

link = (mol_org[["molecule_id", "organism_id"]].dropna().drop_duplicates()
        .query("molecule_id in @df.molecule_id")
        .merge(org[["id", "name", "rank", "aphiaid_worms", "name_aphia_worms"]]
               .rename(columns={"id": "organism_id", "name": "release_name",
                                "rank": "release_rank"}),
               on="organism_id", how="left")
        .merge(tax, on="aphiaid_worms", how="left"))

bar = "=" * 74


def section(t):
    print(f"\n{bar}\n{t}\n{bar}")


# ------------------------------------------------------------------ 1
section("1. Does the AphiaID actually match the organism name?")


def norm(s):
    return (s.fillna("").str.strip().str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.replace(r"\s*\(.*?\)", "", regex=True)
            .str.replace(r"\s+(sp|spp)\.?$", "", regex=True))


o = org.copy()
o = o.merge(tax[["aphiaid_worms", "scientificname", "phylum", "kingdom", "rank"]],
            on="aphiaid_worms", how="left", suffixes=("", "_worms"))
o["rel_n"] = norm(o["name"])
o["worms_n"] = norm(o["scientificname"])
o["genus_rel"] = o["rel_n"].str.split().str[0]
o["genus_worms"] = o["worms_n"].str.split().str[0]

has = o["scientificname"].notna()
exact = has & (o["rel_n"] == o["worms_n"])
genus_ok = has & ~exact & (o["genus_rel"] == o["genus_worms"])
mismatch = has & ~exact & ~genus_ok

print(f"{int(has.sum()):,} organism rows with a resolved WoRMS record")
print(f"  exact name match          {int(exact.sum()):>7,} "
      f"({exact.sum() / max(has.sum(), 1) * 100:5.1f}%)")
print(f"  same genus, diff epithet  {int(genus_ok.sum()):>7,} "
      f"({genus_ok.sum() / max(has.sum(), 1) * 100:5.1f}%)")
print(f"  different genus           {int(mismatch.sum()):>7,} "
      f"({mismatch.sum() / max(has.sum(), 1) * 100:5.1f}%)  <-- wrong match")

bad_ids = set(o.loc[mismatch, "organism_id"] if "organism_id" in o else o.loc[mismatch, "id"])
bad_links = link[link["organism_id"].isin(bad_ids)]
bad_mols = set(bad_links["molecule_id"])
print(f"\nCompounds touched by a wrong-genus match: {len(bad_mols):,} "
      f"({len(bad_mols) / n * 100:.1f}%)")

if mismatch.any():
    idcol = "organism_id" if "organism_id" in o else "id"
    worst = (bad_links.groupby("organism_id")["molecule_id"].nunique()
             .sort_values(ascending=False).head(20).rename("n_compounds")
             .reset_index()
             .merge(o[[idcol, "name", "scientificname", "phylum", "kingdom"]]
                    .rename(columns={idcol: "organism_id"}),
                    on="organism_id", how="left"))
    print("\nWorst wrong matches by compound count:")
    print(worst[["name", "scientificname", "phylum", "kingdom",
                 "n_compounds"]].to_string(index=False))

print("\nIs Thymus vulgaris among them?")
th = o[o["rel_n"].str.contains("thymus", na=False)]
if len(th):
    print(th[["name", "aphiaid_worms", "scientificname", "phylum", "kingdom",
              "environment_aphia_worms"]].head(5).to_string(index=False))
else:
    print("  not found in organisms.csv")

# ------------------------------------------------------------------ 2
section("2. Plants: marine-associated versus genuinely terrestrial")

SEAGRASS = {"zostera", "posidonia", "cymodocea", "halophila", "thalassia",
            "syringodium", "syringodium", "phyllospadix", "ruppia", "amphibolis",
            "enhalus", "heterozostera", "thalassodendron"}
MANGROVE = {"rhizophora", "avicennia", "bruguiera", "sonneratia", "xylocarpus",
            "aegiceras", "excoecaria", "excoecaria", "kandelia", "ceriops",
            "lumnitzera", "heritiera", "nypa", "acanthus", "aegialitis",
            "camptostemon", "osbornia", "pelliciera", "scyphiphora",
            "laguncularia", "conocarpus", "pemphis", "sapium", "dolichandrone"}
HALOPHYTE = {"glehnia", "triglochin", "salicornia", "suaeda", "spartina",
             "limonium", "cakile", "honckenya", "crithmum", "sesuvium", "batis",
             "atriplex", "vitex", "cocos", "pandanus", "barringtonia",
             "calophyllum", "thespesia", "terminalia", "ipomoea", "juncus",
             "tamarix", "borrichia", "uniola", "distichlis", "arthrocnemum",
             "halimione", "aster", "plantago", "armeria", "frankenia"}

tr = link[link["phylum"] == "Tracheophyta"].copy()
tr["genus"] = norm(tr["release_name"]).str.split().str[0]


def plant_group(g):
    if g in SEAGRASS:
        return "seagrass (true marine)"
    if g in MANGROVE:
        return "mangrove (intertidal)"
    if g in HALOPHYTE:
        return "coastal halophyte"
    return "other (likely error)"


tr["plant_group"] = tr["genus"].apply(plant_group)

print(f"Tracheophyta: {tr['molecule_id'].nunique():,} compounds, "
      f"{tr['organism_id'].nunique():,} records\n")
pg = (tr.groupby("plant_group")
      .agg(n_compounds=("molecule_id", "nunique"),
           n_records=("organism_id", "nunique")))
pg["pct_of_release"] = (pg["n_compounds"] / n * 100).round(2)
print(pg.sort_values("n_compounds", ascending=False).to_string())

other = tr[tr["plant_group"] == "other (likely error)"]
if len(other):
    print("\nTop 20 'other' plants, the ones to scrutinise:")
    t = (other.groupby("release_name")
         .agg(n_compounds=("molecule_id", "nunique"),
              isMarine=("isMarine", "first"),
              isTerrestrial=("isTerrestrial", "first"),
              isFreshwater=("isFreshwater", "first"))
         .sort_values("n_compounds", ascending=False).head(20))
    print(t.to_string())
    print("\nAmong these, records WoRMS does not call marine at all:")
    nm = other[other["isMarine"].fillna(0) == 0]
    print(f"  {nm['molecule_id'].nunique():,} compounds from "
          f"{nm['organism_id'].nunique():,} records")
    if len(nm):
        print("  " + ", ".join(sorted(set(nm["release_name"].dropna()))[:15]))

# ------------------------------------------------------------------ 3
section("3. Retuning tier B")

CORE = {"Porifera", "Cnidaria", "Mollusca", "Echinodermata", "Bryozoa", "Annelida",
        "Chordata", "Arthropoda", "Rhodophyta", "Ochrophyta", "Chlorophyta",
        "Heterokontophyta", "Myzozoa", "Foraminifera", "Ciliophora",
        "Platyhelminthes", "Nemertea", "Brachiopoda", "Chaetognatha",
        "Hemichordata", "Cryptophyta", "Haptophyta", "Charophyta"}
MICROBE = {"Actinobacteria", "Proteobacteria", "Cyanobacteria", "Ascomycota",
           "Basidiomycota", "Firmicutes", "Bacteroidetes", "Bacillariophyta",
           "Verrucomicrobia", "Planctomycetota", "Chloroflexi", "Chlorobi",
           "Thermotogae", "Aquificae", "Euryarchaeota", "Crenarchaeota"}

f = {c: link[c].fillna(0) for c in
     ["isMarine", "isBrackish", "isFreshwater", "isTerrestrial"]}
link["strict_marine"] = ((f["isMarine"] == 1) & (f["isBrackish"] == 0)
                         & (f["isFreshwater"] == 0) & (f["isTerrestrial"] == 0))
link["marine_not_terr"] = (f["isMarine"] == 1) & (f["isTerrestrial"] == 0)
link["any_marine"] = (f["isMarine"] == 1) | (f["isBrackish"] == 1)
rk = link["rank"].fillna("").str.strip().str.lower()
link["species_rank"] = rk.str.replace(r"\s*\(?fuzzy\)?", "", regex=True).str.strip() \
    .isin(["species", "subspecies", "forma", "variety"])
link["good_name"] = ~link["organism_id"].isin(bad_ids)

tr_ok_ids = set(tr.loc[tr["plant_group"] != "other (likely error)", "organism_id"])
link["marine_plant"] = link["organism_id"].isin(tr_ok_ids)
link["is_core"] = link["phylum"].isin(CORE) | link["marine_plant"]
link["is_micro"] = link["phylum"].isin(MICROBE)


def mols(mask):
    return set(link.loc[mask, "molecule_id"])


base_a = link["is_core"] & link["strict_marine"] & link["species_rank"]
a_plus = base_a & link["good_name"]

cands = {
    "A  as in 03 (core + strict + species)": base_a,
    "A' A, excluding wrong-genus matches": a_plus,
    "B1 A' + microbes, any marine flag": a_plus | (link["is_micro"] & link["any_marine"]),
    "B2 A' + microbes, marine & not terrestrial": a_plus | (link["is_micro"] & link["marine_not_terr"]),
    "B3 A' + microbes, strictly marine": a_plus | (link["is_micro"] & link["strict_marine"]),
    "B4 A' + microbes, strict + species rank": a_plus | (link["is_micro"] & link["strict_marine"] & link["species_rank"]),
}

print(f"{'definition':46s} {'compounds':>10s} {'% of C':>8s} {'vs A\'':>8s}")
ref = len(mols(a_plus))
for name, m in cands.items():
    s = mols(m)
    print(f"{name:46s} {len(s):>10,} {len(s) / n * 100:7.1f}% "
          f"{len(s) - ref:+8,}")

print(f"\nFull release (C): {n:,}")

print("\nWhat each B candidate adds over A', by phylum:")
for name in ["B2 A' + microbes, marine & not terrestrial",
             "B3 A' + microbes, strictly marine",
             "B4 A' + microbes, strict + species rank"]:
    added = mols(cands[name]) - mols(a_plus)
    sub = link[link["molecule_id"].isin(added) & link["is_micro"]]
    top = sub.groupby("phylum")["molecule_id"].nunique().sort_values(ascending=False)
    print(f"\n  {name}  (+{len(added):,})")
    print("    " + ", ".join(f"{k} {v:,}" for k, v in top.head(8).items()))

print("\nGenus-rank microbial records, the provenance-free ones:")
gm = link[link["is_micro"] & ~link["species_rank"]]
print(f"  {gm['molecule_id'].nunique():,} compounds "
      f"({gm['molecule_id'].nunique() / n * 100:.1f}%) come from a microbial record "
      "identified only to genus")
print("  top:", ", ".join(
    f"{k} ({v:,})" for k, v in
    gm.groupby("release_name")["molecule_id"].nunique()
    .sort_values(ascending=False).head(6).items()))
