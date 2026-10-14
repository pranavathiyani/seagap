"""Interrogate the marine claim and the metadata quality before building anything.

    python scripts/01b_inspect.py

Reads data/interim/compounds.parquet plus the raw organism and location tables.
Writes nothing. The question it answers: how much of this release is actually
marine, and are the taxonomy and location fields usable as analysis axes.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seagap import config  # noqa: E402
from seagap.tables import load_table  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)

print("SCRIPT VERSION: 2026-10-13 inspect\n")

df = load_table(config.INTERIM / "compounds")
org = pd.read_csv(config.RELEASE / config.TABLES["organisms"], low_memory=False)
mol_org = pd.read_csv(config.RELEASE / config.TABLES["molecule_organism"], low_memory=False)
geo = pd.read_csv(config.RELEASE / config.TABLES["geo_locations"], low_memory=False)
geo_mol = pd.read_csv(config.RELEASE / config.TABLES["geo_location_molecule"],
                      low_memory=False)

bar = "=" * 72


def section(t):
    print(f"\n{bar}\n{t}\n{bar}")


# ------------------------------------------------------------------ 1. WoRMS
section("1. WoRMS environment, the independent check on 'marine'")

env = org["environment_aphia_worms"].fillna("(missing)")
vc = env.value_counts()
print(f"{len(org):,} organisms\n")
print(vc.head(15).to_string())
print(f"\ndistinct values: {env.nunique()}")

# Weight by compounds, not by organism: one heavily studied organism matters more.
link = mol_org[["molecule_id", "organism_id"]].dropna().drop_duplicates()
link = link[link["molecule_id"].isin(df["molecule_id"])]
link = link.merge(org[["id", "name", "environment_aphia_worms", "rank",
                       "aphiaid_worms", "molecule_count"]],
                  left_on="organism_id", right_on="id", how="left")

print("\nSame, weighted by compound links:")
print(link["environment_aphia_worms"].fillna("(missing)").value_counts().head(15).to_string())

# A molecule counts as WoRMS-marine if any linked organism says so.
marine_words = ["marine", "brackish"]
is_marine_env = link["environment_aphia_worms"].fillna("").str.lower().apply(
    lambda s: any(w in s for w in marine_words))
mols_marine = set(link.loc[is_marine_env, "molecule_id"])
n = len(df)
print(f"\nMolecules with >=1 WoRMS marine/brackish organism: "
      f"{len(mols_marine):,} / {n:,}  ({len(mols_marine) / n * 100:.1f}%)")

# ------------------------------------------------------------------ 2. taxa
section("2. Which organisms carry the collection")

by_org = link.groupby("name")["molecule_id"].nunique().sort_values(ascending=False)
print("Top 30 source organisms by distinct compounds:\n")
top = by_org.head(30).to_frame("n_compounds")
top["worms_env"] = [
    org.loc[org["name"] == i, "environment_aphia_worms"].iloc[0]
    if (org["name"] == i).any() else None for i in top.index]
top["rank"] = [
    org.loc[org["name"] == i, "rank"].iloc[0]
    if (org["name"] == i).any() else None for i in top.index]
print(top.to_string())

print("\nName specificity:")
nm = org["name"].fillna("")
pat = {
    "ends in ' sp.'": nm.str.strip().str.endswith(" sp."),
    "ends in ' spp.'": nm.str.strip().str.endswith(" spp."),
    "single word (genus or higher)": nm.str.strip().str.split().str.len() == 1,
    "binomial (2 words)": nm.str.strip().str.split().str.len() == 2,
}
for k, m in pat.items():
    print(f"  {k:32s} {int(m.sum()):>7,}  ({m.mean() * 100:5.1f}%)")

print("\nrank field:")
print(org["rank"].fillna("(missing)").value_counts().head(12).to_string())

# ------------------------------------------------------------------ 3. genera
section("3. Genera that look terrestrial")

SUSPECT = ["Streptomyces", "Penicillium", "Aspergillus", "Saccharomyces",
           "Bacillus", "Fusarium", "Trichoderma", "Nocardia"]
genus = link["name"].fillna("").str.strip().str.split().str[0]
link = link.assign(genus=genus)

rows = []
for g in SUSPECT:
    sel = link[link["genus"] == g]
    if sel.empty:
        continue
    rows.append({
        "genus": g,
        "n_compounds": sel["molecule_id"].nunique(),
        "pct_of_total": round(sel["molecule_id"].nunique() / n * 100, 2),
        "worms_env": sel["environment_aphia_worms"].fillna("(missing)").mode().iloc[0]
        if len(sel) else None,
    })
if rows:
    sus = pd.DataFrame(rows).sort_values("n_compounds", ascending=False)
    print(sus.to_string(index=False))
    combined = link[link["genus"].isin(SUSPECT)]["molecule_id"].nunique()
    print(f"\nCombined: {combined:,} compounds ({combined / n * 100:.1f}%)")
else:
    print("none of the suspect genera are present")

print("\nTop 25 genera overall:")
gcount = link.groupby("genus")["molecule_id"].nunique().sort_values(ascending=False).head(25)
print(gcount.to_frame("n_compounds").to_string())

# ------------------------------------------------------------------ 4. places
section("4. Location strings")

gl = geo_mol[["geo_location_id", "molecule_id"]].dropna().drop_duplicates()
gl = gl[gl["molecule_id"].isin(df["molecule_id"])]
gl = gl.merge(geo[["id", "name"]], left_on="geo_location_id", right_on="id", how="left")

names = geo["name"].fillna("")
print(f"{len(geo):,} location rows, {names.nunique():,} distinct names\n")
has_comma = names.str.contains(",")
print(f"contain a comma: {int(has_comma.sum()):,} ({has_comma.mean() * 100:.1f}%)")
if has_comma.any():
    print("  examples:", list(names[has_comma].head(5)))

tok = names.str.split(",").str.len()
print(f"\ncomma-separated field count: max={int(tok.max())}, "
      f"mean={tok.mean():.2f}")

print("\nTop 30 locations by distinct compounds:")
print(gl.groupby("name")["molecule_id"].nunique().sort_values(
    ascending=False).head(30).to_frame("n_compounds").to_string())

print("\nLength of location strings (chars):")
print(names.str.len().describe().to_string())

# ------------------------------------------------------------------ 5. chem
section("5. Chemical classification, the validation labels for step 03")

for col in ["np_classifier_pathway", "np_classifier_superclass", "chemical_super_class"]:
    if col not in df:
        continue
    vc = df[col].fillna("(missing)").value_counts()
    print(f"\n{col}: {df[col].nunique()} distinct, "
          f"{df[col].notna().mean() * 100:.1f}% present")
    print(vc.head(12).to_string())

section("Summary")
print(f"compounds                       {n:,}")
print(f"with WoRMS marine/brackish org  {len(mols_marine):,} "
      f"({len(mols_marine) / n * 100:.1f}%)")
print(f"with any location               {int((df['n_locations'] > 0).sum()):,} "
      f"({(df['n_locations'] > 0).mean() * 100:.1f}%)")
print(f"with NPClassifier pathway       {int(df['np_classifier_pathway'].notna().sum()):,} "
      f"({df['np_classifier_pathway'].notna().mean() * 100:.1f}%)")
