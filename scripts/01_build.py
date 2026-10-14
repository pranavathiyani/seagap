"""Join the MarineNP release into one compound table, and audit it.

    python scripts/01_build.py

Reads marinenp_v1_r20250608_csv/, writes data/interim/compounds.parquet and
data/interim/audit.json. Prints everything it decided so the numbers land in your
notes rather than only in a file.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seagap import config  # noqa: E402
from seagap.tables import save_table  # noqa: E402

pd.set_option("display.width", 160)
audit = {}

TRUE = {"t", "true", "1", "y", "yes"}
FALSE = {"f", "false", "0", "n", "no", "", "nan", "none", "null"}


def as_bool(series, name=""):
    """Coerce a Postgres-style flag column to real booleans.

    The release writes booleans as 't'/'f' strings. Passing those to astype(bool)
    yields True for every row, since 'f' is a non-empty string. Anything not
    recognised is reported and treated as False.
    """
    if series.dtype == bool:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)

    s = series.astype("string").str.strip().str.lower()
    out = s.isin(TRUE)
    unknown = ~(s.isin(TRUE) | s.isin(FALSE) | s.isna())
    if unknown.any():
        vals = sorted(set(s[unknown].dropna()))[:5]
        print(f"  warning: {name or series.name}: {int(unknown.sum()):,} "
              f"unrecognised values treated as False, e.g. {vals}")
    return out


def show_flag(df, col, label):
    raw = df[col].astype("string").str.strip().str.lower().value_counts(dropna=False)
    print(f"  {label}.{col} raw values: {dict(raw.head(6))}")


def load(key, **kw):
    path = config.RELEASE / config.TABLES[key]
    if not path.exists():
        sys.exit(f"missing: {path}\nUnzip the release into {config.RELEASE}")
    df = pd.read_csv(path, low_memory=False, **kw)
    print(f"  {config.TABLES[key]:28s} {len(df):>9,} rows  {df.shape[1]:>3} cols")
    return df


print("Loading")
mol = load("molecules")
prop = load("properties")
org = load("organisms")
mol_org = load("molecule_organism")
geo = load("geo_locations")
geo_mol = load("geo_location_molecule")

audit["n_molecules_raw"] = int(len(mol))
audit["n_organisms_raw"] = int(len(org))

# ---------------------------------------------------------------- marine filter
print("\nMarine flags")
show_flag(mol, "is_marine", "molecules")
show_flag(org, "is_marine", "organisms")
if "is_duplicate" in mol:
    show_flag(mol, "is_duplicate", "molecules")
print()

mol_marine = as_bool(mol["is_marine"], "molecules.is_marine")

marine_org_ids = set(org.loc[as_bool(org["is_marine"], "organisms.is_marine"), "id"])
mo = mol_org[["molecule_id", "organism_id"]].dropna()
via_org = set(mo.loc[mo["organism_id"].isin(marine_org_ids), "molecule_id"])
org_marine = mol["id"].isin(via_org)

counts = {
    "molecule": int(mol_marine.sum()),
    "organism": int(org_marine.sum()),
    "both": int((mol_marine & org_marine).sum()),
    "either": int((mol_marine | org_marine).sum()),
}
for k, v in counts.items():
    print(f"  {k:10s} {v:>9,}")
audit["marine_counts"] = counts
audit["marine_rule"] = config.MARINE_RULE

disagree = int((mol_marine ^ org_marine).sum())
print(f"  flags disagree on {disagree:,} molecules "
      f"({disagree / max(len(mol), 1) * 100:.1f}%)")
audit["marine_flag_disagreement"] = disagree

rule = {"molecule": mol_marine, "organism": org_marine,
        "both": mol_marine & org_marine, "either": mol_marine | org_marine}
if config.MARINE_RULE not in rule:
    sys.exit(f"MARINE_RULE must be one of {list(rule)}")
mol = mol[rule[config.MARINE_RULE]].copy()
print(f"\nKept {len(mol):,} molecules under rule '{config.MARINE_RULE}'")
if len(mol) == 0:
    sys.exit("No molecules survived the marine filter. Check the raw flag values "
             "printed above before going further.")

if config.DROP_IS_DUPLICATE and "is_duplicate" in mol:
    dup = as_bool(mol["is_duplicate"], "molecules.is_duplicate")
    n = int(dup.sum())
    mol = mol[~dup]
    print(f"Dropped {n:,} flagged duplicates -> {len(mol):,}")
    audit["n_flagged_duplicates"] = n
    if len(mol) == 0:
        sys.exit("Every molecule was flagged as a duplicate. Set "
                 "DROP_IS_DUPLICATE = False in seagap/config.py and re-run.")

# ---------------------------------------------------------------- core columns
keep = ["id", "identifier", "name", "canonical_smiles", "standard_inchi_key",
        "murko_framework", "organism_count", "geo_count", "citation_count",
        "has_stereo", "is_parent", "annotation_level", "name_trust_level"]
keep = [c for c in keep if c in mol.columns]
df = mol[keep].rename(columns={"id": "molecule_id"})
for flag in ["has_stereo", "is_parent"]:
    if flag in df.columns:
        df[flag] = as_bool(df[flag], flag)

# ---------------------------------------------------------------- properties
pcols = ["molecule_id", "molecular_weight", "alogp", "qed_drug_likeliness",
         "np_likeness", "fractioncsp3", "aromatic_rings_count",
         "number_of_minimal_rings", "murcko_framework",
         "np_classifier_pathway", "np_classifier_superclass", "np_classifier_class",
         "chemical_super_class", "chemical_class", "chemical_sub_class"]
pcols = [c for c in pcols if c in prop.columns]
prop = prop[pcols].drop_duplicates(subset="molecule_id")
df = df.merge(prop, on="molecule_id", how="left")
print(f"\nProperties joined, {df['np_classifier_pathway'].notna().mean() * 100:.1f}% "
      f"have an NPClassifier pathway")

# ---------------------------------------------------------------- organisms
org_small = org[["id", "name", "rank", "aphiaid_worms", "name_aphia_worms",
                 "environment_aphia_worms", "is_marine"]].rename(
    columns={"id": "organism_id", "name": "organism_name"})
mo = mol_org[["molecule_id", "organism_id"]].dropna().drop_duplicates()
mo = mo.merge(org_small, on="organism_id", how="left")
mo = mo[mo["molecule_id"].isin(df["molecule_id"])]

org_agg = mo.groupby("molecule_id").agg(
    organisms=("organism_name", lambda s: sorted(set(s.dropna()))),
    organism_ids=("organism_id", list),
    n_organisms=("organism_id", "nunique"),
    aphia_ids=("aphiaid_worms", lambda s: sorted(set(s.dropna().astype("Int64")))),
)
df = df.merge(org_agg, on="molecule_id", how="left")
df["n_organisms"] = df["n_organisms"].fillna(0).astype(int)

# ---------------------------------------------------------------- geography
geo_small = geo[["id", "name"]].rename(columns={"id": "geo_location_id",
                                                "name": "location_name"})
gm = geo_mol[["geo_location_id", "molecule_id"]].dropna().drop_duplicates()
gm = gm.merge(geo_small, on="geo_location_id", how="left")
gm = gm[gm["molecule_id"].isin(df["molecule_id"])]

geo_agg = gm.groupby("molecule_id").agg(
    locations=("location_name", lambda s: sorted(set(s.dropna()))),
    n_locations=("geo_location_id", "nunique"),
)
df = df.merge(geo_agg, on="molecule_id", how="left")
df["n_locations"] = df["n_locations"].fillna(0).astype(int)

# ---------------------------------------------------------------- coverage
print("\nCoverage")


def pct(mask):
    return round(float(np.mean(mask) * 100), 2)


cov = {
    "smiles": pct(df["canonical_smiles"].notna()),
    "inchikey": pct(df["standard_inchi_key"].notna()) if "standard_inchi_key" in df else 0,
    "any_organism": pct(df["n_organisms"] > 0),
    "any_location": pct(df["n_locations"] > 0),
    "np_pathway": pct(df["np_classifier_pathway"].notna()) if "np_classifier_pathway" in df else 0,
    "chemical_class": pct(df["chemical_class"].notna()) if "chemical_class" in df else 0,
    "worms_aphia": pct(df["aphia_ids"].apply(lambda v: bool(v) if isinstance(v, list) else False)),
}
for k, v in cov.items():
    flag = "  <-- below 50%" if v < 50 else ""
    print(f"  {k:16s} {v:6.2f}%{flag}")
audit["coverage_pct"] = cov

if cov["any_location"] < 50:
    print("\n  Location coverage is below the 50% kill criterion for the")
    print("  geography half of SEAGAP. Decide before building step 04.")

# ---------------------------------------------------------------- concentration
def gini(counts):
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


print("\nSampling concentration")
conc = {}
for label, series in [("organism", mo["organism_name"]), ("location", gm["location_name"])]:
    c = series.dropna().value_counts()
    if len(c) == 0:
        continue
    conc[label] = {
        "n_unique": int(len(c)),
        "gini": round(gini(c.values), 3),
        "top10_share_pct": round(float(c.head(10).sum() / c.sum() * 100), 1),
    }
    print(f"  {label:10s} n={len(c):>6,}  gini={conc[label]['gini']:.3f}  "
          f"top10={conc[label]['top10_share_pct']:.1f}%")
    print(f"             top 5: {', '.join(c.head(5).index.astype(str))}")
audit["concentration"] = conc

# ---------------------------------------------------------------- structures
before = len(df)
df = df[df["canonical_smiles"].notna() & (df["canonical_smiles"].str.len() > 0)]
print(f"\n{before:,} -> {len(df):,} with a SMILES")

if "standard_inchi_key" in df:
    dup = df["standard_inchi_key"].duplicated(keep=False) & df["standard_inchi_key"].notna()
    print(f"{int(dup.sum()):,} rows share an InChIKey with another row "
          "(collapsed in step 02)")
    audit["n_sharing_inchikey"] = int(dup.sum())

audit["n_final"] = int(len(df))

config.INTERIM.mkdir(parents=True, exist_ok=True)
out = save_table(df.reset_index(drop=True), config.INTERIM / "compounds")
(config.INTERIM / "audit.json").write_text(json.dumps(audit, indent=2))

print(f"\nWrote {out}  ({len(df):,} rows, {df.shape[1]} cols)")
print(f"Wrote {config.INTERIM / 'audit.json'}")
