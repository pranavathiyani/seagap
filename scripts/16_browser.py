"""Compound payload for the browsable tab.

    python scripts/16_browser.py

Writes docs/compounds.js, a separate file loaded with a script tag rather than
inlined into index.html. Three reasons for that shape.

Inlining 29,154 compounds would make every tab pay the parse cost, including
Overview, which needs none of it. A script tag defers nothing but keeps
index.html small enough to read and diff.

Fetch would be the obvious alternative and is wrong here: fetch against a
file:// URL is blocked by CORS, so the page would stop working when opened from
disk. A script tag has no such restriction, and opening the viewer from disk
without a server is a property worth keeping.

The payload is columnar and integer-coded. Array-of-objects would repeat every
key 29,154 times, and floats are scaled to integers at display precision, which
roughly halves the rest.

Run after 13 and before or after 11; they do not interact.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seagap import config  # noqa: E402
from seagap.tables import load_table  # noqa: E402

OUT = config.ROOT / "docs" / "compounds.js"
print("SCRIPT VERSION: 2026-10-15 browser\n")

# (column, label, scale). The scale is what the page divides by, so it sets the
# displayed precision: mw at 10 shows one decimal, fsp3 at 1000 shows three.
NUMERIC = [
    ("mw",                 "MW",        10),
    ("clogp",              "cLogP",    100),
    ("tpsa",               "TPSA",      10),
    ("hbd",                "HBD",        1),
    ("hba",                "HBA",        1),
    ("rotb",               "RotB",       1),
    ("n_rings",            "Rings",      1),
    ("n_aromatic_rings",   "ArRings",    1),
    ("fsp3",               "Fsp3",    1000),
    ("n_stereocentres",    "Stereo",     1),
    ("n_stereo_unspecified", "Unspec",   1),
    ("qed",                "QED",     1000),
    ("sa_score",           "SA",       100),
    ("np_likeness",        "NPL",      100),
    ("ro5_violations",     "Ro5",        1),
]
CATEGORICAL = [("phylum", "Phylum"), ("genus", "Genus"),
               ("np_classifier_pathway", "Pathway"),
               ("np_classifier_superclass", "Superclass")]


def codes(series):
    """(list of distinct values, int array indexing into it). -1 is missing."""
    vals = sorted({v for v in series.dropna().astype(str)})
    idx = {v: i for i, v in enumerate(vals)}
    return vals, [idx.get(v, -1) if isinstance(v, str) else -1
                  for v in series.astype(object).where(series.notna(), None)
                  .map(lambda x: str(x) if x is not None else None)]


prop = load_table(config.INTERIM / "properties")
print(f"{len(prop):,} compounds with a descriptor panel")

# One row per compound. A compound reported from several organisms gets its
# most frequent phylum and genus, chosen before the join so the table stays
# one row per compound rather than silently multiplying.
tax = load_table(config.INTERIM / "organism_taxonomy")
org = pd.read_csv(config.RELEASE / config.TABLES["organisms"], low_memory=False)
mo = pd.read_csv(config.RELEASE / config.TABLES["molecule_organism"],
                 low_memory=False)
link = (mo[["molecule_id", "organism_id"]].dropna().drop_duplicates()
        .merge(org[["id", "aphiaid_worms"]].rename(columns={"id": "organism_id"}),
               on="organism_id", how="left")
        .merge(tax[["aphiaid_worms", "phylum", "genus"]], on="aphiaid_worms",
               how="left"))
agg = (link.dropna(subset=["phylum"]).groupby("molecule_id")
       .agg(phylum=("phylum", lambda s: s.mode().iloc[0]),
            genus=("genus", lambda s: s.mode().iloc[0] if s.notna().any() else None),
            n_organisms=("organism_id", "nunique")))
df = prop.merge(agg, on="molecule_id", how="left")
print(f"{df['phylum'].notna().sum():,} have a resolved phylum")

mol = pd.read_csv(config.RELEASE / config.TABLES["molecules"], low_memory=False)
namecol = next((c for c in ("name", "iupac_name", "identifier")
                if c in mol.columns), None)
if namecol:
    df = df.merge(mol[["id", namecol]].rename(
        columns={"id": "molecule_id", namecol: "cname"}),
        on="molecule_id", how="left")
else:
    df["cname"] = None
print(f"name column: {namecol or 'none'}")

# ------------------------------------------------------------------- encode
dicts, cat_cols = {}, []
for col, label in CATEGORICAL:
    if col not in df.columns:
        print(f"  note: no {col} column, skipped")
        continue
    vals, idx = codes(df[col])
    dicts[label] = vals
    cat_cols.append((label, idx))
    print(f"  {label}: {len(vals):,} distinct")

num_cols = []
for col, label, scale in NUMERIC:
    if col not in df.columns:
        print(f"  note: no {col} column, skipped")
        continue
    v = pd.to_numeric(df[col], errors="coerce") * scale
    # NaN has no integer representation, so missing is a sentinel the page
    # checks for. -2147483648 cannot collide with a real scaled descriptor.
    num_cols.append((label, scale,
                     v.round().fillna(-2147483648).astype("int64").tolist()))

rows = []
n = len(df)
mid = df["molecule_id"].astype("int64").tolist()
fam = pd.to_numeric(df.get("family"), errors="coerce").fillna(-1) \
        .astype("int64").tolist() if "family" in df.columns else [-1] * n
macro = df["macrocyclic"].fillna(False).astype(int).tolist() \
    if "macrocyclic" in df.columns else [0] * n
susp = df["name_suspect"].fillna(False).astype(int).tolist() \
    if "name_suspect" in df.columns else [0] * n

for i in range(n):
    row = [mid[i], fam[i], macro[i], susp[i]]
    row += [c[1][i] for c in cat_cols]
    row += [c[2][i] for c in num_cols]
    rows.append(row)

def clean(s):
    if not isinstance(s, str):
        return ""
    s = s.strip()
    return s if s and s.lower() not in ("nan", "none", "null") else ""


payload = {
    "n": n,
    "head": ["id", "family", "macrocyclic", "suspect"]
            + [c[0] for c in cat_cols] + [c[0] for c in num_cols],
    "nCat": len(cat_cols),
    "scales": [c[1] for c in num_cols],
    "dicts": dicts,
    "rows": rows,
    "names": [clean(s) for s in df.get("cname", pd.Series([""] * n))],
    "smiles": [clean(s) for s in df.get("canonical_smiles",
                                        pd.Series([""] * n))],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
js = "window.COMPOUNDS=" + json.dumps(payload, separators=(",", ":")) + ";\n"
OUT.write_text(js, encoding="utf-8")
mb = OUT.stat().st_size / 1e6
print(f"\nwrote {OUT} ({mb:.1f} MB)")
print(f"  {n:,} rows x {len(payload['head'])} columns")
print(f"  {sum(len(v) for v in dicts.values()):,} dictionary entries")
if mb > 12:
    print("  WARNING: over 12 MB. Browsers will cope but the first load will "
          "drag.\n  Drop the smiles array in this script if it needs to come "
          "down.")
