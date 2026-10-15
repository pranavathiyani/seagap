"""Is marine chemistry actually different?

    python scripts/14_baseline.py

Step 13 described the library. Describing it is not a claim. "Marine natural
products are sp3-rich" means nothing until something else is shown to be less
so, and a reviewer will say exactly that.

MarineNP derives from COCONUT, which makes the non-marine COCONUT remainder the
right control and an unusually clean one: same parent collection, same curation
pipeline, same structure standardisation, same classifier. The only variable
left is habitat. A comparison against a synthetic screening library would
confound habitat with curation, vendor bias and purpose.

Two cautions, both reported in the output rather than buried here.

The COCONUT release available now postdates the MarineNP snapshot. Compounds
added to COCONUT since then are in the baseline and were never eligible for
MarineNP, and some of those are marine. That contaminates the control in the
direction of making marine and non-marine look MORE alike, so it works against
every difference this script reports rather than manufacturing one.

With tens of thousands of compounds per side, every p value is zero and none of
them mean anything. Effect sizes are what is reported. Cliff's delta is used
because it is rank-based and these distributions are skewed and heavy-tailed.

Writes baseline_comparison.csv, baseline_scaffolds.csv and baseline.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seagap import config  # noqa: E402
from seagap.tables import load_table, save_table  # noqa: E402

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("rdkit is required: pip install rdkit")

from scipy.stats import mannwhitneyu  # noqa: E402

from seagap.descriptors import (  # noqa: E402
    NUMERIC, STEREO_ERRORS, inchikey_block, load_contrib, murcko, profile)

pd.set_option("display.width", 240)
print("SCRIPT VERSION: 2026-10-15 baseline\n")
bar = "=" * 98

DOWNLOAD = """
The COCONUT release is not here yet.

  1. Open https://coconut.naturalproducts.net/download
  2. Download coconut_csv-*.zip (the full CSV, roughly 200 MB zipped)
  3. Unzip it into {d}

COCONUT is CC0, so no licence paperwork. Record the release date you used in
data/raw/SOURCE.txt next to the MarineNP version, because the two snapshots do
not coincide and the gap belongs in the methods.
"""


def section(t):
    print(f"\n{bar}\n{t}\n{bar}")


def cliffs_delta(a, b):
    """Rank-based effect size in [-1, 1]. Positive means a tends to exceed b.

    Derived from the Mann-Whitney U rather than the naive pairwise count, which
    would be 29,154 x 50,000 comparisons. Magnitude labels follow the
    conventional cuts: negligible, small, medium, large.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, "n/a"
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    d = 2.0 * u / (len(a) * len(b)) - 1.0
    m = abs(d)
    label = ("negligible" if m < 0.147 else
             "small" if m < 0.330 else
             "medium" if m < 0.474 else "large")
    return float(d), label


# ------------------------------------------------------------------ find input
if not config.COCONUT.is_dir():
    sys.exit(DOWNLOAD.format(d=config.COCONUT))

files = sorted([p for p in config.COCONUT.rglob("*")
                if p.suffix.lower() in (".csv", ".tsv")
                or p.name.lower().endswith((".csv.gz", ".tsv.gz"))])
if not files:
    sys.exit(DOWNLOAD.format(d=config.COCONUT))

src = max(files, key=lambda p: p.stat().st_size)
print(f"baseline source: {src.name} ({src.stat().st_size / 1e6:,.0f} MB)")
sep = "\t" if ".tsv" in src.name.lower() else ","
base = pd.read_csv(src, sep=sep, low_memory=False)
print(f"{len(base):,} rows, {len(base.columns)} columns")
print(f"columns: {list(base.columns)}\n")


def pick(cands, keywords, require=True, what=""):
    """First column whose name contains any keyword, preferring exact order."""
    low = {c.lower(): c for c in cands}
    for k in keywords:
        if k in low:
            return low[k]
    for k in keywords:
        for c in cands:
            if k in c.lower():
                return c
    if require:
        sys.exit(f"could not find a {what} column among: {list(cands)}")
    return None


smi_col = pick(base.columns,
               ["canonical_smiles", "smiles", "absolute_smiles", "structure"],
               what="SMILES")
key_col = pick(base.columns,
               ["standard_inchi_key", "inchikey", "inchi_key"],
               require=False)
print(f"using SMILES column: {smi_col}")
print(f"using InChIKey column: {key_col or 'none, will compute'}")


# -------------------------------------------------- remove the marine compounds
section("1. Building the non-marine remainder")
prop = load_table(config.INTERIM / "properties")
print(f"{len(prop):,} marine compounds profiled in step 13")

mol = pd.read_csv(config.RELEASE / config.TABLES["molecules"], low_memory=False)
mkey = pick(mol.columns, ["standard_inchi_key", "inchikey", "inchi_key"],
            require=False)

# Match on the InChIKey connectivity block, not the full key. A compound
# recorded with different stereochemistry, tautomer or salt form in the two
# collections is the same compound for this purpose, and leaving it in the
# baseline would understate every difference.
if mkey:
    marine_keys = set(mol[mkey].dropna().astype(str).str.split("-").str[0])
    print(f"{len(marine_keys):,} marine connectivity keys read from "
          f"molecules.{mkey}")
    print("Exclusion uses the WHOLE release, not just the profiled subset. A "
          "compound\nMarineNP holds but this analysis dropped is still a "
          "marine compound and has\nno business in the control.")
else:
    print("no InChIKey in the release, computing from SMILES (slower)")
    marine_keys = {k for k in
                   (inchikey_block(s) for s in prop["canonical_smiles"])
                   if k}

if key_col:
    base_keys = base[key_col].astype(str).str.split("-").str[0]
else:
    print("no InChIKey in the baseline, computing from SMILES (slow)")
    base_keys = pd.Series([inchikey_block(s) for s in base[smi_col]],
                          index=base.index)

overlap = base_keys.isin(marine_keys)
print(f"{overlap.sum():,} baseline rows match a marine compound "
      f"({overlap.mean() * 100:.1f}%), removed")
rem = base[~overlap].copy()

# COCONUT may carry its own marine or habitat annotation. If it does, use it:
# compounds added since the MarineNP snapshot are the contamination this
# script's docstring warns about, and this removes the ones that are labelled.
hab = [c for c in rem.columns
       if any(k in c.lower() for k in ("marine", "habitat"))]
for c in hab:
    s = rem[c]
    flag = (s.astype(str).str.lower().isin(["t", "true", "1", "yes"])
            if s.dtype != bool else s)
    if flag.sum():
        print(f"baseline column '{c}' flags {flag.sum():,} rows, removed")
        rem = rem[~flag]
if not hab:
    print("no marine or habitat column in the baseline, so compounds added to "
          "COCONUT\nsince the MarineNP snapshot cannot be screened out. That "
          "dilutes the contrast\nrather than inventing one.")

print(f"{len(rem):,} compounds in the non-marine remainder")

n = min(config.BASELINE_SAMPLE, len(rem))
samp = rem.sample(n=n, random_state=config.BASELINE_SEED)
print(f"profiling a random sample of {n:,} (seed {config.BASELINE_SEED})")


# ------------------------------------------------------------------- profile
sa_fn, np_fn = load_contrib()
id_col = pick(samp.columns, ["identifier", "coconut_id", "id"], require=False)
rows, failed = profile(samp[smi_col].tolist(), sa_fn, np_fn,
                       ids=(samp[id_col].tolist() if id_col else None),
                       label=f"/ {n:,}  at ")
bp = pd.DataFrame(rows)
# Keep the SMILES so later steps can work from this table instead of
# re-profiling 50,000 compounds for every question asked of the baseline.
# Joined on the identifier, never positionally: `rows` omits any compound that
# failed to parse, so row i of bp is not row i of samp.
if id_col:
    bp = bp.merge(samp[[id_col, smi_col]].rename(columns={id_col: "id"}),
                  on="id", how="left").rename(columns={"id": id_col})
else:
    bp = bp.merge(samp[[smi_col]].reset_index(drop=True)
                  .rename_axis("id").reset_index(), on="id", how="left")
print(f"{len(bp):,} baseline compounds profiled, {len(failed):,} unparseable")
if STEREO_ERRORS:
    print(f"WARNING: stereo perception failed on {len(STEREO_ERRORS):,}; "
          f"first error: {STEREO_ERRORS[0]}")


# ------------------------------------------------------------- 2. the contrast
section("2. Marine against the non-marine remainder")
have = [c for c in NUMERIC
        if c in prop.columns and c in bp.columns
        and prop[c].notna().any() and bp[c].notna().any()]

recs = []
for c in have:
    d, label = cliffs_delta(prop[c], bp[c])
    recs.append({
        "descriptor": c,
        "marine_median": round(float(prop[c].median()), 3),
        "baseline_median": round(float(bp[c].median()), 3),
        "difference": round(float(prop[c].median() - bp[c].median()), 3),
        "cliffs_delta": round(d, 3),
        "magnitude": label,
    })
cmp = pd.DataFrame(recs).sort_values(
    "cliffs_delta", key=lambda s: s.abs(), ascending=False)
print(cmp.to_string(index=False))
print("\nCliff's delta is the share of cross-library pairs in which marine "
      "exceeds\nbaseline, rescaled to [-1, 1]. Positive means marine runs "
      "higher. Read the\nmagnitude column, not a p value: at this sample size "
      "every p is zero.")


# ---------------------------------------------------------- 3. the categoricals
section("3. Proportions")


def prop_row(name, m_mask, b_mask):
    pm, pb = float(m_mask.mean() * 100), float(b_mask.mean() * 100)
    return {"trait": name, "marine_pct": round(pm, 1),
            "baseline_pct": round(pb, 1), "ratio": round(pm / pb, 2) if pb else np.nan}


cats = pd.DataFrame([
    prop_row("beyond Ro5 (2+ violations)",
             prop["ro5_violations"] >= 2, bp["ro5_violations"] >= 2),
    prop_row("macrocyclic", prop["macrocyclic"], bp["macrocyclic"]),
    prop_row("fsp3 above 0.5", prop["fsp3"] > 0.5, bp["fsp3"] > 0.5),
    prop_row("Veber compatible", prop["veber_pass"], bp["veber_pass"]),
    prop_row("has a stereocentre",
             prop["n_stereocentres"] > 0, bp["n_stereocentres"] > 0),
    prop_row("has an aromatic ring",
             prop["n_aromatic_rings"] > 0, bp["n_aromatic_rings"] > 0),
])
print(cats.to_string(index=False))


# ------------------------------------------------------------- 4. the scaffolds
section("4. Scaffold overlap")
print("Bemis-Murcko skeletons, stereochemistry stripped, so two libraries "
      "match on\nframework rather than configuration. This is the structural "
      "sense of a gap:\nskeletons the marine set has and the rest of the "
      "natural-product world does not.")

m_scaf = pd.Series([murcko(s) for s in prop["canonical_smiles"]]).dropna()
b_scaf = pd.Series([murcko(s) for s in samp[smi_col]]).dropna()
M, B = set(m_scaf), set(b_scaf)
only_m, only_b, both = M - B, B - M, M & B
pct_only = len(only_m) / len(M) * 100 if M else float("nan")
jacc = len(both) / len(M | B) if (M or B) else float("nan")
print(f"\nmarine:   {len(m_scaf):,} compounds, {len(M):,} distinct scaffolds")
print(f"baseline: {len(b_scaf):,} compounds, {len(B):,} distinct scaffolds")
print(f"shared:          {len(both):,}")
print(f"marine only:     {len(only_m):,} ({pct_only:.1f}% of marine scaffolds)")
print(f"baseline only:   {len(only_b):,}")
print(f"Jaccard:         {jacc:.3f}")

vc = m_scaf[m_scaf.isin(only_m)].value_counts()
print(f"\nmost populated marine-only scaffolds (top 15 of {len(vc):,}):")
sc = vc.head(15).rename("n_compounds").to_frame()
print(sc.to_string())
print("\nA scaffold seen once is a curiosity. One carrying many compounds is a "
      "marine\nskeleton that the rest of the collection does not reach, which "
      "is the claim\nthe project title has been making without evidence until "
      "now.")

print("\nThe baseline is a sample, so 'marine only' means absent from this "
      f"sample of\n{n:,}, not absent from COCONUT. A scaffold with few "
      "compounds on either side is\nthe case most likely to flip, which is "
      "why the table is ranked by count.")


# ----------------------------------------------------------------------- save
save_table(bp, config.INTERIM / "baseline_properties")
cmp.to_csv(config.INTERIM / "baseline_comparison.csv", index=False)
cats.to_csv(config.INTERIM / "baseline_proportions.csv", index=False)
sc.reset_index().rename(columns={"index": "scaffold"}).to_csv(
    config.INTERIM / "baseline_scaffolds.csv", index=False)
(config.INTERIM / "baseline.json").write_text(json.dumps({
    "baseline_file": src.name,
    "baseline_rows_total": int(len(base)),
    "baseline_overlap_removed": int(overlap.sum()),
    "baseline_remainder": int(len(rem)),
    "baseline_sampled": int(n),
    "baseline_profiled": int(len(bp)),
    "marine_profiled": int(len(prop)),
    "effects": {r["descriptor"]: r["cliffs_delta"] for r in recs},
    "magnitudes": {r["descriptor"]: r["magnitude"] for r in recs},
    "scaffolds_marine": len(M),
    "scaffolds_baseline": len(B),
    "scaffolds_marine_only": len(only_m),
    "scaffolds_shared": len(both),
    "scaffold_jaccard": None if pd.isna(jacc) else round(jacc, 4),
}, indent=2, default=float))
print(f"\nWrote baseline_properties, baseline_comparison.csv, baseline_proportions.csv,\n"
      f"baseline_scaffolds.csv, baseline.json\nto {config.INTERIM}")
