"""Two checks the baseline result needs before anything is published.

    python scripts/15_axis.py

Step 14 reported five differences between marine and non-marine natural
products and one scaffold statistic. Both need work before they can be claimed.

1. Scaffolds on the full population, not a sample. Step 14 compared 29,154
   marine compounds against a 50,000 sample of a 611,721 remainder, roughly 8%.
   Scaffold discovery is a rarefaction problem: a skeleton present in COCONUT at
   low frequency is missed at that rate, so "76.1% of marine scaffolds are
   absent" is an upper bound, not an estimate. Both releases ship a precomputed
   Murcko column, from the same pipeline, so the comparison can run over every
   row at almost no cost. The columns are validated against RDKit first, because
   an undocumented difference in how they were generated would be silent.

2. Is it one axis or five? NP-likeness, fsp3, aromatic-ring count and
   stereocentre count may be four views of a single contrast between flat
   aromatic chemistry and saturated three-dimensional chemistry. Reporting them
   as four findings would inflate one result into four, and a reviewer will say
   so. Effects are recomputed inside strata of aromatic-ring count: if they
   collapse, aromaticity is the axis and the rest is downstream.

Needs baseline_properties from step 14. Re-run step 14 first if it is missing;
the version that wrote it is dated 2026-10-15.

Writes scaffold_overlap.csv, scaffold_sources.csv, axis_strata.csv, axis.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seagap import config  # noqa: E402
from seagap.tables import load_table, table_path  # noqa: E402

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("rdkit is required: pip install rdkit")

from scipy.stats import mannwhitneyu  # noqa: E402

from seagap.descriptors import murcko  # noqa: E402

pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", 72)
print("SCRIPT VERSION: 2026-10-15 axis\n")
bar = "=" * 98

VALIDATE_N = 3_000      # rows per side to check the precomputed column against
MIN_STRATUM = 300       # below this a stratum's effect size is not worth citing
MATERIAL = 0.05         # an opposing stratum below this is noise crossing zero
rng = np.random.default_rng(0)


def section(t):
    print(f"\n{bar}\n{t}\n{bar}")


def cliffs_delta(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, "n/a"
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    d = 2.0 * u / (len(a) * len(b)) - 1.0
    m = abs(d)
    return float(d), ("negligible" if m < 0.147 else "small" if m < 0.330
                      else "medium" if m < 0.474 else "large")


def pick(cols, keywords, require=True, what=""):
    low = {c.lower(): c for c in cols}
    for k in keywords:
        if k in low:
            return low[k]
    for k in keywords:
        for c in cols:
            if k in c.lower():
                return c
    if require:
        sys.exit(f"could not find a {what} column among: {list(cols)}")
    return None


def recanon(s):
    """Re-canonicalise a SMILES string through RDKit, or None."""
    if not isinstance(s, str) or not s:
        return None
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


def norm_scaffold(s):
    """Blank-safe. Empty string and 'None' both mean acyclic, not a scaffold."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    return s if s and s.lower() not in ("none", "nan", "null") else None


# ===================================================================== inputs
if not config.COCONUT.is_dir():
    sys.exit(f"baseline missing, see step 14. Expected {config.COCONUT}")
files = sorted([p for p in config.COCONUT.rglob("*")
                if p.suffix.lower() in (".csv", ".tsv")])
if not files:
    sys.exit(f"no CSV in {config.COCONUT}, see step 14")
src = max(files, key=lambda p: p.stat().st_size)

prop = load_table(config.INTERIM / "properties")
mol = pd.read_csv(config.RELEASE / config.TABLES["molecules"], low_memory=False)
base = pd.read_csv(src, sep="\t" if ".tsv" in src.name.lower() else ",",
                   low_memory=False)
print(f"marine profiled: {len(prop):,}   release: {len(mol):,}   "
      f"baseline: {len(base):,}")

# Note the spelling difference: the release says 'murko', COCONUT says
# 'murcko'. The keyword search catches both rather than relying on either.
m_scaf_col = pick(mol.columns, ["murko_framework", "murcko_framework",
                                "murko", "murcko"], what="Murcko (release)")
b_scaf_col = pick(base.columns, ["murcko_framework", "murko_framework",
                                 "murcko", "murko"], what="Murcko (baseline)")
m_smi_col = pick(mol.columns, ["canonical_smiles", "smiles"], what="SMILES")
b_smi_col = pick(base.columns, ["canonical_smiles", "smiles"], what="SMILES")
m_key_col = pick(mol.columns, ["standard_inchi_key", "inchikey"], what="key")
b_key_col = pick(base.columns, ["standard_inchi_key", "inchikey"], what="key")
print(f"release Murcko column:  {m_scaf_col}")
print(f"baseline Murcko column: {b_scaf_col}")


# ============================================ 1a. validate the stored columns
section("1a. Do the precomputed Murcko columns mean the same thing?")
print("Both releases ship a scaffold column. Using them without checking "
      "assumes they\nwere generated the same way. Each is compared against "
      "RDKit on a sample.")


def validate(df, scaf_col, smi_col, label):
    sub = df[[scaf_col, smi_col]].dropna()
    if not len(sub):
        print(f"  {label}: column empty")
        return None
    take = sub.sample(n=min(VALIDATE_N, len(sub)), random_state=0)
    ours = [murcko(s) for s in take[smi_col]]
    # Round-trip their string through RDKit before comparing. COCONUT's SMILES
    # are written by a different toolkit, so the same scaffold can be a
    # different string: c1ccc2c3c([nH]c2c1)C1(CCCCC1)NCC3 against
    # c1ccc2c(c1)[nH]c3c2CCNC34CCCCC4 is one molecule written two ways.
    # Comparing raw strings measures canonicalisation, not chemistry.
    theirs = [recanon(norm_scaffold(s)) for s in take[scaf_col]]
    pairs = [(o, t) for o, t in zip(ours, theirs) if o is not None
             and t is not None]
    if not pairs:
        print(f"  {label}: nothing comparable")
        return None
    agree = sum(o == t for o, t in pairs) / len(pairs)
    print(f"  {label}: {agree * 100:.1f}% exact match on {len(pairs):,} "
          "comparable rows")
    if agree < 0.90:
        # Search every pair for a mismatch, not the first three. With 68% of
        # rows disagreeing an earlier version still printed nothing, because
        # the first three it looked at happened to match.
        shown = 0
        for o, t in pairs:
            if o != t:
                print(f"      ours:   {o}")
                print(f"      theirs: {t}")
                shown += 1
                if shown == 3:
                    break
    return agree


a_m = validate(mol, m_scaf_col, m_smi_col, "release ")
a_b = validate(base, b_scaf_col, b_smi_col, "baseline")

USE_STORED = (a_m or 0) >= 0.90 and (a_b or 0) >= 0.90
if USE_STORED:
    print("\nBoth columns agree with RDKit, so the full population can be "
          "compared\ndirectly. This is the comparison step 14 could only "
          "sample.")
else:
    print("\nAt least one column disagrees with RDKit. The stored columns are "
          "NOT\ninterchangeable, so the full-population comparison would be "
          "comparing two\ndifferent definitions. Falling back to computing "
          "both sides, which is slower.\nIf this run is too slow, that is the "
          "reason.")


# ================================================ 1b. full-population overlap
section("1b. Scaffold overlap over every compound")
marine_keys = set(mol[m_key_col].dropna().astype(str).str.split("-").str[0])
b_keys = base[b_key_col].astype(str).str.split("-").str[0]
rem = base[~b_keys.isin(marine_keys)].copy()
print(f"{len(base) - len(rem):,} baseline rows removed as marine, "
      f"{len(rem):,} remain")

# The marine side is the analysed scope, so results stay comparable with every
# other number in the project.
m_sub = mol[mol["id"].isin(set(prop["molecule_id"]))] if "id" in mol.columns \
    else mol

# Index is preserved in both branches so the per-molecule mapping below works.
# An earlier version built that mapping from the stored release column, which
# is empty, so every scaffold resolved to None and section 1c came back blank.
if USE_STORED:
    m_scaf = m_sub[m_scaf_col].map(norm_scaffold)
    b_scaf = rem[b_scaf_col].map(norm_scaffold)
else:
    print("computing scaffolds with RDKit, this takes a while")
    m_scaf = pd.Series([murcko(s) for s in m_sub[m_smi_col]],
                       index=m_sub.index)
    b_scaf = pd.Series([murcko(s) for s in rem[b_smi_col]], index=rem.index)
m_s, b_s = m_scaf.dropna(), b_scaf.dropna()

M, B = set(m_s), set(b_s)
only_m, both = M - B, M & B
pct = len(only_m) / len(M) * 100 if M else float("nan")
jac = len(both) / len(M | B) if (M or B) else float("nan")
print(f"\nmarine:   {len(m_s):,} compounds, {len(M):,} distinct scaffolds")
print(f"baseline: {len(b_s):,} compounds, {len(B):,} distinct scaffolds")
print(f"shared:        {len(both):,}")
print(f"marine only:   {len(only_m):,} ({pct:.1f}% of marine scaffolds)")
print(f"Jaccard:       {jac:.3f}")
print("\nThis is over the whole remainder, so 'marine only' now means absent "
      "from\nCOCONUT rather than absent from a sample. Compare it with step "
      "14's 76.1%:\nthe gap between the two is how much of that number was "
      "rarefaction.")

vc = m_s[m_s.isin(only_m)].value_counts()
top = vc.head(20).rename("n_compounds").to_frame()


# ================================= 1c. who makes the marine-only scaffolds
section("1c. Source lineage of the marine-only scaffolds")
print("If these skeletons trace to the lineages step 12 flagged, the "
      "structural and\nthe taxonomic results corroborate each other. If they "
      "trace anywhere, they are\nat least real rather than curation debris.")

tax = load_table(config.INTERIM / "organism_taxonomy")
org = pd.read_csv(config.RELEASE / config.TABLES["organisms"], low_memory=False)
mo = pd.read_csv(config.RELEASE / config.TABLES["molecule_organism"],
                 low_memory=False)
scaf_of = dict(zip(m_sub["id"], m_scaf)) if "id" in m_sub.columns else {}
if not scaf_of:
    print("no molecule id column in the release, cannot trace lineage")

link = (mo[["molecule_id", "organism_id"]].dropna().drop_duplicates()
        .merge(org[["id", "aphiaid_worms"]].rename(columns={"id": "organism_id"}),
               on="organism_id", how="left")
        .merge(tax[["aphiaid_worms", "phylum", "genus"]], on="aphiaid_worms",
               how="left")
        .dropna(subset=["phylum"]))
link["scaffold"] = link["molecule_id"].map(scaf_of)
link = link.dropna(subset=["scaffold"])
if "name_suspect" in prop.columns:
    link = link[~link["molecule_id"].isin(
        set(prop.loc[prop["name_suspect"], "molecule_id"]))]

path_of = dict(zip(prop["molecule_id"], prop.get(
    "np_classifier_pathway", pd.Series(dtype=object))))
link["pathway"] = link["molecule_id"].map(path_of)

src_rows = []
for s in top.index:
    sub = link[link["scaffold"] == s]
    if not len(sub):
        src_rows.append({"scaffold": s, "n_compounds": int(top.loc[s, "n_compounds"]),
                         "phyla": "", "top_genus": "", "pathway": ""})
        continue
    phy = sub["phylum"].value_counts()
    gen = sub["genus"].value_counts()
    pw = sub["pathway"].dropna()
    src_rows.append({
        "scaffold": s,
        "n_compounds": int(top.loc[s, "n_compounds"]),
        "phyla": ", ".join(phy.head(2).index.tolist()),
        "top_genus": gen.index[0] if len(gen) else "",
        "pathway": pw.mode().iloc[0] if len(pw) else "",
    })
srcs = pd.DataFrame(src_rows, columns=["scaffold", "n_compounds", "phyla",
                                       "top_genus", "pathway"])
print()
print(srcs.to_string(index=False) if len(srcs)
      else "no marine-only scaffolds to trace")


# ======================================================== 2. one axis or five
section("2. One axis or five?")
bpath = table_path(config.INTERIM / "baseline_properties")
if bpath is None:
    sys.exit("baseline_properties not found. Re-run scripts/14_baseline.py "
             "(the version dated 2026-10-15 saves it), then run this again.")
bp = load_table(config.INTERIM / "baseline_properties")
print(f"{len(bp):,} baseline compounds with a descriptor panel\n")

# Continuous descriptors get Cliff's delta. A boolean does not: delta on a
# 0/1 variable is bounded by the base rate, so a trait at 9% against 5% scores
# as negligible while the risk ratio is 1.8. Booleans are handled separately.
TARGETS = ["np_likeness", "fsp3", "n_stereocentres", "sa_score"]
BOOLS = ["macrocyclic"]
STRATA = [("0 aromatic rings", 0, 0), ("1", 1, 1), ("2", 2, 2),
          ("3 or more", 3, 99)]

rows = []
for name, lo, hi in [("all compounds", 0, 99)] + STRATA:
    m = prop[(prop["n_aromatic_rings"] >= lo) & (prop["n_aromatic_rings"] <= hi)]
    b = bp[(bp["n_aromatic_rings"] >= lo) & (bp["n_aromatic_rings"] <= hi)]
    rec = {"stratum": name, "n_marine": len(m), "n_baseline": len(b)}
    for t in TARGETS:
        if t in m.columns and t in b.columns:
            d, _ = cliffs_delta(m[t].astype(float), b[t].astype(float))
            rec[t] = round(d, 3) if not np.isnan(d) else np.nan
    for t in BOOLS:
        if t in m.columns and t in b.columns and len(m) and len(b):
            pm, pb = float(m[t].mean()), float(b[t].mean())
            rec[f"{t}_rr"] = round(pm / pb, 2) if pb else np.nan
    rows.append(rec)
strata = pd.DataFrame(rows)
print(strata.to_string(index=False))
print(f"\nStrata with fewer than {MIN_STRATUM:,} compounds on either side are "
      "not worth\nreading; the n columns are there for that. Columns ending "
      "_rr are risk ratios,\nnot deltas: 1.0 means no difference.")

base_row = strata.iloc[0]
inner = strata.iloc[1:]
ok = inner[(inner["n_marine"] >= MIN_STRATUM)
           & (inner["n_baseline"] >= MIN_STRATUM)]

print("\nverdict per descriptor:")
verdict = {}
for t in TARGETS + [f"{t}_rr" for t in BOOLS]:
    if t not in strata.columns or not len(ok):
        continue
    is_rr = t.endswith("_rr")
    overall = float(base_row[t])
    vals = ok[t].dropna().astype(float)
    if not len(vals):
        print(f"  {t:18s} overall {overall:+.3f}, no adequately powered "
              "stratum with a defined value")
        verdict[t] = "cannot tell"
        continue

    # Weight each stratum by its effective pair count. Using the largest
    # within-stratum value instead would let one stratum stand for all four,
    # which is how an effect confined to non-aromatic compounds reads as
    # "independent of aromaticity".
    w = (ok["n_marine"] * ok["n_baseline"]
         / (ok["n_marine"] + ok["n_baseline"])).loc[vals.index]
    pooled = float((vals * w).sum() / w.sum())

    if is_rr:
        # A risk ratio has no sign; distance from 1 is the effect.
        kept = abs(pooled - 1) / abs(overall - 1) if overall != 1 else np.nan
        flip = (pooled - 1) * (overall - 1) < 0
        negligible = abs(overall - 1) < 0.15
    else:
        kept = abs(pooled) / abs(overall) if overall else np.nan
        # A stratum pointing the other way only counts as a reversal if it
        # points there by a material amount. Requiring only a sign change
        # called -0.004 a reversal, which is noise crossing zero, not
        # Simpson's paradox. The pooled estimate must agree before the label
        # is used at all.
        opposed = vals[(np.sign(vals) != np.sign(overall))
                       & (vals.abs() >= MATERIAL)]
        flip = bool(len(opposed)) and np.sign(pooled) != np.sign(overall)
        negligible = abs(overall) < 0.147

    if negligible:
        tag = "no effect to explain"
    elif flip:
        tag = "REVERSES inside strata, the overall figure is composition"
    elif np.isnan(kept):
        tag = "cannot tell"
    elif kept < 0.5:
        tag = "collapses inside strata, explained by aromaticity"
    elif kept < 0.8:
        tag = "weakens substantially, mostly aromaticity"
    else:
        tag = "survives, independent of aromaticity"
    verdict[t] = tag
    print(f"  {t:18s} overall {overall:+.3f}, pooled within-stratum "
          f"{pooled:+.3f}  ->  {tag}")

print("\nA descriptor that survives stratification is a separate finding. One "
      "that\ncollapses is the aromaticity contrast under another name. One "
      "that reverses is\nSimpson's paradox: the unstratified difference comes "
      "from the two libraries\nhaving different aromatic-ring composition, "
      "not from the descriptor itself, and\nthe unstratified claim has to be "
      "withdrawn rather than softened.")


# ----------------------------------------------------------------------- save
pd.DataFrame({"scaffold": list(only_m)}).to_csv(
    config.INTERIM / "scaffold_overlap.csv", index=False)
srcs.to_csv(config.INTERIM / "scaffold_sources.csv", index=False)
strata.to_csv(config.INTERIM / "axis_strata.csv", index=False)
(config.INTERIM / "axis.json").write_text(json.dumps({
    "used_stored_scaffold_columns": bool(USE_STORED),
    "scaffold_column_agreement_release": None if a_m is None else round(a_m, 4),
    "scaffold_column_agreement_baseline": None if a_b is None else round(a_b, 4),
    "baseline_remainder": int(len(rem)),
    "scaffolds_marine": len(M),
    "scaffolds_baseline": len(B),
    "scaffolds_shared": len(both),
    "scaffolds_marine_only": len(only_m),
    "pct_marine_only": None if np.isnan(pct) else round(pct, 2),
    "scaffold_jaccard": None if np.isnan(jac) else round(jac, 4),
    "stratified_verdict": verdict,
}, indent=2, default=float))
print(f"\nWrote scaffold_overlap.csv, scaffold_sources.csv, axis_strata.csv, "
      f"axis.json\nto {config.INTERIM}")
