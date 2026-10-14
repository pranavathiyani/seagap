"""Correct heteroscedastic breadth residuals, attach uncertainty, re-rank.

    python scripts/09_uncertainty.py

Steps 07 and 08 divided accumulation-curve residuals by a single global standard
deviation. Residual scatter is not constant: a genus with 12 compounds has a far
noisier family count than one with 1,200, so dividing both by the same number
inflates the small ones. In simulation, with every taxon given identical true
breadth, residual sd was 2.1x larger in the smallest sample-size bin, 18.5% of
small-n taxa scored z > 1 against 6.0% of large-n taxa, and the top 50 by global z
had a median sample size of 46 against an overall median of 155. That is the whole
reason the previous top 30 contained nothing above 33 compounds.

Two fixes:

  local standardisation  residual sd is modelled as a function of log(n) and each
  residual divided by the sd expected at its own sample size. Under a simulated
  null this gives 92% coverage at a nominal 95%, and z sd within 0.85-1.13 in
  every sample-size bin.

  variance decomposition  residual scatter mixes real differences in breadth with
  sampling noise. Resampling a taxon's compounds estimates the sampling part, and
  the remainder is the biological part. That ratio says whether breadth variation
  across taxa is worth interpreting at all.

A per-taxon bootstrap confidence interval was tried and discarded. Richness is not
a smooth functional: resampling n compounds with replacement from n observed
compounds misses about 37% of them, so the interval is shifted rather than centred
and cannot represent families a fresh sample would have found. Measured coverage
was 22% for the percentile interval, 21% bias-corrected and 61% using the
bootstrap sd, against a nominal 95%. Resampling is therefore used only to size the
sampling variance, which is a valid use of it, and significance comes from
z_local.

Writes taxon_final.parquet, priority_final.csv and uncertainty.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seagap import config  # noqa: E402
from seagap.tables import load_table, save_table  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 36)
print("SCRIPT VERSION: 2026-10-14 uncertainty\n")

RANK = config.TAXON_RANK
MIN_N = 10
BOOT = 500
SEED = 0

bar = "=" * 98


def section(t):
    print(f"\n{bar}\n{t}\n{bar}")


# ------------------------------------------------------------------ load
work = load_table(config.INTERIM / "families")
fam = load_table(config.INTERIM / "family_profile").set_index("family")
tax = load_table(config.INTERIM / "organism_taxonomy")
org = pd.read_csv(config.RELEASE / config.TABLES["organisms"], low_memory=False)
mol_org = pd.read_csv(config.RELEASE / config.TABLES["molecule_organism"],
                      low_memory=False)

link = (mol_org[["molecule_id", "organism_id"]].dropna().drop_duplicates()
        .merge(org[["id", "aphiaid_worms"]].rename(columns={"id": "organism_id"}),
               on="organism_id", how="left")
        .merge(tax[["aphiaid_worms", "genus", "order", "phylum"]],
               on="aphiaid_worms", how="left")
        .merge(work[["molecule_id", "family"]], on="molecule_id", how="inner")
        .dropna(subset=[RANK])
        .drop_duplicates(subset=["molecule_id", RANK, "family"]))

# specialised-metabolite scope, as fixed in step 08 (near-inert but documented)
spec = set(fam[~fam["is_primary"]].index)
sl = link[link["family"].isin(spec)]
obs = (sl.groupby(RANK)
       .agg(n_compounds=("molecule_id", "nunique"),
            n_families=("family", "nunique")))
obs = obs[obs["n_compounds"] >= MIN_N].copy()
print(f"{len(obs):,} {RANK} values with >={MIN_N} specialised-metabolite compounds")

# ------------------------------------------------------------------ fit
x = np.log10(obs["n_compounds"].values.astype(float))
y = np.log10(obs["n_families"].values.astype(float))
b1, b0 = np.polyfit(x, y, 1)
resid = y - (b0 + b1 * x)
r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
obs["resid"] = resid
print(f"accumulation curve: families = {10 ** b0:.2f} x n^{b1:.3f}, R2 = {r2:.3f}")

# ------------------------------------------------------------------ 1
section("1. Is residual scatter constant across sample size?")
obs["bin"] = pd.qcut(x, 6, labels=False, duplicates="drop")
diag = obs.groupby("bin").agg(n_taxa=("n_compounds", "size"),
                              median_n=("n_compounds", "median"),
                              resid_sd=("resid", "std"))
g_sd = resid.std(ddof=1)
diag["z_global_sd"] = (diag["resid_sd"] / g_sd).round(3)
print(f"global residual sd: {g_sd:.4f}\n")
print(diag.round(4).to_string())
spread = diag["resid_sd"].iloc[0] / diag["resid_sd"].iloc[-1]
print(f"\nsmallest-n bin scatter is {spread:.2f}x the largest-n bin")
if spread > 1.2:
    print("Dividing every residual by one global sd therefore overstates small-n "
          "taxa by\nroughly that factor, which is a ranking artefact rather than "
          "a finding.")

# local sd: E|resid| regressed on log n, converted to sd under normality
c1, c0 = np.polyfit(x, np.abs(resid), 1)
sd_hat = np.maximum((c0 + c1 * x) * np.sqrt(np.pi / 2), 1e-6)
obs["sd_local"] = sd_hat
obs["z_global"] = (resid / g_sd).round(2)
obs["z_local"] = (resid / sd_hat).round(2)
print(f"\nlocal sd model: sd(n) = ({c0:.4f} {c1:+.4f} * log10 n) * sqrt(pi/2)")
print(f"  at n=10   sd = {(c0 + c1 * 1) * np.sqrt(np.pi / 2):.4f}")
print(f"  at n=100  sd = {(c0 + c1 * 2) * np.sqrt(np.pi / 2):.4f}")
print(f"  at n=1000 sd = {(c0 + c1 * 3) * np.sqrt(np.pi / 2):.4f}")

print("\nmedian sample size among the top 50 by each measure:")
print(f"  z_global {obs.nlargest(50, 'z_global')['n_compounds'].median():>7.0f}")
print(f"  z_local  {obs.nlargest(50, 'z_local')['n_compounds'].median():>7.0f}")
print(f"  overall  {obs['n_compounds'].median():>7.0f}")

# ------------------------------------------------------------------ 2
section("2. How much of the breadth variation is real?")
print(f"Residual scatter mixes genuine differences in chemical breadth with "
      f"sampling\nnoise. Resampling each taxon's compounds ({BOOT} draws) sizes "
      "the sampling part.")

rng = np.random.default_rng(SEED)
fam_by_taxon = sl.groupby(RANK)["family"].apply(np.array)
samp_sd = {}
for t, arr in fam_by_taxon.items():
    if t not in obs.index:
        continue
    k = len(arr)
    draws = np.array([len(np.unique(rng.choice(arr, k, replace=True)))
                      for _ in range(BOOT)], dtype=float)
    # sd on the log10 scale, comparable to the residuals
    samp_sd[t] = float(np.log10(np.maximum(draws, 1)).std(ddof=1))

obs["samp_sd"] = pd.Series(samp_sd)
obs["z_sampling_sd"] = (obs["samp_sd"] / obs["sd_local"]).round(3)

mean_frac = float((obs["z_sampling_sd"] ** 2).mean())
bio_frac = max(0.0, 1.0 - mean_frac)
print(f"\nmean sampling variance as a fraction of total residual variance: "
      f"{mean_frac:.3f}")
print(f"remainder attributable to real between-taxon variation:            "
      f"{bio_frac:.3f}")
print("\nBootstrap underestimates the sampling variance of a richness count, so "
      "the\nbiological fraction above is an upper bound. It is reported because "
      "a value\nnear zero would mean breadth differences are noise and the "
      "ranking is empty.")

print("\nsampling share of variance by sample-size bin:")
print(obs.groupby("bin").agg(
    median_n=("n_compounds", "median"),
    sampling_var_share=("z_sampling_sd", lambda s: round(float((s ** 2).mean()), 3)),
).to_string())

# Significance from z_local, which carries the verified scale.
obs["notable"] = obs["z_local"].abs() > 1.96
print(f"\n|z_local| > 1.96: {int(obs['notable'].sum()):,} of {len(obs):,} taxa "
      f"({obs['notable'].mean() * 100:.1f}%)")
print("z_local is scaled by the TOTAL across-taxon scatter, so this is a "
      "conservative\ntest against the curve and the right scale for the question "
      "SEAGAP asks:\nunusual relative to other taxa at comparable effort.")

# ------------------------------------------------------------------ 3
section("3. Final ranking")
prof = load_table(config.INTERIM / "taxon_ranked").set_index(RANK)
R = obs.join(prof[["spec_rarity", "primary_frac", "phylum"]], how="left")
R = R.dropna(subset=["spec_rarity"])

R["breadth_pct"] = R["z_local"].rank(pct=True) * 100
R["rarity_pct"] = R["spec_rarity"].rank(pct=True) * 100
R["headroom_pct"] = (1 - R["n_compounds"].rank(pct=True)) * 100
R["priority"] = ((R["breadth_pct"] * R["rarity_pct"] * R["headroom_pct"])
                 ** (1 / 3)).round(1)

cols = ["phylum", "n_compounds", "n_families", "z_local", "z_global",
        "spec_rarity", "notable", "priority"]
top = R.sort_values("priority", ascending=False)
print(f"All {len(R):,} ranked. Top 25:")
print(top.head(25)[cols].to_string())

conf = R[R["notable"] & (R["z_local"] > 0)].sort_values("priority",
                                                        ascending=False)
print(f"\nRestricted to taxa above the curve at |z_local| > 1.96 "
      f"({len(conf):,} taxa). Top 25:")
print(conf.head(25)[cols].to_string() if len(conf) else "  none")

print("\nmedian sample size in each list:")
print(f"  all ranked            {R['n_compounds'].median():.0f}")
print(f"  unrestricted top 100  {top.head(100)['n_compounds'].median():.0f}")
if len(conf):
    print(f"  notable top 100       {conf.head(100)['n_compounds'].median():.0f}")
print("\nIf the unrestricted top 100 still sits far below the overall median, "
      "headroom_pct\nis pulling the ranking toward thinly sampled taxa by "
      "design, not by artefact.")

for lab, frame in [("unrestricted top 100", top.head(100)),
                   ("notable top 100", conf.head(100) if len(conf) else None)]:
    if frame is None or not len(frame):
        continue
    sp = pd.DataFrame({"top": frame["phylum"].value_counts(),
                       "all": R["phylum"].value_counts()}).fillna(0).astype(int)
    sp["enrichment"] = (sp["top"] / len(frame)
                        / (sp["all"] / len(R))).round(2)
    print(f"\nphylum enrichment, {lab}:")
    print(sp.sort_values("top", ascending=False).head(10).to_string())

# ------------------------------------------------------------------ save
save_table(R.reset_index(), config.INTERIM / "taxon_final")
top.reset_index().to_csv(config.INTERIM / "priority_final.csv", index=False)
(config.INTERIM / "uncertainty.json").write_text(json.dumps({
    "taxon_rank": RANK, "min_compounds": MIN_N, "bootstrap": BOOT, "seed": SEED,
    "accumulation": {"a": round(float(10 ** b0), 4), "b": round(float(b1), 4),
                     "r2": round(float(r2), 4)},
    "global_resid_sd": round(float(g_sd), 4),
    "sd_model": {"intercept": round(float(c0), 5), "slope": round(float(c1), 5)},
    "scatter_ratio_small_over_large_n": round(float(spread), 3),
    "sampling_var_share": round(float(mean_frac), 4),
    "biological_var_share_upper_bound": round(float(bio_frac), 4),
    "ci_method_rejected": ("per-taxon bootstrap CI: coverage 22% percentile, "
                           "21% bias-corrected, 61% via bootstrap sd, vs nominal "
                           "95%; richness is not a smooth functional"),
    "significance": "|z_local| > 1.96",
    "n_ranked": int(len(R)),
    "n_notable": int(R["notable"].sum()),
    "n_notable_positive": int(len(conf)),
    "median_n_top50_zglobal": float(obs.nlargest(50, "z_global")["n_compounds"].median()),
    "median_n_top50_zlocal": float(obs.nlargest(50, "z_local")["n_compounds"].median()),
    "median_n_overall": float(obs["n_compounds"].median()),
}, indent=2, default=float))
print(f"\nWrote taxon_final, priority_final.csv, uncertainty.json\n"
      f"to {config.INTERIM}")
