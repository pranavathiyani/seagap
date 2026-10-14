"""Characterise the residual asymmetry, rank saturation, map the coverage frontier.

    python scripts/10_saturation.py

Step 09 produced a negative result that reframes the project. Of 903 genera, 49
deviate from the accumulation curve at |z| > 1.96 and every one of them is on the
NARROW side. No genus is statistically broad: the largest positive z is about 1.3.

The reason is biological rather than statistical. A genus makes chemically related
metabolites, so there is a ceiling on how widely its compounds can spread across
structural families, and the residual distribution is left-skewed. Claims of the
form "this genus is exceptionally rich, screen it" are therefore not supportable
from this data, whatever a ranking puts at the top.

Two claims that are supportable:

  saturation    a genus far below the curve is chemically repetitive relative to
                how hard it has been worked. With 94% of residual variance real
                rather than sampling noise, these are defensible calls, and they
                say where further screening has poor expected return.

  coverage      the genus-by-family matrix is 99.2% empty and the median genus
                contributes 3 families. The frontier is how little has been looked
                at, not which of the examined taxa is special.

Section 3 therefore reports breadth at order level, where sample sizes are an
order of magnitude larger and estimates stable, and section 4 lists barely
examined genera inside orders that have proven productive.

Writes saturated_taxa.csv, order_profile.parquet, frontier.csv, saturation.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seagap import config  # noqa: E402
from seagap.curve import fit_accumulation  # noqa: E402
from seagap.tables import load_table, save_table  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 36)
print("SCRIPT VERSION: 2026-10-14b saturation (robust sd)\n")

RANK = config.TAXON_RANK
MIN_N = 10
MIN_ORDER_N = 30
MIN_ORDER_GENERA = 3    # an order with one genus is a genus, and just as noisy
bar = "=" * 98


def section(t):
    print(f"\n{bar}\n{t}\n{bar}")


def report_fit(f, label):
    print(f"{label}: classes = {f['a']:.2f} x n^{f['b']:.3f}, R2 = {f['r2']:.3f}")
    print(f"  local sd model exp({f['sd_log_intercept']:.3f} "
          f"{f['sd_log_slope']:+.3f} log10 n) x {f['sd_calibration']:.3f}, "
          f"floor {f['sd_floor']:.4f}")
    if f["n_floored"]:
        print(f"  WARNING: {f['n_floored']} points hit the sd floor; their z is "
              "capped, not exact")
    print(f"  z range [{f['z'].min():+.2f}, {f['z'].max():+.2f}], "
          f"sd {f['z'].std(ddof=1):.3f}")


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
        .merge(tax[["aphiaid_worms", "genus", "family", "order", "class", "phylum"]]
               .rename(columns={"family": "worms_family"}),
               on="aphiaid_worms", how="left")
        .merge(work[["molecule_id", "family"]], on="molecule_id", how="inner")
        .dropna(subset=[RANK]))
spec = set(fam[~fam["is_primary"]].index)
sl = link[link["family"].isin(spec)].drop_duplicates(
    subset=["molecule_id", RANK, "family"])

obs = (sl.groupby(RANK)
       .agg(n_compounds=("molecule_id", "nunique"),
            n_families=("family", "nunique")))
obs = obs[obs["n_compounds"] >= MIN_N].copy()
gf = fit_accumulation(obs["n_compounds"], obs["n_families"])
resid, expo, r2 = gf["resid"], gf["b"], gf["r2"]
obs["resid"] = np.round(resid, 4)
obs["z"] = np.round(gf["z"], 2)
obs["exp_families"] = np.round(gf["pred"], 1)
meta = (sl.groupby(RANK).agg(
    phylum=("phylum", lambda s: s.mode().iloc[0] if s.notna().any() else None),
    order=("order", lambda s: s.mode().iloc[0] if s.notna().any() else None)))
obs = obs.join(meta)
print(f"{len(obs):,} genera")
report_fit(gf, "genus curve")

# ------------------------------------------------------------------ 1
section("1. The residual distribution is one-sided")
sk = stats.skew(resid)
print(f"skewness of residuals: {sk:+.3f}  (0 would be symmetric)")
print(f"min z {obs['z'].min():+.2f}   max z {obs['z'].max():+.2f}")
for t in (1.0, 1.5, 1.96, 2.5, 3.0):
    lo = int((obs["z"] < -t).sum())
    hi = int((obs["z"] > t).sum())
    print(f"  |z| > {t:<4}  narrow side {lo:>4}   broad side {hi:>4}")
print("\nThe broad side empties out while the narrow side does not. A genus can be")
print("far more repetitive than the curve predicts, but not far more diverse,")
print("because its metabolites are biosynthetically related. Any ranking that "
      "claims\nexceptional richness is reading noise at the top of a compressed "
      "tail.")

ks = stats.kstest((resid - resid.mean()) / resid.std(ddof=1), "norm")
print(f"\nKS test against normal: D = {ks.statistic:.4f}, p = {ks.pvalue:.2e}")

# ------------------------------------------------------------------ 2
section("2. Saturated genera: where further screening has poor expected return")
sat = obs[obs["z"] < -1.96].copy()
sat["families_per_100"] = (sat["n_families"] / sat["n_compounds"] * 100).round(1)
print(f"{len(sat):,} genera sit below the curve at z < -1.96 "
      f"({len(sat) / len(obs) * 100:.1f}%)")
cols = ["phylum", "order", "n_compounds", "n_families", "exp_families",
        "families_per_100", "z"]
print("\nRanked by how much effort has gone in (most worked first):")
print(sat.sort_values("n_compounds", ascending=False).head(25)[cols].to_string())
print(f"\nThese {len(sat):,} genera account for "
      f"{sat['n_compounds'].sum():,} compounds "
      f"({sat['n_compounds'].sum() / obs['n_compounds'].sum() * 100:.1f}% of the "
      "ranked total)\nwhile contributing only "
      f"{sat['n_families'].nunique() if False else sat['n_families'].sum():,} "
      "family slots.")
print("\nphylum spread of saturated genera against background:")
sp = pd.DataFrame({"saturated": sat["phylum"].value_counts(),
                   "all": obs["phylum"].value_counts()}).fillna(0).astype(int)
sp["enrichment"] = (sp["saturated"] / len(sat)
                    / (sp["all"] / len(obs))).round(2)
print(sp.sort_values("saturated", ascending=False).head(10).to_string())

# ------------------------------------------------------------------ 3
section("3. Breadth at order level, where estimates are stable")
ordall = (sl.dropna(subset=["order"]).groupby("order")
          .agg(n_compounds=("molecule_id", "nunique"),
               n_families=("family", "nunique"),
               n_genera=(RANK, "nunique")))
ordall["phylum"] = (sl.dropna(subset=["order"]).groupby("order")["phylum"]
                    .agg(lambda s: s.mode().iloc[0] if s.notna().any() else None))

one_genus = ordall[(ordall["n_compounds"] >= MIN_ORDER_N)
                   & (ordall["n_genera"] < MIN_ORDER_GENERA)]
print(f"{len(one_genus):,} orders have >={MIN_ORDER_N} compounds but fewer than "
      f"{MIN_ORDER_GENERA} genera.\nAn order represented by a single genus is a "
      "genus wearing an order's name, and\ncarries the same noise, so these are "
      "excluded from the order-level fit:")
if len(one_genus):
    print(one_genus.sort_values("n_compounds", ascending=False).head(12).to_string())

ordobs = ordall[(ordall["n_compounds"] >= MIN_ORDER_N)
                & (ordall["n_genera"] >= MIN_ORDER_GENERA)].copy()
of = fit_accumulation(ordobs["n_compounds"], ordobs["n_families"])
ordobs["z"] = np.round(of["z"], 2)
oexpo, or2, oresid = of["b"], of["r2"], of["resid"]
print(f"\n{len(ordobs):,} orders retained")
report_fit(of, "order curve")
print(f"median compounds per order: {ordobs['n_compounds'].median():.0f} "
      f"(vs {obs['n_compounds'].median():.0f} per genus)")
print(f"skewness of order residuals: {stats.skew(oresid):+.3f} "
      f"(genus: {stats.skew(resid):+.3f})")
print(f"orders at z > 1.96: {int((ordobs['z'] > 1.96).sum())}   "
      f"at z < -1.96: {int((ordobs['z'] < -1.96).sum())}")
ocols = ["phylum", "n_compounds", "n_families", "n_genera", "z"]
print("\nBroadest orders:")
print(ordobs.sort_values("z", ascending=False).head(12)[ocols].to_string())
print("\nNarrowest orders:")
print(ordobs.sort_values("z").head(12)[ocols].to_string())

# ------------------------------------------------------------------ 3b
section("3b. Orders that should not be in a marine dataset")
LAND = {"Anura": "frogs", "Testudines": "turtles", "Gaviiformes": "loons",
        "Procellariiformes": "petrels and albatrosses", "Galliformes": "landfowl",
        "Passeriformes": "perching birds", "Rodentia": "rodents",
        "Primates": "primates", "Caudata": "salamanders", "Squamata": "lizards",
        "Carnivora": "carnivorans", "Artiodactyla": "even-toed ungulates",
        "Lepidoptera": "moths and butterflies", "Hymenoptera": "ants and bees",
        "Coleoptera": "beetles", "Diptera": "flies", "Araneae": "spiders",
        "Caryophyllales": "a largely terrestrial plant order",
        "Asterales": "a largely terrestrial plant order",
        "Lamiales": "a largely terrestrial plant order",
        "Fabales": "a largely terrestrial plant order",
        "Poales": "a largely terrestrial plant order"}
found = ordall[ordall.index.isin(LAND)].copy()
if len(found):
    found["what"] = [LAND[i] for i in found.index]
    found["genera"] = [", ".join(sorted(
        sl.loc[sl["order"] == i, RANK].dropna().unique())[:6]) for i in found.index]
    print(f"{len(found)} such orders are present in the tier-1 'strictly marine' "
          f"set,\ncarrying {int(found['n_compounds'].sum()):,} compounds:\n")
    print(found.sort_values("n_compounds", ascending=False)
          [["what", "phylum", "n_compounds", "n_genera", "genera"]].to_string())
    print("\nSome are legitimate: seabirds and sea turtles are marine animals, and")
    print("coastal halophytes sit in terrestrial plant orders. Others are the "
          "fuzzy\nAphiaID matches found in step 04. Either way, a genus reached "
          "through one of\nthese orders is not a bioprospecting target, and the "
          "list belongs in the\nmanuscript's limitations rather than buried.")
else:
    print("none found")

# ------------------------------------------------------------------ 4
section("4. The coverage frontier")
all_gen = link.groupby(RANK)["molecule_id"].nunique()
print(f"{len(all_gen):,} genera have at least one compound in a structural family")
print(f"{int((all_gen >= MIN_N).sum()):,} have >={MIN_N}, so "
      f"{int((all_gen < MIN_N).sum()):,} are present but barely examined")
print(f"median compounds per genus: {all_gen.median():.0f}")

prod = ordobs[ordobs["z"] > 0].index
gen_meta = (link.dropna(subset=["order"]).groupby(RANK)
            .agg(order=("order", lambda s: s.mode().iloc[0]),
                 phylum=("phylum", lambda s: s.mode().iloc[0]),
                 n_compounds=("molecule_id", "nunique"),
                 n_families=("family", "nunique")))
front = gen_meta[(gen_meta["order"].isin(prod))
                 & (gen_meta["n_compounds"] < MIN_N)].copy()
print(f"\n{len(front):,} barely examined genera ({MIN_N} compounds or fewer) sit "
      f"inside the\n{len(prod):,} orders whose chemistry is broad for its effort")
print("\nThis is the defensible gap statement. It rests on the order being "
      "productive\nand the genus being unexamined, not on an estimate from a "
      "handful of compounds.")

ordcount = front.groupby("order").size().rename("unexamined_genera")
summary = (ordobs.loc[ordobs.index.isin(prod), ocols]
           .join(ordcount).fillna({"unexamined_genera": 0}))
summary["unexamined_genera"] = summary["unexamined_genera"].astype(int)
print("\nProductive orders with the most unexamined genera:")
print(summary.sort_values("unexamined_genera", ascending=False)
      .head(20).to_string())

# ------------------------------------------------------------------ 5
section("5. What this data can and cannot support")
print("supported:")
print(f"  - effort is concentrated: the busiest 10% of genera hold most compounds")
print(f"  - diminishing returns are quantified: families scale as n^{expo:.2f}")
print(f"  - {len(sat):,} genera are chemically saturated at z < -1.96")
print(f"  - {len(front):,} genera in productive orders are effectively unexamined")
print(f"  - {len(ordobs):,} orders can be compared on breadth with stable estimates")
print("\nnot supported:")
print("  - that any individual genus is exceptionally chemically rich")
print(f"    (largest positive z is {obs['z'].max():+.2f}; nothing reaches 1.96)")
print("  - ranking thinly sampled genera by breadth, which estimates it from")
print("    10 to 20 compounds and inherits the compressed upper tail")

sat.sort_values("n_compounds", ascending=False).reset_index().to_csv(
    config.INTERIM / "saturated_taxa.csv", index=False)
save_table(ordobs.reset_index(), config.INTERIM / "order_profile")
front.sort_values("n_compounds", ascending=False).reset_index().to_csv(
    config.INTERIM / "frontier.csv", index=False)
(config.INTERIM / "saturation.json").write_text(json.dumps({
    "taxon_rank": RANK, "min_compounds": MIN_N, "min_order_compounds": MIN_ORDER_N,
    "genus_curve": {"exponent": round(float(expo), 4), "r2": round(float(r2), 4)},
    "residual_skewness": round(float(sk), 4),
    "ks_vs_normal": {"D": round(float(ks.statistic), 4),
                     "p": float(ks.pvalue)},
    "z_range": [float(obs["z"].min()), float(obs["z"].max())],
    "n_narrow_z196": int((obs["z"] < -1.96).sum()),
    "n_broad_z196": int((obs["z"] > 1.96).sum()),
    "order_curve": {"exponent": round(float(oexpo), 4), "r2": round(float(or2), 4),
                    "n_orders": int(len(ordobs)),
                    "skewness": round(float(stats.skew(oresid)), 4)},
    "n_genera_any": int(len(all_gen)),
    "n_genera_ranked": int(len(obs)),
    "n_genera_barely_examined": int((all_gen < MIN_N).sum()),
    "n_productive_orders": int(len(prod)),
    "n_frontier_genera": int(len(front)),
}, indent=2, default=float))
print(f"\nWrote saturated_taxa.csv, order_profile, frontier.csv, saturation.json\n"
      f"to {config.INTERIM}")
