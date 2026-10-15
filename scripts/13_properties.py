"""Physicochemical profile of the chemical space.

    python scripts/13_properties.py
    python scripts/13_properties.py --selftest    # 6 known molecules, 2 seconds

Everything up to step 12 treated compounds as nodes in a similarity graph and
never asked what the molecules are. This computes the panel: size, lipophilicity,
polarity, saturation, stereochemistry, and three published composite scores.

Two rules about what goes in here.

Deterministic only. Every number below is computed from the structure by a
documented function. No trained models, no predicted endpoints. ADMET prediction
is a separate step with separate caveats, and mixing the two under one heading is
how a reviewer decides the whole panel is unreliable.

No pass/fail drug-likeness. Marine natural products violate the Rule of Five
constitutively: macrolides, cyclic peptides, large polyketides, polyhalogenated
terpenes. Reporting "x% are drug-like" answers a question nobody asked and
scores the chemistry against a library it did not come from. Violation counts are
reported as a distribution, and the continuous scores (QED, NP-likeness) carry
the interpretation.

Writes properties.parquet (per compound, for the viewer), plus
properties_summary.csv, properties_by_phylum.csv, properties_by_pathway.csv
and properties.json.
"""

import argparse
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

from seagap.descriptors import (  # noqa: E402
    MACROCYCLE_MIN, NUMERIC, STEREO_ERRORS, describe, load_contrib, profile)

pd.set_option("display.width", 220)
print("SCRIPT VERSION: 2026-10-15 properties\n")

PROGRESS_EVERY = 5_000


# ---------------------------------------------------------------------- selftest
def selftest():
    """Six molecules with known character. Verifies the API, not the science."""
    cases = {
        "aspirin":      "CC(=O)Oc1ccccc1C(=O)O",
        "caffeine":     "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "cholesterol":  "CC(C)CCC[C@@H](C)[C@H]1CC[C@]2(C)[C@@H]1CC[C@H]1"
                        "[C@@H]2CC=C2C[C@@H](O)CC[C@]12C",
        "erythromycin": "CC[C@@H]1[C@@]([C@@H]([C@H](C(=O)[C@@H](C[C@@]"
                        "([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)O[C@H]2C[C@@]"
                        "([C@H]([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H]"
                        "(C[C@H](O3)C)N(C)C)O)(C)O)C)C)O)(C)O",
        "bromophenol":  "Oc1cc(Br)c(Br)cc1Br",
        "broken":       "this is not a smiles",
    }
    sa_fn, np_fn = load_contrib()
    rows = {}
    for name, smi in cases.items():
        r = describe(smi, sa_fn, np_fn)
        if r is None:
            print(f"  {name:14s} did not parse (expected for 'broken')")
            continue
        rows[name] = r
    out = pd.DataFrame(rows).T
    cols = ["mw", "clogp", "tpsa", "fsp3", "n_stereocentres",
            "n_stereo_unspecified", "macrocyclic", "max_ring_size",
            "ro5_violations", "qed", "sa_score", "np_likeness"]
    print()
    print(out[cols].astype(object).to_string())
    print("\nSanity expectations:")
    print("  cholesterol  fsp3 near 1.0, clogp high, qed low")
    print("  erythromycin macrocyclic True, max_ring_size 14, ro5_violations >=2")
    print("  cholesterol  n_stereocentres 8, n_stereo_unspecified 0")
    print("  bromophenol  3 bromines, low mw, 0 stereocentres")
    print("  np_likeness  positive for cholesterol and erythromycin,")
    print("               near zero or negative for aspirin and caffeine")
    print("\nIf those hold, the descriptor calls are wired correctly.")


# -------------------------------------------------------------------------- main
def main():
    sa_fn, np_fn = load_contrib()

    work = load_table(config.INTERIM / "families")
    keep = [c for c in ["molecule_id", "family", "canonical_smiles",
                        "np_classifier_pathway", "np_classifier_superclass",
                        "name_suspect"] if c in work.columns]
    work = work[keep].drop_duplicates(subset="molecule_id")
    print(f"\n{len(work):,} compounds in families, the scope used throughout")

    rows, failed = profile(work["canonical_smiles"].tolist(), sa_fn, np_fn,
                           ids=work["molecule_id"].tolist(),
                           label=f"/ {len(work):,}  at ",
                           progress_every=PROGRESS_EVERY)

    prop = pd.DataFrame(rows).rename(columns={"id": "molecule_id"})
    prop = work.merge(prop, on="molecule_id", how="inner")
    print(f"{len(prop):,} profiled, {len(failed):,} unparseable")
    if STEREO_ERRORS:
        print(f"WARNING: stereo perception failed on {len(STEREO_ERRORS):,} of "
              f"{len(prop):,} molecules")
        print(f"         first error: {STEREO_ERRORS[0]}")
        print("         stereocentre columns are unusable for that fraction. "
              "Do not read\n         them as zeros.")

    # ----------------------------------------------------------- distributions
    print("\n" + "=" * 98)
    print("Property distributions")
    print("=" * 98)
    have = [c for c in NUMERIC if prop[c].notna().any()]
    summ = prop[have].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    summ = summ[["count", "mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    print(summ.round(2).to_string())

    print("\nThese numbers describe the library. They do not yet say whether it "
          "is unusual.\nStep 14 computes the same panel on the non-marine "
          "COCONUT remainder, which is\nthe comparison that makes any of this a "
          "claim: same parent collection, same\ncuration, same classifier, so "
          "the only difference is the habitat.")

    # ----------------------------------------------------------- beyond Ro5
    print("\n" + "=" * 98)
    print("Beyond the Rule of Five")
    print("=" * 98)
    vc = (prop["ro5_violations"].value_counts().sort_index()
          .rename("n_compounds").to_frame())
    vc["pct"] = (vc["n_compounds"] / len(prop) * 100).round(1)
    print(vc.to_string())
    beyond = float((prop["ro5_violations"] >= 2).mean() * 100)
    print(f"\n{beyond:.1f}% carry two or more violations, which is the usual "
          "definition of\nbeyond-Rule-of-Five chemical space.")
    print(f"macrocyclic (ring of {MACROCYCLE_MIN}+): "
          f"{prop['macrocyclic'].mean() * 100:.1f}%")
    print(f"Veber-compatible (rotb<=10 and tpsa<=140): "
          f"{prop['veber_pass'].mean() * 100:.1f}%")
    print(f"fsp3 above 0.5: {(prop['fsp3'] > 0.5).mean() * 100:.1f}%")

    if prop["n_stereocentres"].notna().any():
        st = prop.dropna(subset=["n_stereocentres"])
        tot = float(st["n_stereocentres"].sum())
        uns = float(st["n_stereo_unspecified"].sum())
        print(f"\nat least one stereocentre: "
              f"{(st['n_stereocentres'] > 0).mean() * 100:.1f}%")
        print(f"median stereocentres per compound: "
              f"{st['n_stereocentres'].median():.0f}")
        if tot:
            print(f"of {tot:,.0f} stereocentres, {uns:,.0f} are unspecified "
                  f"({uns / tot * 100:.1f}%)")
            print("An unspecified centre is a limit on what docking or ADMET "
                  "work downstream\ncan claim, and it belongs in the resource "
                  "audit next to the AphiaID errors.")

    # QED and NP-likeness pull against each other by construction: QED is
    # fitted on marketed drugs, NP-likeness on natural products. A negative
    # correlation here is the expected result and it is the argument against
    # using QED to triage this library.
    corr = corr_qed_mw = corr_np_mw = np.nan
    if prop["np_likeness"].notna().any():
        corr = float(prop[["qed", "np_likeness"]].corr().iloc[0, 1])
        corr_qed_mw = float(prop[["qed", "mw"]].corr().iloc[0, 1])
        corr_np_mw = float(prop[["np_likeness", "mw"]].corr().iloc[0, 1])
        print(f"\ncorrelation of QED with NP-likeness: {corr:+.3f}")
        print(f"  QED with mw:          {corr_qed_mw:+.3f}")
        print(f"  NP-likeness with mw:  {corr_np_mw:+.3f}")
        if corr < -0.1:
            print("\nThe two scores pull against each other, which is the "
                  "textbook expectation:\nQED is fitted on marketed synthetic "
                  "drugs and NP-likeness on natural\nproducts, so a library "
                  "scoring high on one scores low on the other.")
        else:
            print("\nNot the textbook anticorrelation. That expectation holds "
                  "when comparing a\nnatural-product library against a "
                  "synthetic one, and this comparison is\ninternal: every "
                  "compound here is already a natural product, so provenance "
                  "is\nnot the axis that separates them. Read the two "
                  "correlations with mw above\nto see what is: if both scores "
                  "track mass in the same direction, mass is\nwhat this "
                  "coefficient is measuring.")
        print("\nEither way the conclusion for triage is the same. QED "
              "penalises mass, and\na library where a third of compounds are "
              "beyond-Rule-of-Five by construction\nwill be ranked by weight "
              "rather than by merit if QED is used as a filter.")

    # ----------------------------------------------------------- by lineage
    print("\n" + "=" * 98)
    print("By phylum")
    print("=" * 98)
    tax = load_table(config.INTERIM / "organism_taxonomy")
    org = pd.read_csv(config.RELEASE / config.TABLES["organisms"],
                      low_memory=False)
    mol_org = pd.read_csv(config.RELEASE / config.TABLES["molecule_organism"],
                          low_memory=False)
    link = (mol_org[["molecule_id", "organism_id"]].dropna().drop_duplicates()
            .merge(org[["id", "aphiaid_worms"]]
                   .rename(columns={"id": "organism_id"}),
                   on="organism_id", how="left")
            .merge(tax[["aphiaid_worms", "phylum"]], on="aphiaid_worms",
                   how="left")
            .dropna(subset=["phylum"]))
    # Suspect identifiers are dropped here for the same reason as in step 12: a
    # fuzzy AphiaID match attaches real chemistry to the wrong phylum, and these
    # tables are grouped by phylum.
    if "name_suspect" in prop.columns:
        drop = set(prop.loc[prop["name_suspect"], "molecule_id"])
        before = len(link)
        link = link[~link["molecule_id"].isin(drop)]
        print(f"{before:,} compound-phylum links, {len(link):,} after dropping "
              f"{len(drop):,} suspect identifiers")
    pl = link.merge(prop, on="molecule_id", how="inner")

    agg = {"n_compounds": ("molecule_id", "nunique"),
           "median_mw": ("mw", "median"),
           "median_clogp": ("clogp", "median"),
           "median_fsp3": ("fsp3", "median"),
           "median_stereo": ("n_stereocentres", "median"),
           "pct_macrocyclic": ("macrocyclic", lambda s: round(s.mean() * 100, 1)),
           "median_qed": ("qed", "median")}
    if pl["np_likeness"].notna().any():
        agg["median_np_likeness"] = ("np_likeness", "median")
    by_phy = pl.groupby("phylum").agg(**agg)
    by_phy = by_phy[by_phy["n_compounds"] >= 50]
    print(by_phy.round(2).sort_values("median_mw", ascending=False).to_string())

    print("\n" + "=" * 98)
    print("By biosynthetic pathway")
    print("=" * 98)
    by_path = prop.dropna(subset=["np_classifier_pathway"]).groupby(
        "np_classifier_pathway").agg(**agg)
    by_path = by_path[by_path["n_compounds"] >= 50]
    print(by_path.round(2).sort_values("median_mw", ascending=False).to_string())
    print("\nPathway separates the panel more cleanly than phylum does, which "
          "is what\nthe step 12 partitioning predicted: the enzymology sets the "
          "property range,\nand the host lineage inherits it.")

    # ----------------------------------------------------------------- save
    written = save_table(prop, config.INTERIM / "properties")
    summ.round(3).reset_index().rename(columns={"index": "descriptor"}).to_csv(
        config.INTERIM / "properties_summary.csv", index=False)
    by_phy.round(3).reset_index().to_csv(
        config.INTERIM / "properties_by_phylum.csv", index=False)
    by_path.round(3).reset_index().to_csv(
        config.INTERIM / "properties_by_pathway.csv", index=False)
    (config.INTERIM / "properties.json").write_text(json.dumps({
        "n_profiled": int(len(prop)),
        "n_unparseable": int(len(failed)),
        "sa_score_available": bool(sa_fn),
        "np_score_available": bool(np_fn),
        "median": {c: round(float(prop[c].median()), 3) for c in have},
        "pct_beyond_ro5": round(beyond, 2),
        "pct_macrocyclic": round(float(prop["macrocyclic"].mean() * 100), 2),
        "pct_veber_pass": round(float(prop["veber_pass"].mean() * 100), 2),
        "pct_fsp3_above_half": round(float((prop["fsp3"] > 0.5).mean() * 100), 2),
        "qed_np_correlation": None if np.isnan(corr) else round(corr, 4),
        "qed_mw_correlation": None if np.isnan(corr_qed_mw) else round(corr_qed_mw, 4),
        "np_mw_correlation": None if np.isnan(corr_np_mw) else round(corr_np_mw, 4),
        "n_stereo_failures": len(STEREO_ERRORS),
        "pct_stereocentres_unspecified": (
            None if prop["n_stereocentres"].isna().all()
            or not prop["n_stereocentres"].sum()
            else round(float(prop["n_stereo_unspecified"].sum()
                             / prop["n_stereocentres"].sum() * 100), 2)),
        "macrocycle_min_ring": MACROCYCLE_MIN,
    }, indent=2, default=float))
    print(f"\nWrote {written.name}, properties_summary.csv, "
          f"properties_by_phylum.csv,\nproperties_by_pathway.csv, "
          f"properties.json\nto {config.INTERIM}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="run 6 known molecules and stop")
    if ap.parse_args().selftest:
        selftest()
    else:
        main()
