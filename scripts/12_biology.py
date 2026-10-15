"""Biological reading of the chemical space.

    python scripts/12_biology.py

Everything up to step 11 characterised the resource and the sampling. This asks
what the chemistry itself says about the organisms, using only data already in
hand.

1. Chemotaxonomic signal. Do related taxa make related chemistry? If chemical
   distance tracks taxonomic distance, metabolite profiles carry phylogenetic
   information. If it does not, the compounds attributed to a host may not be
   made by it.

2. Cross-phylum families. A structural family reported from both an invertebrate
   and a bacterium is a candidate for a symbiont being the true producer, which
   is the long-running question in sponge and tunicate natural products. Records
   with a suspect identifier are excluded, since a fuzzy AphiaID match manufactures
   exactly this pattern.

3. Halogenation. Marine natural products are distinctively halogenated, because
   seawater supplies bromide and iodide and marine organisms carry vanadium
   haloperoxidases and FADH2-dependent halogenases. Halogen content is readable
   straight off the SMILES, so this is a chemical trait that can be mapped onto
   the phylogeny without any extra data.

4. Biosynthetic partitioning. NPClassifier pathway assignments stand in for the
   enzymatic machinery: terpene synthases against PKS and NRPS. Which lineages
   specialise, and does that specialisation follow the phylogeny?

Writes biology.json plus chemotaxonomy.csv, symbiont_candidates.csv,
halogenation.csv and pathway_enrichment.csv.
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
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("rdkit is required: pip install rdkit")

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 38)
print("SCRIPT VERSION: 2026-10-14 biology\n")

RANK = config.TAXON_RANK
MIN_N = 10
MIN_FAM_FOR_DIST = 5     # a genus needs some breadth before Jaccard means anything
PERMS = 500
rng = np.random.default_rng(0)
bar = "=" * 98


def section(t):
    print(f"\n{bar}\n{t}\n{bar}")


# ------------------------------------------------------------------ load
work = load_table(config.INTERIM / "families")
famp = load_table(config.INTERIM / "family_profile").set_index("family")
tax = load_table(config.INTERIM / "organism_taxonomy")
org = pd.read_csv(config.RELEASE / config.TABLES["organisms"], low_memory=False)
mol_org = pd.read_csv(config.RELEASE / config.TABLES["molecule_organism"],
                      low_memory=False)

suspect = set()
if "name_suspect" in work.columns:
    suspect = set(work.loc[work["name_suspect"], "molecule_id"])
print(f"{len(work):,} compounds in families; {len(suspect):,} carry a suspect "
      "organism identifier")

link = (mol_org[["molecule_id", "organism_id"]].dropna().drop_duplicates()
        .merge(org[["id", "aphiaid_worms"]].rename(columns={"id": "organism_id"}),
               on="organism_id", how="left")
        .merge(tax[["aphiaid_worms", "genus", "family", "order", "class", "phylum",
                    "kingdom"]].rename(columns={"family": "tax_family",
                                                "class": "tax_class"}),
               on="aphiaid_worms", how="left")
        .merge(work[["molecule_id", "family", "canonical_smiles",
                     "np_classifier_pathway", "np_classifier_superclass"]],
               on="molecule_id", how="inner")
        .dropna(subset=[RANK]))
clean = link[~link["molecule_id"].isin(suspect)]
print(f"{len(link):,} links, {len(clean):,} after dropping suspect identifiers")

# ------------------------------------------------------------------ 1
section("1. Does chemistry follow phylogeny?")
prof = (clean.groupby(RANK)["family"].agg(lambda s: frozenset(s)))
size = clean.groupby(RANK)["molecule_id"].nunique()
keep = [g for g in prof.index
        if len(prof[g]) >= MIN_FAM_FOR_DIST and size[g] >= MIN_N]
print(f"{len(keep):,} genera with >={MIN_N} compounds in >={MIN_FAM_FOR_DIST} "
      "families")

meta = (clean[clean[RANK].isin(keep)].groupby(RANK)
        .agg(tax_family=("tax_family", lambda s: s.mode().iloc[0] if s.notna().any() else None),
             order=("order", lambda s: s.mode().iloc[0] if s.notna().any() else None),
             tax_class=("tax_class", lambda s: s.mode().iloc[0] if s.notna().any() else None),
             phylum=("phylum", lambda s: s.mode().iloc[0] if s.notna().any() else None)))

sets = {g: prof[g] for g in keep}
LEVELS = ["same family", "same order", "same class", "same phylum",
          "different phylum"]

# Taxonomic level per pair, vectorised. Doing this with .loc inside a 400k-pair
# Python loop takes minutes; integer codes plus numpy broadcasting takes a moment.
def codes(col):
    v = meta[col].reindex(keep)
    u = {x: i for i, x in enumerate(sorted(v.dropna().unique()))}
    return np.array([u.get(x, -1 - i) for i, x in enumerate(v)], dtype=np.int64)


cf, co, cc, cp = (codes("tax_family"), codes("order"),
                  codes("tax_class"), codes("phylum"))
iu, ju = np.triu_indices(len(keep), k=1)
pairs_lv = np.full(len(iu), 4, dtype=np.int8)
pairs_lv[cp[iu] == cp[ju]] = 3
pairs_lv[cc[iu] == cc[ju]] = 2
pairs_lv[co[iu] == co[ju]] = 1
pairs_lv[cf[iu] == cf[ju]] = 0

# Jaccard distance between family profiles, as a sparse boolean matrix product.
fam_ids = sorted({f for g in keep for f in sets[g]})
fidx = {f: i for i, f in enumerate(fam_ids)}
M = np.zeros((len(keep), len(fam_ids)), dtype=bool)
for r, g in enumerate(keep):
    for f in sets[g]:
        M[r, fidx[f]] = True
Mi = M.astype(np.float64)
sz = M.sum(1).astype(np.int32)
inter = (Mi @ Mi.T)
uni = sz[:, None] + sz[None, :] - inter
with np.errstate(divide="ignore", invalid="ignore"):
    D_raw = np.where(uni > 0, 1.0 - inter / uni, 1.0)

# A handful of structural families are reported from almost every phylum: sterols,
# fatty acids, carotenoids, sphingolipids. Two unrelated genera both making sterols
# says nothing about their relatedness, but on a plain Jaccard it counts the same
# as sharing a rare family, so each family is also weighted by -log(prevalence).
#
# This is a robustness check, not a repair, and the distinction matters. In
# simulation the weighting PRESERVES the absolute gap between related and
# unrelated taxa (+0.119 plain against +0.120 weighted) while raising the
# baseline, so the relative gap actually shrinks. On data with no clade structure
# both metrics return a gap of ~0.0001, so neither invents a signal. If both
# columns show a small gap, the weakness is in the data and not in the metric.
#
# Reference point from that simulation: taxa with genuine clade-specific
# chemistry gave an absolute gap near 0.12. Compare whatever this run reports.
prev = M.mean(0)
w = -np.log(np.clip(prev, 1e-9, 1.0))
Mw = Mi * w
inter_w = Mw @ Mi.T                       # weight of families shared
tot_w = (Mi * w).sum(1)
uni_w = tot_w[:, None] + tot_w[None, :] - inter_w
with np.errstate(divide="ignore", invalid="ignore"):
    D_w = np.where(uni_w > 0, 1.0 - inter_w / uni_w, 1.0)

pairs_d = D_raw[iu, ju]
pairs_dw = D_w[iu, ju]
print(f"{len(pairs_d):,} genus pairs compared")
print(f"family prevalence across genera: median {np.median(prev) * 100:.2f}%, "
      f"max {prev.max() * 100:.1f}% "
      f"({int((prev > 0.5).sum())} families are in over half of them)\n")

tbl = pd.DataFrame({
    "n_pairs": [int((pairs_lv == k).sum()) for k in range(5)],
    "jaccard": [round(float(pairs_d[pairs_lv == k].mean()), 4)
                if (pairs_lv == k).any() else None for k in range(5)],
    "rarity_weighted": [round(float(pairs_dw[pairs_lv == k].mean()), 4)
                        if (pairs_lv == k).any() else None for k in range(5)],
    "pct_sharing_a_family": [round(float((pairs_d[pairs_lv == k] < 1).mean() * 100), 1)
                             if (pairs_lv == k).any() else None for k in range(5)],
}, index=LEVELS)
print(tbl.to_string())

mono_raw = all(tbl["jaccard"].iloc[k] <= tbl["jaccard"].iloc[k + 1] + 1e-9
               for k in range(4) if tbl["jaccard"].iloc[k] is not None
               and tbl["jaccard"].iloc[k + 1] is not None)
mono_w = all(tbl["rarity_weighted"].iloc[k] <= tbl["rarity_weighted"].iloc[k + 1] + 1e-9
             for k in range(4) if tbl["rarity_weighted"].iloc[k] is not None
             and tbl["rarity_weighted"].iloc[k + 1] is not None)
print(f"\nmonotone with taxonomic distance?  jaccard: {mono_raw}   "
      f"rarity-weighted: {mono_w}")
print("A signal that is not monotone across ranks is not a phylogenetic signal, "
      "whatever\nits p value. Both columns are reported so the reader can see "
      "which one, if either,\nbehaves.")

obs_gap = (pairs_dw[pairs_lv == 4].mean() - pairs_dw[pairs_lv == 0].mean()
           if (pairs_lv == 0).any() else float("nan"))
null = np.empty(PERMS)
m4, m0 = pairs_lv == 4, pairs_lv == 0
for p in range(PERMS):
    sh = rng.permutation(pairs_dw)
    null[p] = sh[m4].mean() - sh[m0].mean()
pval = float((np.abs(null) >= abs(obs_gap)).mean())
print(f"\nrarity-weighted gap, different phylum minus same family: {obs_gap:+.4f}")
print(f"relative to the same-family baseline: "
      f"{obs_gap / max(tbl['rarity_weighted'].iloc[0], 1e-9) * 100:+.1f}%")
print(f"permutation test over {PERMS} shuffles: p = {pval:.4f}  "
      f"(null spread {null.std():.4f})")
print("\nRead the effect, not the p value: at this many pairs almost any gap is "
      "'significant'.\nA gap of a few percent against a baseline near 0.9 means "
      "two genera in the same\nfamily are barely more chemically alike than two "
      "from different phyla. Two further\ncautions: genera in one order are often "
      "worked by one laboratory on one assay,\nwhich manufactures similarity that "
      "is not biology; and a weak host signal is what\nwould be expected if much "
      "of this chemistry is not made by the host at all.")

# ------------------------------------------------------------------ 2
section("2. Cross-phylum families: who actually makes the compound?")
MICROBE_PHYLA = {"Actinobacteria", "Proteobacteria", "Cyanobacteria", "Ascomycota",
                 "Basidiomycota", "Firmicutes", "Bacteroidetes", "Verrucomicrobia",
                 "Planctomycetota", "Euryarchaeota", "Crenarchaeota", "Chloroflexi"}
HOST_PHYLA = {"Porifera", "Cnidaria", "Chordata", "Mollusca", "Bryozoa",
              "Echinodermata", "Annelida", "Arthropoda"}

fp = (clean.dropna(subset=["phylum"]).groupby("family")["phylum"]
      .agg(lambda s: sorted(set(s))))
fam_n = clean.groupby("family")["molecule_id"].nunique()
cross = pd.DataFrame({"phyla": fp, "n_phyla": fp.map(len), "n_compounds": fam_n})
cross = cross.dropna(subset=["n_phyla"])
cross["has_microbe"] = cross["phyla"].map(lambda v: bool(set(v) & MICROBE_PHYLA))
cross["has_host"] = cross["phyla"].map(lambda v: bool(set(v) & HOST_PHYLA))
cross["candidate"] = cross["has_microbe"] & cross["has_host"]

print(f"{len(cross):,} families with a resolved phylum")
print(f"{int((cross['n_phyla'] == 1).sum()):,} confined to one phylum, "
      f"{int((cross['n_phyla'] > 1).sum()):,} span more than one")
print(f"{int(cross['candidate'].sum()):,} span both a microbial phylum and an "
      "animal phylum")
print(f"those carry {int(cross.loc[cross['candidate'], 'n_compounds'].sum()):,} "
      "compounds\n")

# A family spanning 18 phyla is not evidence of symbiosis, it is a sterol or a
# fatty acid. The informative case is a family confined to a few taxa that still
# crosses the animal-microbe line, so candidates are restricted by span and the
# ubiquitous ones are reported separately rather than ranked as discoveries.
MAX_SPAN = 4
ubiq = cross[cross["candidate"] & (cross["n_phyla"] > MAX_SPAN)]
print(f"of those, {len(ubiq):,} span more than {MAX_SPAN} phyla and carry "
      f"{int(ubiq['n_compounds'].sum()):,} compounds.")
if len(ubiq):
    top_u = ubiq.sort_values("n_compounds", ascending=False).head(6)
    print("  widest, with their chemistry:")
    for f, r in top_u.iterrows():
        print(f"    family {f:<6} {int(r.n_phyla):>2} phyla  "
              f"{int(r.n_compounds):>5,} compounds  "
              f"{famp['superclass'].get(f, '?')}")
    print("  Sterols, fatty acids, carotenoids and sugars occur in everything "
          "with cells.\n  They are excluded, not ranked.")

cand = cross[cross["candidate"] & (cross["n_phyla"] <= MAX_SPAN)].copy()
print(f"\n{len(cand):,} families cross the animal-microbe line within "
      f"{MAX_SPAN} phyla or fewer")
if len(cand):
    cand["microbes"] = cand["phyla"].map(
        lambda v: ", ".join(sorted(set(v) & MICROBE_PHYLA)))
    cand["hosts"] = cand["phyla"].map(
        lambda v: ", ".join(sorted(set(v) & HOST_PHYLA)))
    cand["pathway"] = famp["pathway"].reindex(cand.index)
    cand["superclass"] = famp["superclass"].reindex(cand.index)
    cand["n_genera"] = clean.groupby("family")["genus"].nunique().reindex(cand.index)
    print(f"\nRanked by specificity, narrowest span first "
          f"(top 25 of {len(cand):,}):")
    print(cand.sort_values(["n_phyla", "n_compounds"], ascending=[True, False])
          .head(25)[["n_phyla", "n_compounds", "n_genera", "hosts", "microbes",
                     "superclass"]].to_string())

    tight = cand[cand["n_phyla"] == 2]
    print(f"\n{len(tight):,} families sit in exactly one animal phylum and one "
          "microbial phylum.")
    print("Those are the cleanest candidates: a single host lineage and a single "
          "candidate\nproducer, with nothing else to explain the co-occurrence.")
    if len(tight):
        print(tight.sort_values("n_compounds", ascending=False)
              .head(15)[["n_compounds", "n_genera", "hosts", "microbes",
                         "superclass"]].to_string())

    print("\nPairings, counted over the restricted set:")
    pc = (cand.groupby(["hosts", "microbes"])["n_compounds"]
          .agg(["size", "sum"]).sort_values("sum", ascending=False).head(12))
    pc.columns = ["n_families", "n_compounds"]
    print(pc.to_string())

# ------------------------------------------------------------------ 3
section("3. Halogenation, a marine chemical trait")
smi = work.drop_duplicates("molecule_id").set_index("molecule_id")["canonical_smiles"]
hal = {}
for mid, s in smi.items():
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    if m is None:
        continue
    c = {"Br": 0, "Cl": 0, "I": 0, "F": 0}
    for at in m.GetAtoms():
        sym = at.GetSymbol()
        if sym in c:
            c[sym] += 1
    hal[mid] = c
H = pd.DataFrame(hal).T
H["total"] = H.sum(axis=1)
H["halogenated"] = H["total"] > 0
print(f"{len(H):,} structures parsed, "
      f"{int(H['halogenated'].sum()):,} halogenated "
      f"({H['halogenated'].mean() * 100:.1f}%)")
print("\nhalogen atoms present, by element:")
print(pd.Series({k: int((H[k] > 0).sum()) for k in ["Br", "Cl", "I", "F"]})
      .sort_values(ascending=False).to_string())
print("\nBromine over chlorine is the marine signature: seawater is bromide-rich "
      "relative\nto the terrestrial environment, and vanadium bromoperoxidases "
      "are widespread in\nmarine algae.")

cl = clean.drop_duplicates(subset=["molecule_id", RANK])
cl = cl.merge(H[["halogenated", "Br", "Cl", "total"]],
              left_on="molecule_id", right_index=True, how="inner")
by_phy = (cl.dropna(subset=["phylum"]).groupby("phylum")
          .agg(n_compounds=("molecule_id", "nunique"),
               pct_halogenated=("halogenated", lambda s: round(s.mean() * 100, 1)),
               mean_halogens=("total", lambda s: round(s.mean(), 2)),
               br_over_cl=("Br", "sum")))
by_phy["cl_sum"] = cl.dropna(subset=["phylum"]).groupby("phylum")["Cl"].sum()
by_phy["br_cl_ratio"] = (by_phy["br_over_cl"]
                         / by_phy["cl_sum"].replace(0, np.nan)).round(2)
by_phy = by_phy[by_phy["n_compounds"] >= 50].drop(columns=["br_over_cl", "cl_sum"])
print("\nby phylum (50+ compounds), most halogenated first:")
print(by_phy.sort_values("pct_halogenated", ascending=False).to_string())

by_gen = (cl.groupby(RANK)
          .agg(phylum=("phylum", lambda s: s.mode().iloc[0] if s.notna().any() else None),
               n_compounds=("molecule_id", "nunique"),
               pct_halogenated=("halogenated", lambda s: round(s.mean() * 100, 1)),
               mean_halogens=("total", lambda s: round(s.mean(), 2))))
by_gen = by_gen[by_gen["n_compounds"] >= 20]
print(f"\ntop 20 halogenating genera ({len(by_gen):,} with 20+ compounds):")
print(by_gen.sort_values("pct_halogenated", ascending=False).head(20).to_string())

# ------------------------------------------------------------------ 4
section("4. Biosynthetic partitioning by lineage")
ct = pd.crosstab(clean.dropna(subset=["phylum"])["phylum"],
                 clean.dropna(subset=["phylum"])["np_classifier_pathway"])
ct = ct[ct.sum(axis=1) >= 100]
exp = np.outer(ct.sum(axis=1), ct.sum(axis=0)) / ct.values.sum()
resid = (ct.values - exp) / np.sqrt(exp)
R = pd.DataFrame(resid.round(1), index=ct.index, columns=ct.columns)
print("standardised residuals, observed minus expected under independence:")
print("(positive = this lineage makes more of this chemistry than its overall "
      "share predicts)\n")
print(R.to_string())

chi2 = float((resid ** 2).sum())
dof = (ct.shape[0] - 1) * (ct.shape[1] - 1)
print(f"\nchi-square {chi2:,.0f} on {dof} df")
print("\nStrong partitioning means the biosynthetic machinery differs by lineage "
      "in a way\nthat a genome-mining follow-up could test directly: terpene "
      "synthase counts\nagainst PKS and NRPS cluster counts in the corresponding "
      "genomes.")

print("\nmost distinctive pathway per lineage:")
for p in R.index:
    best = R.loc[p].idxmax()
    print(f"  {p:18s} {best:34s} residual {R.loc[p, best]:+6.1f}  "
          f"({ct.loc[p, best]:,} of {ct.loc[p].sum():,} compounds)")

# ------------------------------------------------------------------ save
tbl.reset_index().rename(columns={"index": "level"}).to_csv(
    config.INTERIM / "chemotaxonomy.csv", index=False)
if len(cand):
    cand.drop(columns=["phyla"]).reset_index().to_csv(
        config.INTERIM / "symbiont_candidates.csv", index=False)
by_gen.reset_index().to_csv(config.INTERIM / "halogenation.csv", index=False)
R.reset_index().to_csv(config.INTERIM / "pathway_enrichment.csv", index=False)
(config.INTERIM / "biology.json").write_text(json.dumps({
    "n_genera_compared": len(keep),
    "n_pairs": int(len(pairs_d)),
    "chemotaxonomy_jaccard": {LEVELS[k]: tbl["jaccard"].iloc[k] for k in range(5)},
    "chemotaxonomy_weighted": {LEVELS[k]: tbl["rarity_weighted"].iloc[k]
                               for k in range(5)},
    "monotone_jaccard": bool(mono_raw),
    "monotone_weighted": bool(mono_w),
    "distance_gap": round(float(obs_gap), 4),
    "permutation_p": pval,
    "n_families_one_phylum": int((cross["n_phyla"] == 1).sum()),
    "n_families_multi_phylum": int((cross["n_phyla"] > 1).sum()),
    "n_crossing_animal_microbe": int(cross["candidate"].sum()),
    "max_span_for_candidate": MAX_SPAN,
    "n_specific_candidates": int(len(cand)),
    "n_two_phylum_candidates": int((cand["n_phyla"] == 2).sum()) if len(cand) else 0,
    "pct_halogenated": round(float(H["halogenated"].mean() * 100), 2),
    "halogen_counts": {k: int((H[k] > 0).sum()) for k in ["Br", "Cl", "I", "F"]},
    "pathway_chi2": round(chi2, 1), "pathway_dof": int(dof),
}, indent=2, default=float))
print(f"\nWrote chemotaxonomy.csv, symbiont_candidates.csv, halogenation.csv, "
      f"pathway_enrichment.csv, biology.json\nto {config.INTERIM}")
