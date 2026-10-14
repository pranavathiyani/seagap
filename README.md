# SEAGAP

Mapping under-explored chemical space in marine natural products.

SEAGAP groups the [MarineNP](https://zenodo.org/records/15641716) release into
structural families, resolves every source organism against
[WoRMS](https://www.marinespecies.org/), and asks how chemical breadth scales
with the effort spent on a taxon. The output is an interactive viewer and a set
of tables: which genera are chemically saturated, which productive lineages
remain unexamined, and where the release's own provenance cannot be trusted.

**Viewer:** https://pranavathiyani.github.io/seagap/

## What it finds

**Diminishing returns are quantifiable.** Structural families grow as about
n^0.69 in the number of compounds reported from a taxon, so a genus studied ten
times as hard yields roughly five times the families, not ten.

**No marine taxon is exceptionally broad.** Deviations from that curve are
one-sided. Genera can be far more chemically repetitive than their sampling
predicts, but nothing is far more diverse, because a genus produces
biosynthetically related metabolites. Nothing reaches the positive significance
threshold at genus or order level. A ranking of "promising" taxa by chemical
breadth is therefore not supportable from this data, and SEAGAP does not publish
one.

**Some genera are saturated.** Forty-five sit below the curve at z < -1.96,
including heavily worked groups where further screening has poor expected return
per compound isolated. Roughly 94% of the variation in breadth is real rather
than sampling noise.

**Almost nothing has been examined.** The genus-by-family matrix is over 99%
empty and the median genus contributes three structural families. The frontier is
breadth across taxa, not depth within a few.

**The release's organism identifiers are unreliable.** MarineNP reconciles
organism names to WoRMS by fuzzy matching without validation. About 6% of
organism records carry an AphiaID belonging to a different genus, affecting
roughly 9% of compounds. *Thymus vulgaris*, garden thyme, resolves to *Thynnus
vulgaris*, a tuna. *Croton* to *Chiton*, *Magnolia* to *Mangelia*, *Ficus* to
*Fucus*. Every property inherited from those identifiers, phylum and habitat
flags included, is wrong for those records.

## Install

```bash
conda create -n seagap python=3.12 -y
conda activate seagap
pip install pandas numpy pyarrow scipy scikit-learn matplotlib requests \
            rdkit python-igraph leidenalg
python scripts/00_check_env.py
```

`environment.yml` does the same through conda-forge if you prefer.

## Data

Not tracked here. Download the MarineNP v1 CSV release and unzip it so that
`marinenp_v1_r20250608_csv/` sits beside `scripts/`. Record the version and
access date in `data/raw/SOURCE.txt`, and do not update the snapshot
mid-analysis.

## Run

```bash
python scripts/00_check_env.py      # verify the environment first
python scripts/01_build.py          # join the 9 release tables, audit coverage
python scripts/01b_inspect.py       # read-only: interrogate the marine claim
python scripts/02_taxonomy.py       # fetch WoRMS records (~30 min, resumable)
python scripts/03b_review.py        # read-only: audit name matching
python scripts/04_scope.py          # fix the scope tiers, write flags
python scripts/05_chemspace.py      # fingerprints, similarity, threshold sweep
python scripts/06_granularity.py    # choose threshold and resolution on evidence
python scripts/07_bias.py           # bipartite network, concentration, breadth
python scripts/08_rank.py           # diagnose confounds, rank on rarity
python scripts/09_uncertainty.py    # heteroscedasticity, variance decomposition
python scripts/10_saturation.py     # asymmetry, saturation, coverage frontier
python scripts/11_export.py         # build docs/index.html
```

Step 02 is the only slow one and caches to `data/interim/worms_cache.json`, so it
is safe to interrupt and re-run. `03_tiers.py` is superseded by `04_scope.py` and
kept only for reference.

## Hosting the viewer

`scripts/11_export.py` writes `docs/index.html`: one self-contained file with the
data inlined, no CDN, no build step, no server. Open it from disk, or publish it:

1. Push the repo to GitHub.
2. **Settings → Pages → Build and deployment**, source **Deploy from a branch**.
3. Branch `main`, folder **`/docs`**. Save.
4. It appears at `https://<user>.github.io/seagap/` within a minute or two.

The page has three tabs. **Overview** carries the accumulation curve and the
deviation distribution. **Network** draws the genus-by-family bipartite graph one
taxonomic order at a time, capped to the busiest nodes and reporting what it
leaves out. **Tables** holds the saturated genera, the coverage frontier and the
provenance errors, all sortable.

To re-skin or edit the page, change `viewer/template.html` and re-run step 11.
`docs/index.html` is generated and should not be hand-edited.

## Method notes

A few choices worth knowing before reading the numbers.

**Scope.** The release flags every record as marine, so that flag carries no
information. Tier 1, the scope used throughout, keeps compounds whose source is a
marine eukaryote or marine plant at species rank with a strictly marine WoRMS
record and a name that matches its identifier. Seagrasses, mangroves and coastal
halophytes are kept; they are genuinely intertidal.

**Families.** Leiden communities on a Tanimoto graph of Morgan fingerprints. The
threshold was chosen by agreement with NPClassifier class assignments, which the
graph never sees, rather than by eye. These communities span many Murcko
scaffolds, so they are called *structural families*, not scaffold families.

**The null.** Breadth is measured against other taxa at comparable effort, fitted
as a power law. An earlier version compared each taxon against a random draw from
the whole compound pool; that is wrong, because real taxa produce related
metabolites, and in simulation it correlated -0.91 with sampling effort and +0.13
with true breadth. It measured effort.

**Standardisation.** Residual scatter shrinks as sample size grows, so residuals
are divided by the sd expected at their own n. Using one global sd inflates small
taxa: in simulation, with every taxon given identical true breadth, 19.5% of the
smallest-n taxa scored z > 1 against 1.3% of the largest.

**Rejected.** Per-taxon bootstrap confidence intervals. Richness is not a smooth
functional, and measured coverage was 22% (percentile), 21% (bias-corrected) and
61% (bootstrap sd) against a nominal 95%. Resampling is used only to size the
sampling variance, which is a valid use of it.

## Layout

```
marinenp_v1_r20250608_csv/   source release, not tracked
seagap/                      shared code: config, tables, curve fitting
scripts/                     numbered steps, run in order
viewer/template.html         viewer source; edit this, not docs/
docs/index.html              generated viewer, served by GitHub Pages
data/interim/                outputs, not tracked
```

## Licence

MIT for the code. MarineNP and WoRMS retain their own terms.
