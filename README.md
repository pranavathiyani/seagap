# SEAGAP

Mapping under-explored chemical space in marine natural products.

[![viewer](https://img.shields.io/badge/viewer-github.io-2a78d6)](https://pranavathiyani.github.io/seagap/)
[![licence](https://img.shields.io/badge/code-MIT-informational)](LICENSE)
[![data](https://img.shields.io/badge/data-MarineNP%20v1-lightgrey)](https://zenodo.org/records/15641716)
[![taxonomy](https://img.shields.io/badge/taxonomy-WoRMS-lightgrey)](https://www.marinespecies.org/)

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

**Half of marine structural skeletons appear nowhere else in COCONUT.** Of 6,331
Bemis-Murcko scaffolds in the marine set, 3,352 (52.9%) are absent from the
611,721 non-marine COCONUT compounds, and the scaffold Jaccard between the two
libraries is 0.025. The most populated marine-only skeletons trace to the
lineages the biosynthetic analysis independently flags: three of the top five,
127 compounds, come from *Briareum*, the gorgonian genus that gives briarane
diterpenes their name.

**Marine chemistry differs on two axes, not five.** Against the non-marine
COCONUT remainder, marine compounds score higher on NP-likeness (Cliff's delta
+0.471) and are 1.8 times as likely to be macrocyclic. Mass, rotatable bonds,
polar surface area and Rule-of-Five violations are indistinguishable: 31.6% of
marine compounds carry two or more Ro5 violations against 31.8% of the baseline,
so beyond-Rule-of-Five chemistry is a property of natural products rather than
of marine ones. Stratifying by aromatic-ring count, NP-likeness and
macrocyclicity hold, synthetic accessibility and stereocentre count weaken, and
the apparent sp3 enrichment reverses sign. SEAGAP therefore does not claim
marine natural products are sp3-rich.

**The release's organism identifiers are unreliable.** MarineNP reconciles
organism names to WoRMS by fuzzy matching without validation. About 6% of
organism records carry an AphiaID belonging to a different genus, affecting
roughly 9% of compounds. *Thymus vulgaris*, garden thyme, resolves to *Thynnus
vulgaris*, a tuna. *Croton* to *Chiton*, *Magnolia* to *Mangelia*, *Ficus* to
*Fucus*. Every property inherited from those identifiers, phylum and habitat
flags included, is wrong for those records. Two of the release's columns are
also empty throughout: `murko_framework` and `sugar_free_smiles`.

## Biology

Beyond the resource audit, the chemical space answers questions about the
organisms. `scripts/12_biology.py` asks four.

**Chemistry does not follow phylogeny.** Jaccard distance over structural-family
profiles is compared across taxonomic ranks and tested by permutation. Two genera
in the same family are only 3.9% more chemically alike than two from different
phyla, and the trend is not monotone across ranks under either the raw or the
rarity-weighted metric. A non-monotone difference is not a phylogenetic signal
whatever its p value, and at 118,828 pairs the permutation p is uninformative.
Two readings follow. Genera in one order are often worked by one laboratory on
one assay, which manufactures similarity that is not biology. And a weak host
signal is what would be expected if much of this chemistry is not made by the
host at all, which is the next question.

**Some compounds are probably made by symbionts.** Structural families reported
from both an animal phylum and a microbial phylum are candidates for a symbiont
being the true producer. Records with a suspect identifier are excluded first,
since a fuzzy AphiaID match manufactures exactly this pattern, and families
spanning more than four phyla are excluded as ubiquitous primary metabolism —
sterols and fatty acids occur in everything with cells. That leaves 48 specific
families, of which 10 sit in exactly one animal phylum and one microbial phylum.
One of those, family 159, is Chordata plus Cyanobacteria, 29 compounds,
oligopeptides: the patellamide system, where the producer is the obligate
ascidian symbiont *Prochloron didemni* rather than the host, established by
sequencing the biosynthetic genes and expressing the pathway in *E. coli*
(Schmidt et al., PNAS 2005, [doi:10.1073/pnas.0501424102](https://doi.org/10.1073/pnas.0501424102)).
Recovering a known case without being told it exists is the validation the rest
of the list needed. The strongest untested candidate is family 62, Annelida plus
Actinobacteria, 58 polycyclic aromatic polyketides. These are hypotheses to
resolve experimentally, not attributions.

**Halogenation separates two enzyme families.** Halogen content reads straight off
the SMILES, so a marine-specific trait maps onto the phylogeny with no extra data.
11.1% of structures are halogenated, bromine over chlorine overall (2,347 to
1,357), which is the marine signature: seawater is bromide-rich and marine algae
carry vanadium bromoperoxidases. The Br/Cl ratio splits the tree. Rhodophyta 2.16
and Bryozoa 8.54 against Cyanobacteria 0.05, Actinobacteria 0.20 and Cnidaria
0.24 — algal bromoperoxidase against the chlorine-preferring flavin-dependent
halogenases of bacteria. Vertebrate rows need manual inspection: marine
vertebrates bioaccumulate persistent organohalogens, so a high percentage there
may be pollutant residues rather than natural products.

**Biosynthetic machinery is strongly partitioned by lineage.** NPClassifier
pathways stand in for enzyme classes — terpene synthases against PKS and NRPS.
Standardised residuals reach +22.9 for Cnidaria terpenoids, +24.4 for Myzozoa
polyketides, +21.6 for Porifera alkaloids and +20.6 for Proteobacteria shikimates,
on a chi-square of 5,915 with 102 df. This is the result with the clearest
experimental follow-up: the predictions are testable by counting biosynthetic
gene clusters in the corresponding genomes. It also cross-checks the symbiont
list, since Annelida's single most distinctive pathway is polyketides at +10.8,
independently of the clustering that produced family 62.

## Chemical space

`scripts/13_properties.py` computes a physicochemical panel for every compound.
Everything in it is calculated from the structure by a documented function: size,
lipophilicity, polarity, saturation, stereochemistry, ring systems, plus QED,
synthetic accessibility and NP-likeness. No trained ADMET model is involved, and
mixing predictions into that table is how the whole panel stops being trusted.

Describing a library is not a claim about it. `scripts/14_baseline.py` computes
the same panel on the non-marine COCONUT remainder. MarineNP derives from
COCONUT, so the remainder is a control that shares its parent collection,
curation and classifier, leaving habitat as the variable. Compounds are matched
out on the InChIKey connectivity block rather than the full key, so a structure
recorded with different stereochemistry or salt form in the two collections
still matches. Effects are Cliff's delta: at 29,154 against 50,000 every p value
is zero and reporting one would mean nothing.

`scripts/15_axis.py` runs the two checks that result needs. Scaffolds are
recomputed over all 611,721 non-marine compounds rather than a sample, because
scaffold discovery is a rarefaction problem and sampling inflates the
marine-only count — the figure fell from 76.1% to 52.9% when the sample was
replaced by the full population. Effects are then recomputed inside strata of
aromatic-ring count, which is what separates a finding from a composition
artefact.

Two cautions are recorded in the output rather than left implicit. The COCONUT
release available now postdates the MarineNP snapshot, so compounds added since
sit in the control and some are marine, which pushes the two libraries toward
looking more alike and works against every difference reported. And the
scaffold count depends on the scaffold definition: RDKit keeps exocyclic
double-bonded atoms, COCONUT's own column strips them, so a coarser definition
would merge some marine skeletons with baseline ones.

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

Neither dataset is tracked here. Download the MarineNP v1 CSV release and unzip
it so that `marinenp_v1_r20250608_csv/` sits beside `scripts/`.

Steps 14 and 15 need the comparison baseline. Download `coconut_csv-*.zip` from
[the COCONUT download page](https://coconut.naturalproducts.net/download),
roughly 200 MB zipped and CC0, and unzip it into `data/coconut/`.

Record both versions and access dates in `data/raw/SOURCE.txt` and do not update
either snapshot mid-analysis. The two releases do not coincide in date, and that
gap belongs in the methods of anything built on them.

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
python scripts/12_biology.py        # chemotaxonomy, symbionts, halogens, pathways
python scripts/13_properties.py     # physicochemical panel (--selftest first)
python scripts/13b_stereo.py        # read-only: audit stereo perception
python scripts/14_baseline.py       # same panel on the non-marine COCONUT remainder
python scripts/15_axis.py           # full-population scaffolds, stratified effects
python scripts/16_browser.py        # docs/compounds.js for the Compounds tab
```

Step 02 is the slowest and caches to `data/interim/worms_cache.json`, so it is
safe to interrupt and re-run. Steps 14 and 15 take several minutes each, mostly
reading a 700 MB CSV and computing scaffolds across 611,721 structures.
`03_tiers.py` is superseded by `04_scope.py` and kept only for reference.

Run `python scripts/13_properties.py --selftest` before the full pass. It
profiles six molecules with known character and prints what each should look
like, so a wrong descriptor call surfaces on six compounds rather than after
29,154.

## Hosting the viewer

`scripts/11_export.py` writes `docs/index.html`: one self-contained file with the
data inlined, no CDN, no build step, no server. Open it from disk, or publish it:

1. Push the repo to GitHub.
2. **Settings → Pages → Build and deployment**, source **Deploy from a branch**.
3. Branch `main`, folder **`/docs`**. Save.
4. It appears at `https://<user>.github.io/seagap/` within a minute or two.

The page has four tabs. **Overview** carries the accumulation curve and the
deviation distribution. **Compounds** is a filterable table of all 29,154
compounds with their computed panel. **Network** draws the genus-by-family
bipartite graph one taxonomic order at a time, capped to the busiest nodes and
reporting what it leaves out. **Tables** holds the saturated genera, the coverage
frontier and the provenance errors, all sortable.

The Compounds tab reads `docs/compounds.js`, written by step 16 and loaded
through a script tag rather than inlined or fetched. Inlining would make every
tab pay the parse cost for 29,154 rows, and `fetch` against a `file://` URL is
blocked by CORS, which would break opening the viewer from disk. Only the
visible rows are in the DOM, so scrolling the full table stays responsive.

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

**Scaffolds.** Computed with RDKit on both sides rather than read from either
release's stored column. MarineNP's `murko_framework` is empty, and COCONUT's
`murcko_framework` uses a different definition: it strips exocyclic
double-bonded atoms, so coumarin reduces to its ring system without the
carbonyl, where RDKit keeps it. Agreement between COCONUT's column and RDKit is
31.6% after re-canonicalising their strings, which is a definitional difference
rather than a toolkit one. Stereochemistry is stripped before comparison so the
two libraries match on skeleton rather than configuration.

**Stereocentres.** Perceived with `AssignStereochemistry(...,
flagPossibleStereoCenters=True)`. Without that argument both
`CalcNumAtomStereoCenters` and `CalcNumUnspecifiedAtomStereoCenters` return zero
rather than raising, which reports "no stereocentres" for an entire library.
`CalcNumAtomStereoCenters` returns the total, so specified centres are the
difference against the unspecified count, not a separate call. 88.1% of
compounds carry at least one centre, median five, and 7.0% of centres are
unspecified.

**Effect sizes, not p values.** Library comparisons run at 29,154 against
50,000, where every p value is zero. Cliff's delta is used for continuous
descriptors because the distributions are skewed and heavy-tailed, and a risk
ratio for binary traits, where delta is bounded by the base rate and scored
macrocyclicity at 9.4% against 5.2% as negligible.

**Withdrawn.** The claim that marine natural products are sp3-enriched. The
unstratified Cliff's delta is +0.199, but every aromatic-ring stratum points the
other way and the pooled within-stratum estimate is −0.074. The difference comes
from the two libraries having different aromatic-ring composition, not from
saturation itself.

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
data/coconut/                COCONUT CSV for the baseline, not tracked
seagap/                      shared code: config, tables, curve fitting, descriptors
scripts/                     numbered steps, run in order
viewer/template.html         viewer source; edit this, not docs/
docs/index.html              generated viewer, served by GitHub Pages
docs/compounds.js            generated compound payload for the Compounds tab
data/interim/                outputs, not tracked
```

The descriptor panel lives in `seagap/descriptors.py` and is imported by both
step 13 and step 14. Two copies would let one edit to a cutoff invalidate every
comparison in the project without anything failing.

## Related resources

- [MarineNP](https://zenodo.org/records/15641716) — the source release
- [WoRMS](https://www.marinespecies.org/) — marine taxonomy and habitat flags
- [NPClassifier](https://npclassifier.gnps2.org/) — biosynthetic classification
- [CMNPD](https://www.cmnpd.org/) — an alternative open marine NP database
- [Natural Products Atlas](https://www.npatlas.org/) — microbial natural products
- [COCONUT](https://coconut.naturalproducts.net/) — the collection MarineNP derives from

## Citing

If SEAGAP is useful, cite the MarineNP release and WoRMS alongside this
repository. A manuscript is not yet available.

**Topics:** `marine-natural-products` `cheminformatics` `chemical-space`
`bioprospecting` `marine-biology` `natural-products` `rdkit` `worms`
`network-analysis` `open-science` `fair-data` `bioinformatics`

## Licence

MIT for the code. MarineNP and WoRMS retain their own terms.
