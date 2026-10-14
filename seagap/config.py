"""Paths and build-time choices.

Layout assumed:
    /mnt/d/seagap/
        marinenp_v1_r20250608_csv/   the unzipped release
        seagap/                      this package
        scripts/                     runnable steps
        data/interim/                outputs
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "marinenp_v1_r20250608_csv"
INTERIM = ROOT / "data" / "interim"

TABLES = {
    "molecules": "molecules.csv",
    "properties": "properties.csv",
    "organisms": "organisms.csv",
    "molecule_organism": "molecule_organism.csv",
    "geo_locations": "geo_locations.csv",
    "geo_location_molecule": "geo_location_molecule.csv",
    "citations": "citations.csv",
    "citables": "citables.csv",
}

# How a molecule qualifies as marine. The release carries is_marine on both
# molecules and organisms, and they need not agree. 01_build.py reports all three
# counts; set this once you have seen them.
#   "molecule"  trust molecules.is_marine
#   "organism"  any linked organism has is_marine
#   "both"      intersection (conservative)
MARINE_RULE = "molecule"

# Drop rows molecules.csv already flags as redundant.
DROP_IS_DUPLICATE = True

# Which scope the chemical-space analysis runs on: "tier1", "tier2" or "all".
# Tier definitions live in scripts/04_scope.py; sizes are in data/interim/scope.json.
TIER = "tier1"

# Fingerprinting. Changing these invalidates data/interim/fps.npy.
FP_RADIUS = 2
FP_NBITS = 2048

# Similarity thresholds swept in step 05.
THRESHOLD_SWEEP = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

# Leiden resolution. Higher splits more finely.
LEIDEN_RESOLUTION = 1.0
LEIDEN_SEED = 0

# ---------------------------------------------------------------------------
# Operating point, fixed by the sweep in 06_granularity.py.
#
# Chosen on V-measure against np_classifier_class (540 categories), the only
# label level whose granularity is comparable to the partitions being scored.
# v_class across thresholds: .754 .777 .786 .780 .769 .753 -- an interior
# maximum at 0.55. V against superclass (72 categories) falls monotonically and
# cannot choose, for the reason demonstrated in that script.
#
# Resolution is near-inert: at modularity ~0.90 the graph is close to
# disconnected, so community count varies ~2% across resolution 0.05-2.0. The
# default stands.
#
# At 0.55 / 1.0: 5,795 communities, 866 with >=5 members, 79.3% of compounds
# inside them, largest component 3.2%, homogeneity 0.891, purity 0.924.
#
# These communities span many Murcko scaffolds (0.42 distinct scaffolds per
# compound), so they are STRUCTURAL FAMILIES, not scaffold families. Use that
# term throughout.
# ---------------------------------------------------------------------------
CHOSEN_THRESHOLD = 0.55
CHOSEN_RESOLUTION = 1.0

# Minimum members for a family to enter the bipartite analysis.
MIN_FAMILY = 5

# Taxonomic rank the gap analysis is reported at.
TAXON_RANK = "genus"

# Don't write an edge list larger than this (keeps disk sane at low thresholds).
MAX_EDGES_TO_SAVE = 60_000_000
