"""Fingerprint the chosen tier, sweep similarity thresholds, pick one on evidence.

    python scripts/05_chemspace.py

The threshold that defines "similar enough to be the same scaffold family" is
usually chosen by eye. The release already carries NPClassifier assignments, which
are an independent chemical classification, so the choice can be measured instead:
run Leiden at each threshold and score the resulting partition against NPClassifier
with adjusted mutual information. The threshold that best recovers an independent
classification is the defensible one.

Writes, per threshold, data/interim/edges_t<thr>.parquet, plus
data/interim/fps.npy, chemspace_members.parquet and chemspace_sweep.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seagap import config  # noqa: E402
from seagap.tables import load_table, save_table, table_path  # noqa: E402

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("rdkit is required: conda install -c conda-forge rdkit")

try:
    import igraph as ig
    import leidenalg
except ImportError:
    sys.exit("igraph and leidenalg are required:\n"
             "  conda install -c conda-forge python-igraph leidenalg")

try:
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    HAVE_SK = True
except ImportError:
    HAVE_SK = False
    print("note: scikit-learn not found, AMI/ARI will be skipped\n")

pd.set_option("display.width", 200)
print("SCRIPT VERSION: 2026-10-14 chemspace\n")

# ------------------------------------------------------------------ select
df = load_table(config.INTERIM / "compounds_scoped")
print(f"{len(df):,} compounds in the release")

if config.TIER == "all":
    sub = df.copy()
elif config.TIER in df.columns:
    sub = df[df[config.TIER]].copy()
else:
    sys.exit(f"config.TIER = {config.TIER!r} is not a column in compounds_scoped")
print(f"scope = {config.TIER}: {len(sub):,} compounds")

# ------------------------------------------------------------------ dedup
before = len(sub)
if "standard_inchi_key" in sub.columns:
    key = sub["standard_inchi_key"]
    rep = key.map(key.value_counts())
    sub["n_reports"] = rep.fillna(1).astype(int)
    sub = sub[key.notna()].drop_duplicates(subset="standard_inchi_key", keep="first")
    print(f"structural dedup on InChIKey: {before:,} -> {len(sub):,}")
else:
    sub["n_reports"] = 1

sub = sub.reset_index(drop=True)

# ------------------------------------------------------------------ fingerprints
print("\nFingerprinting")
gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=config.FP_RADIUS, fpSize=config.FP_NBITS)

fps, ok, scaffolds = [], np.zeros(len(sub), dtype=bool), []
t0 = time.time()
for i, smi in enumerate(sub["canonical_smiles"].tolist()):
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None:
        scaffolds.append(None)
        continue
    fps.append(gen.GetFingerprint(mol))
    ok[i] = True
    try:
        scaffolds.append(Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)))
    except Exception:
        scaffolds.append(None)
    if (i + 1) % 5000 == 0:
        print(f"  {i + 1:>7,}/{len(sub):,}  ({time.time() - t0:.0f}s)", flush=True)

sub["murcko_rdkit"] = scaffolds
print(f"  parsed {int(ok.sum()):,} / {len(sub):,} "
      f"({int((~ok).sum()):,} failed) in {time.time() - t0:.0f}s")

sub = sub[ok].reset_index(drop=True)
n = len(sub)
assert len(fps) == n, f"fingerprint/row mismatch: {len(fps)} vs {n}"

# Dense bit matrix, kept for later steps (embedding, nearest neighbours).
arr = np.zeros((n, config.FP_NBITS), dtype=np.uint8)
for i, fp in enumerate(fps):
    DataStructs.ConvertToNumpyArray(fp, arr[i])
np.save(config.INTERIM / "fps.npy", np.packbits(arr, axis=1))
print(f"  saved fps.npy ({np.packbits(arr, axis=1).nbytes / 1e6:.0f} MB packed)")

# ------------------------------------------------------------------ labels
labels = {}
for col in ["np_classifier_pathway", "np_classifier_superclass",
            "np_classifier_class", "chemical_super_class"]:
    if col in sub.columns and sub[col].notna().any():
        labels[col] = sub[col].fillna("(missing)").astype(str).values
print(f"\nvalidation labels available: {list(labels)}")


def edges_at(threshold):
    """All pairs i<j with Tanimoto >= threshold, via RDKit bulk similarity."""
    src, dst, sim = [], [], []
    step = max(1, n // 20)
    t = time.time()
    for i in range(n - 1):
        s = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
        hit = np.nonzero(s >= threshold)[0]
        if hit.size:
            src.append(np.full(hit.size, i, dtype=np.int32))
            dst.append((hit + i + 1).astype(np.int32))
            sim.append(s[hit].astype(np.float32))
        if i and i % step == 0:
            print(f"    {i:>7,}/{n:,}  ({time.time() - t:.0f}s)", flush=True)
    if not src:
        return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32))
    return np.concatenate(src), np.concatenate(dst), np.concatenate(sim)


# ------------------------------------------------------------------ one pass
# Edges above a high threshold are a subset of those above a lower one, so all
# pairs are scored once at the lowest threshold and the rest are filtered from
# that result. The earlier version rescored every pair per threshold.
thresholds = sorted(config.THRESHOLD_SWEEP)
base = thresholds[0]
print(f"\nScoring all {n * (n - 1) // 2:,} pairs once at threshold {base}")
print(f"(higher thresholds are filtered from this result, not rescored)\n")

t0 = time.time()
b_src, b_dst, b_sim = edges_at(base)
print(f"  {len(b_src):,} edges at >= {base} in {time.time() - t0:.0f}s")

# ------------------------------------------------------------------ sweep
print(f"\nDeriving {len(thresholds)} thresholds\n")

rows, parts = [], {}
for thr in thresholds:
    t0 = time.time()
    keep = b_sim >= thr
    src, dst, sim = b_src[keep], b_dst[keep], b_sim[keep]
    print(f"  threshold {thr}: {len(src):,} edges")
    el = time.time() - t0

    g = ig.Graph(n=n, edges=list(zip(src.tolist(), dst.tolist())), directed=False)
    g.es["weight"] = sim.tolist()
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, weights="weight",
        resolution_parameter=config.LEIDEN_RESOLUTION, seed=config.LEIDEN_SEED)
    memb = np.asarray(part.membership)
    parts[thr] = memb

    sizes = np.bincount(memb)
    row = {
        "threshold": thr,
        "edges": int(len(src)),
        "density": round(len(src) / (n * (n - 1) / 2) * 100, 4),
        "communities": int(len(sizes)),
        "largest_frac": round(float(sizes.max() / n), 4),
        "singletons": int((sizes == 1).sum()),
        "n_ge5": int((sizes >= 5).sum()),
        "modularity": round(float(part.modularity), 4),
        "seconds": round(el, 1),
    }
    if HAVE_SK:
        for name, y in labels.items():
            m = y != "(missing)"
            if m.sum() > 100:
                row[f"ami_{name.replace('np_classifier_', '')}"] = round(
                    adjusted_mutual_info_score(y[m], memb[m]), 4)
        y = labels.get("np_classifier_pathway")
        if y is not None:
            m = y != "(missing)"
            row["ari_pathway"] = round(adjusted_rand_score(y[m], memb[m]), 4)
    rows.append(row)
    print(f"    edges={len(src):,}  communities={row['communities']:,}  "
          f"largest={row['largest_frac']:.3f}  "
          f"AMI(superclass)={row.get('ami_superclass', float('nan'))}")

    if len(src) <= config.MAX_EDGES_TO_SAVE:
        p = save_table(pd.DataFrame({"src": src, "dst": dst, "sim": sim}),
                       config.INTERIM / f"edges_t{thr}")
        print(f"    saved {p.name}")
    else:
        print(f"    (not saved: {len(src):,} edges exceeds MAX_EDGES_TO_SAVE)")

sweep = pd.DataFrame(rows)
print("\n" + "=" * 74)
print("Sweep")
print("=" * 74)
print(sweep.to_string(index=False))

# ------------------------------------------------------------------ choose
ami_cols = [c for c in sweep.columns if c.startswith("ami_")]
if ami_cols:
    key = "ami_superclass" if "ami_superclass" in ami_cols else ami_cols[0]
    best = sweep.loc[sweep[key].idxmax()]
    print(f"\nBest agreement with NPClassifier ({key}): "
          f"threshold {best['threshold']} (AMI {best[key]})")
    print("A partition that recovers an independent chemical classification is "
          "doing\nsomething real. Check the shape columns too: a largest_frac "
          "near 1 means one\ngiant blob regardless of what AMI says.")
    chosen = float(best["threshold"])
else:
    sane = sweep[sweep["largest_frac"] < 0.25]
    chosen = float((sane if len(sane) else sweep).iloc[0]["threshold"])
    print(f"\nNo AMI available; falling back to component shape: {chosen}")

memb = parts[chosen]
sub["family"] = memb
sizes = pd.Series(memb).value_counts()
sub["family_size"] = sub["family"].map(sizes)

print(f"\nChosen threshold {chosen}: {len(sizes):,} families, "
      f"{int((sizes >= 5).sum()):,} with >=5 members")

big = sub[sub["family_size"] >= 5]
if len(big):
    fam = (big.groupby("family")
           .agg(n=("molecule_id", "size"),
                n_scaffolds=("murcko_rdkit", "nunique"),
                pathway=("np_classifier_pathway",
                         lambda s: s.mode().iloc[0] if s.notna().any() else None),
                purity=("np_classifier_pathway",
                        lambda s: round(s.value_counts(normalize=True).iloc[0], 2)
                        if s.notna().any() else None),
                superclass=("np_classifier_superclass",
                            lambda s: s.mode().iloc[0] if s.notna().any() else None),
                example=("name", lambda s: s.dropna().iloc[0]
                         if s.notna().any() else None))
           .sort_values("n", ascending=False))
    print("\nTop 20 families:")
    print(fam.head(20).to_string())
    print(f"\nMean pathway purity across families with >=5 members: "
          f"{fam['purity'].mean():.3f}")
    print("Purity near 1 means families map cleanly onto known chemistry.")

# Every threshold must have left its own file behind, with distinct edge counts.
print("\nEdge lists on disk:")
seen = {}
for thr in thresholds:
    p = table_path(config.INTERIM / f"edges_t{thr}")
    if p is None:
        print(f"  MISSING edges_t{thr}")
        continue
    sz = p.stat().st_size
    clash = [k for k, v in seen.items() if v == p.name]
    seen[thr] = p.name
    print(f"  {p.name:28s} {sz / 1e6:7.1f} MB"
          + (f"   COLLISION with threshold {clash[0]}" if clash else ""))

out = save_table(sub.reset_index(drop=True),
                 config.INTERIM / "chemspace_members")
(config.INTERIM / "chemspace_sweep.json").write_text(json.dumps({
    "tier": config.TIER,
    "n_compounds": n,
    "fp_radius": config.FP_RADIUS,
    "fp_nbits": config.FP_NBITS,
    "leiden_resolution": config.LEIDEN_RESOLUTION,
    "leiden_seed": config.LEIDEN_SEED,
    "chosen_threshold": chosen,
    "sweep": rows,
}, indent=2, default=float))
print(f"\nWrote {out}")
print(f"Wrote {config.INTERIM / 'chemspace_sweep.json'}")
