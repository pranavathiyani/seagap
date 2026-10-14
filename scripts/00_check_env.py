"""Verify the environment before running anything expensive.

    python scripts/00_check_env.py

Checks every import the pipeline needs, prints versions, and runs a two-molecule
RDKit fingerprint plus a four-node Leiden partition so a broken install fails here
in a second rather than twenty minutes into step 05.
"""

import importlib
import sys

REQUIRED = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("pyarrow", "pyarrow"),
    ("scipy", "scipy"),
    ("rdkit", "rdkit"),
    ("igraph", "python-igraph"),
    ("leidenalg", "leidenalg"),
    ("requests", "requests"),
]
OPTIONAL = [
    ("sklearn", "scikit-learn  (threshold choice falls back to graph shape)"),
    ("matplotlib", "matplotlib    (plots only)"),
]

print(f"python {sys.version.split()[0]}\n{sys.executable}\n")

missing = []
for mod, pkg in REQUIRED:
    try:
        m = importlib.import_module(mod)
        print(f"  ok       {mod:12s} {getattr(m, '__version__', '?')}")
    except ImportError:
        print(f"  MISSING  {mod:12s} -> {pkg}")
        missing.append(pkg)

for mod, note in OPTIONAL:
    try:
        m = importlib.import_module(mod)
        print(f"  ok       {mod:12s} {getattr(m, '__version__', '?')}")
    except ImportError:
        print(f"  absent   {mod:12s} {note}")

if missing:
    print("\nInstall the missing packages:")
    print("  pip install " + " ".join(missing))
    sys.exit(1)

# ------------------------------------------------------------------ smoke tests
print("\nSmoke tests")

from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

RDLogger.DisableLog("rdApp.*")

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
a = gen.GetFingerprint(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))   # aspirin
b = gen.GetFingerprint(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"))      # paracetamol
sim = DataStructs.TanimotoSimilarity(a, b)
assert 0.0 < sim < 1.0, sim
print(f"  rdkit fingerprint + Tanimoto     ok  (aspirin vs paracetamol = {sim:.3f})")

bulk = DataStructs.BulkTanimotoSimilarity(a, [a, b])
assert abs(bulk[0] - 1.0) < 1e-9 and abs(bulk[1] - sim) < 1e-9
print(f"  rdkit BulkTanimotoSimilarity     ok")

scaf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(
    Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")))
assert scaf, "empty scaffold"
print(f"  rdkit Murcko scaffold            ok  ({scaf})")

import numpy as np  # noqa: E402
arr = np.zeros((1, 2048), dtype=np.uint8)
DataStructs.ConvertToNumpyArray(a, arr[0])
assert arr.sum() > 0
print(f"  rdkit -> numpy conversion        ok  ({int(arr.sum())} bits set)")

import igraph as ig  # noqa: E402
import leidenalg  # noqa: E402

g = ig.Graph(n=4, edges=[(0, 1), (2, 3)], directed=False)
g.es["weight"] = [1.0, 1.0]
part = leidenalg.find_partition(
    g, leidenalg.RBConfigurationVertexPartition, weights="weight",
    resolution_parameter=1.0, seed=0)
assert len(set(part.membership)) == 2, part.membership
print(f"  igraph + leidenalg partition     ok  (2 communities from 2 components)")

import pandas as pd  # noqa: E402
import tempfile, pathlib  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d) / "t.parquet"
    pd.DataFrame({"a": [[1, 2], [3]], "b": ["x", "y"]}).to_parquet(p, index=False)
    back = pd.read_parquet(p)
    assert isinstance(back["a"].iloc[0], (list, np.ndarray)), type(back["a"].iloc[0])
print("  parquet round-trip (list column) ok")

print("\nEnvironment is ready. Run the steps in order:")
for s in ["01_build.py", "02_taxonomy.py", "04_scope.py", "05_chemspace.py"]:
    print(f"  python scripts/{s}")
