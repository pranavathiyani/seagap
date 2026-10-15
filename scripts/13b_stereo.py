"""Read-only: why is every stereocentre count zero?

    python scripts/13b_stereo.py

Step 13 reported 0 stereocentres for all 29,154 compounds, with no exception
raised. Two things can produce that and they need opposite fixes.

A. The SMILES carry no stereochemistry. COCONUT, which MarineNP derives from,
   stores more than one structure representation and not all of them keep
   stereo descriptors. If the column in use is stereo-stripped, the count is
   correct and the fix is to read a different column.

B. The descriptor call needs more setup. CalcNumUnspecifiedAtomStereoCenters
   only answers after AssignStereochemistry has been called with
   flagPossibleStereoCenters=True. Without it the function returns 0 rather than
   raising, which is exactly the failure seen.

Both can be true at once. This script checks the strings first, because a
stereo marker in the text is decisive and needs no RDKit at all, then runs the
descriptor calls four ways on a handful of molecules.

Writes nothing.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seagap import config  # noqa: E402
from seagap.tables import load_table  # noqa: E402

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("rdkit is required: pip install rdkit")

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 60)
print("SCRIPT VERSION: 2026-10-15 stereo diagnostic\n")
bar = "=" * 98


def marker_counts(s):
    """How many strings carry each kind of stereo marker."""
    s = s.dropna().astype(str)
    if not len(s):
        return None
    return {
        "n": len(s),
        "pct_at": round(s.str.contains("@", regex=False).mean() * 100, 1),
        "pct_slash": round(s.str.contains("/", regex=False).mean() * 100, 1),
        "pct_backslash": round(s.str.contains("\\", regex=False).mean() * 100, 1),
    }


# ------------------------------------------------- 1. the column now in use
print(bar)
print("1. Stereo markers in the column step 13 reads")
print(bar)
work = load_table(config.INTERIM / "families")
mc = marker_counts(work["canonical_smiles"])
print(f"families.canonical_smiles: {mc}")
if mc and mc["pct_at"] == 0.0:
    print("\nNo '@' anywhere. This column is stereo-stripped, so a zero "
          "stereocentre\ncount for specified centres is the honest answer for "
          "it. Cause A holds.")
else:
    print("\nStereo descriptors are present in the strings, so a zero count "
          "is a\ndescriptor-call problem rather than a data problem. Cause B.")

print("\nthree examples:")
for s in work["canonical_smiles"].dropna().head(3):
    print(f"  {s[:110]}")


# ------------------------------------- 2. every SMILES-like release column
print("\n" + bar)
print("2. Other structure columns in the release")
print(bar)
mol = pd.read_csv(config.RELEASE / config.TABLES["molecules"], low_memory=False,
                  nrows=20_000)
print(f"molecules.csv columns: {list(mol.columns)}\n")

# No dtype test here. Pandas 3 gives string columns a str dtype rather than
# object, so an `== object` guard silently drops every structure column.
cands = [c for c in mol.columns
         if any(k in c.lower() for k in
                ("smiles", "smile", "structure", "inchi"))]
if not cands:
    print("no structure-like column names found")
for c in cands:
    print(f"  {c:28s} {marker_counts(mol[c])}")

print("\nA column with a high pct_at is the one to read. For InChI the stereo "
      "layer is\n/t or /m, not '@', so an inchi column showing 0% here may "
      "still carry it.")
inchi = [c for c in cands if "inchi" in c.lower()]
for c in inchi:
    s = mol[c].dropna().astype(str)
    if len(s):
        print(f"  {c:28s} pct with /t layer: "
              f"{round(s.str.contains('/t', regex=False).mean() * 100, 1)}")


# ------------------------------------------- 3. the descriptor calls, 4 ways
print("\n" + bar)
print("3. Descriptor behaviour on real molecules")
print(bar)
probe = "C[C@H](N)C(=O)O"                      # L-alanine, one specified centre
probe_flat = "CC(N)C(=O)O"                     # same skeleton, unspecified
samples = [("L-alanine (specified)", probe), ("alanine (flat)", probe_flat)]
samples += [(f"release row {i}", s) for i, s in
            enumerate(work["canonical_smiles"].dropna().head(3), start=1)]

rows = []
for label, smi in samples:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        continue
    r = {"molecule": label}

    # as step 13 ran it
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    r["spec_plain"] = rdMolDescriptors.CalcNumAtomStereoCenters(m)
    r["unspec_plain"] = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(m)

    # with possible centres flagged, which the docs require for the second call
    m2 = Chem.MolFromSmiles(smi)
    Chem.AssignStereochemistry(m2, cleanIt=True, force=True,
                               flagPossibleStereoCenters=True)
    r["spec_flagged"] = rdMolDescriptors.CalcNumAtomStereoCenters(m2)
    r["unspec_flagged"] = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(m2)

    # the modern perception API, which does its own setup
    try:
        si = Chem.FindPotentialStereo(Chem.MolFromSmiles(smi))
        r["potential_total"] = len(si)
        r["potential_atoms"] = sum(
            1 for e in si if "Atom" in str(e.type))
    except Exception as e:                                      # noqa: BLE001
        r["potential_total"] = f"raised: {e}"
        r["potential_atoms"] = None
    rows.append(r)

print(pd.DataFrame(rows).to_string(index=False))
print("\nL-alanine must show 1 specified. If spec_plain is 0 there, the call "
      "itself is\nwrong. If it shows 1 there but 0 on the release rows, the "
      "release rows have\nno stereochemistry to find.")
print("\nFlat alanine must show an unspecified centre once flagged. If "
      "unspec_plain is 0\nand unspec_flagged is 1, flagPossibleStereoCenters "
      "is the missing argument.")
