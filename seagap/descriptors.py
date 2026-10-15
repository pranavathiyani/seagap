"""The physicochemical panel, defined once.

Steps 13 and 14 compute descriptors on two different libraries and compare the
results. If each kept its own copy of the descriptor code, one edit to a cutoff
would silently invalidate every comparison in the paper. The panel lives here so
there is exactly one definition of what a stereocentre or a macrocycle is.

Deterministic only. Every number is computed from the structure by a documented
function. Trained ADMET models are a separate concern with separate caveats and
do not belong in this module.
"""

import sys
from pathlib import Path

import numpy as np

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, RDConfig, rdMolDescriptors

MACROCYCLE_MIN = 12       # ring size at which a ring is called macrocyclic

# Collects stereo-perception failures so a total failure cannot pass as a
# legitimate zero, which is exactly what happened before flagPossibleStereo-
# Centers was added below. Callers read it after a pass and report the count.
STEREO_ERRORS = []

NUMERIC = ["mw", "clogp", "tpsa", "hbd", "hba", "rotb", "heavy_atoms",
           "n_rings", "n_aromatic_rings", "fsp3", "n_stereocentres",
           "n_stereo_specified", "n_stereo_unspecified",
           "n_heteroatoms", "qed", "sa_score", "np_likeness"]


def load_contrib(quiet=False):
    """(sa_score_fn, np_score_fn), either of which may be None.

    SA_Score (Ertl & Schuffenhauer 2009) and NP_Score (Ertl et al. 2008) ship
    with RDKit but live in Contrib, outside the importable package, so they need
    the path added by hand. Some builds omit Contrib. Both are optional: a
    missing scorer warns and leaves its column empty rather than killing a run.
    """
    sa_fn = np_fn = None
    contrib = Path(RDConfig.RDContribDir)

    sa_dir = contrib / "SA_Score"
    if sa_dir.is_dir():
        sys.path.append(str(sa_dir))
        try:
            import sascorer
            sa_fn = sascorer.calculateScore
        except Exception as e:                                  # noqa: BLE001
            print(f"  note: SA_Score present but unusable ({e})")
    elif not quiet:
        print(f"  note: no SA_Score in {contrib}, synthetic accessibility "
              "skipped")

    np_dir = contrib / "NP_Score"
    if np_dir.is_dir():
        sys.path.append(str(np_dir))
        try:
            import npscorer
            fscore = npscorer.readNPModel()
            np_fn = lambda m: npscorer.scoreMol(m, fscore)      # noqa: E731
        except Exception as e:                                  # noqa: BLE001
            print(f"  note: NP_Score present but unusable ({e})")
    elif not quiet:
        print(f"  note: no NP_Score in {contrib}, NP-likeness skipped")

    return sa_fn, np_fn


def describe(smiles, sa_fn=None, np_fn=None):
    """One compound's panel, or None if the SMILES will not parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    clogp = Crippen.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)

    rings = mol.GetRingInfo().AtomRings()
    max_ring = max((len(r) for r in rings), default=0)

    try:
        # flagPossibleStereoCenters=True is required by BOTH calls below, not
        # just the unspecified one. Without it they return 0 instead of
        # raising, which reported "0 stereocentres" for all 29,154 compounds in
        # an earlier run even though 83% of the SMILES carry an @.
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True,
                                   flagPossibleStereoCenters=True)
        # CalcNumAtomStereoCenters returns the TOTAL, specified and unspecified
        # together. Specified is the difference, not a separate call. Adding
        # the two functions double-counts every centre.
        total = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        unspec = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
        spec = total - unspec
    except Exception as e:                                      # noqa: BLE001
        STEREO_ERRORS.append(str(e))
        total = unspec = spec = np.nan

    # Rule of Five, counted not judged. Marine natural products violate it
    # constitutively, so the count is a description and not a verdict.
    ro5 = int(mw > 500) + int(clogp > 5) + int(hbd > 5) + int(hba > 10)

    row = {
        "mw": mw,
        "clogp": clogp,
        "tpsa": tpsa,
        "hbd": hbd,
        "hba": hba,
        "rotb": rotb,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "n_rings": rdMolDescriptors.CalcNumRings(mol),
        "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "max_ring_size": max_ring,
        "macrocyclic": max_ring >= MACROCYCLE_MIN,
        "fsp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "n_stereocentres": total,
        "n_stereo_specified": spec,
        "n_stereo_unspecified": unspec,
        "n_heteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
        "ro5_violations": ro5,
        "veber_pass": bool(rotb <= 10 and tpsa <= 140),
        "qed": QED.qed(mol),
    }
    row["sa_score"] = float(sa_fn(mol)) if sa_fn else np.nan
    row["np_likeness"] = float(np_fn(mol)) if np_fn else np.nan
    return row


def profile(smiles_iter, sa_fn=None, np_fn=None, ids=None, label="",
            progress_every=5_000):
    """Panel for an iterable of SMILES. Returns (list of rows, list of failures).

    Rows carry an `id` key when `ids` is given, so the caller can rejoin.
    """
    rows, failed = [], []
    ids = ids if ids is not None else range(len(smiles_iter))
    for i, (key, smi) in enumerate(zip(ids, smiles_iter), start=1):
        if progress_every and i % progress_every == 0:
            print(f"  {label}{i:,}")
        if not isinstance(smi, str) or not smi:
            failed.append(key)
            continue
        r = describe(smi, sa_fn, np_fn)
        if r is None:
            failed.append(key)
            continue
        r["id"] = key
        rows.append(r)
    return rows, failed


def murcko(smiles):
    """Bemis-Murcko scaffold SMILES, or None. Used for set overlap, so the
    string must be canonical and stereochemistry is stripped: two libraries
    should match on skeleton, not on configuration."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if core is None or core.GetNumAtoms() == 0:
            return None
        Chem.RemoveStereochemistry(core)
        return Chem.MolToSmiles(core)
    except Exception:                                           # noqa: BLE001
        return None


def inchikey_block(smiles):
    """First block of the InChIKey: the connectivity layer, 14 characters.

    Matching two libraries on this rather than the full key means a compound
    recorded with different stereochemistry, tautomer or salt form in each
    still matches, which is what cross-database overlap should mean.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        k = Chem.MolToInchiKey(mol)
        return k.split("-")[0] if k else None
    except Exception:                                           # noqa: BLE001
        return None
