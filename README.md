# SEAGAP

Mapping under-explored chemical space in marine natural products.

## Layout

```
/mnt/d/seagap/
  marinenp_v1_r20250608_csv/   unzipped release (not tracked)
  seagap/                      shared code
  scripts/                     runnable steps, in order
  data/interim/                outputs (not tracked)
```

## Run

```bash
cd /mnt/d/seagap
python scripts/01_build.py
```

## Steps

| Script | Does | Produces |
|---|---|---|
| `01_build.py` | Joins the 9 release tables into one compound table, applies the marine filter, audits coverage and sampling concentration | `compounds.parquet`, `audit.json` |

## Release schema notes

The release is relational, nine CSVs, COCONUT-derived. Points that drive the code:

- `molecules.is_marine` and `organisms.is_marine` both exist and need not agree.
  `config.MARINE_RULE` picks which defines the marine set. The dump contains
  terrestrial records, so filtering is not optional.
- `geo_locations.csv` is `id,name` only. There are no coordinates; geography is
  named places.
- `organisms.csv` carries a WoRMS AphiaID but no lineage. Ranks above the recorded
  one need resolving against WoRMS.
- `properties.csv` already carries NPClassifier and ClassyFire assignments, and
  `molecules.csv` already carries a Murcko framework. Neither needs recomputing.
- `standard_inchi_key` is present, so structural dedup does not need RDKit.
- `molecule_organism` and `geo_location_molecule` are many-to-many.

## Licence

MIT (code). Source data retains its own terms.
