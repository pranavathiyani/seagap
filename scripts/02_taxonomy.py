"""Fetch WoRMS classification for every organism AphiaID in the release.

    python scripts/02_taxonomy.py

The release carries an AphiaID but no lineage, and its environment_aphia_worms
string is a flattened concatenation of WoRMS flags assigned at whatever rank the
record sits at. That string is not sufficient to tell a sponge from a garden herb.
This fetches the real record for each AphiaID so later steps can filter on phylum.

Resumable: results are cached in data/interim/worms_cache.json and existing entries
are never re-fetched. Safe to interrupt and re-run.

Writes data/interim/organism_taxonomy.parquet.
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seagap import config  # noqa: E402
from seagap.tables import save_table  # noqa: E402

API = "https://www.marinespecies.org/rest/AphiaRecordByAphiaID/{}"
CACHE = config.INTERIM / "worms_cache.json"
DELAY = 0.12          # polite pause between calls
TIMEOUT = 25
MAX_RETRY = 3

FIELDS = ["scientificname", "status", "rank", "kingdom", "phylum", "class",
          "order", "family", "genus", "isMarine", "isBrackish", "isFreshwater",
          "isTerrestrial", "isExtinct"]

print("SCRIPT VERSION: 2026-10-13 taxonomy\n")

org = pd.read_csv(config.RELEASE / config.TABLES["organisms"], low_memory=False)
ids = (org["aphiaid_worms"].dropna().astype("int64").unique().tolist())
ids = sorted(set(ids))
print(f"{len(org):,} organism rows, {len(ids):,} distinct AphiaIDs "
      f"({org['aphiaid_worms'].isna().sum():,} rows have none)")

cache = {}
if CACHE.exists():
    cache = json.loads(CACHE.read_text())
    print(f"cache: {len(cache):,} already fetched")

todo = [i for i in ids if str(i) not in cache]
print(f"to fetch: {len(todo):,}")
if todo:
    est = len(todo) * (DELAY + 0.25) / 60
    print(f"rough estimate: {est:.0f} min\n")


def save_cache():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache))


session = requests.Session()
session.headers["User-Agent"] = "SEAGAP/0.1 (research; marine natural products)"

for n, aid in enumerate(todo, 1):
    rec = None
    for attempt in range(MAX_RETRY):
        try:
            r = session.get(API.format(aid), timeout=TIMEOUT)
            if r.status_code == 200:
                rec = r.json()
                break
            if r.status_code == 204:          # no content, valid but empty
                rec = {}
                break
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            rec = {"_http": r.status_code}
            break
        except requests.RequestException as e:
            if attempt == MAX_RETRY - 1:
                rec = {"_error": str(e)[:200]}
            else:
                time.sleep(2 ** attempt)

    cache[str(aid)] = {k: (rec or {}).get(k) for k in FIELDS} if rec is not None else {}
    if rec and ("_error" in rec or "_http" in rec):
        cache[str(aid)].update({k: v for k, v in rec.items() if k.startswith("_")})

    if n % 250 == 0 or n == len(todo):
        save_cache()
        print(f"  {n:>6,}/{len(todo):,}  ({n / len(todo) * 100:5.1f}%)", flush=True)
    time.sleep(DELAY)

save_cache()

# ------------------------------------------------------------------ assemble
rows = []
for aid, rec in cache.items():
    row = {"aphiaid_worms": int(aid)}
    row.update({k: rec.get(k) for k in FIELDS})
    rows.append(row)
tax = pd.DataFrame(rows)

print(f"\nResolved {tax['phylum'].notna().sum():,} / {len(tax):,} AphiaIDs to a phylum")

failed = tax[tax["scientificname"].isna()]
if len(failed):
    print(f"{len(failed):,} returned nothing usable "
          "(re-running the script will retry only those if you delete them "
          "from worms_cache.json)")

print("\nTop phyla by organism count:")
print(tax["phylum"].fillna("(unresolved)").value_counts().head(20).to_string())

print("\nWoRMS flags:")
for f in ["isMarine", "isBrackish", "isFreshwater", "isTerrestrial"]:
    if f in tax:
        s = pd.to_numeric(tax[f], errors="coerce")
        print(f"  {f:16s} =1 for {int((s == 1).sum()):,} organisms")

print("\nStrictly marine (isMarine=1, all others 0):")
num = {f: pd.to_numeric(tax[f], errors="coerce").fillna(0) for f in
       ["isMarine", "isBrackish", "isFreshwater", "isTerrestrial"]}
strict = (num["isMarine"] == 1) & (num["isBrackish"] == 0) & \
         (num["isFreshwater"] == 0) & (num["isTerrestrial"] == 0)
print(f"  {int(strict.sum()):,} / {len(tax):,} organisms")

out = save_table(tax, config.INTERIM / "organism_taxonomy")
print(f"\nWrote {out}")
print(f"Cache at {CACHE} ({CACHE.stat().st_size / 1e6:.1f} MB)")
