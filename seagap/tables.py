"""Save and load interim tables.

Parquet when an engine is available, pickle otherwise. Some columns hold Python
lists (organisms, locations, phyla), which CSV would silently mangle into strings,
so CSV is not a fallback here.

Extensions are APPENDED, never substituted. Path.with_suffix() replaces whatever
it reads as the current suffix, so a stem carrying a dot loses part of its name:
Path("edges_t0.45").with_suffix(".parquet") yields "edges_t0.parquet". Six
thresholds then collide on one filename.
"""

import pandas as pd


def _path(path, ext):
    """path plus ext, unless path already ends with ext."""
    return path if path.name.endswith(ext) else path.parent / (path.name + ext)


def _engine_available():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import fastparquet  # noqa: F401
        return True
    except ImportError:
        return False


def save_table(df, path):
    """Write df at `path`, adding .parquet or .pkl. Returns the path used."""
    pq = _path(path, ".parquet")
    if _engine_available():
        df.to_parquet(pq, index=False)
        return pq
    alt = _path(path, ".pkl")
    df.to_pickle(alt)
    print(f"  note: no parquet engine, wrote {alt.name} instead "
          "(pip install pyarrow for the parquet path)")
    return alt


def table_path(path):
    """The existing file for `path`, or None."""
    for ext in (".parquet", ".pkl"):
        p = _path(path, ext)
        if p.exists():
            return p
    return None


def load_table(path):
    """Load a table saved by save_table, whichever format it landed in."""
    p = table_path(path)
    if p is None:
        raise FileNotFoundError(
            f"neither {_path(path, '.parquet').name} nor "
            f"{_path(path, '.pkl').name} in {path.parent}")
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_pickle(p)
