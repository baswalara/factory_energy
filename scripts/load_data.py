"""Loader (guide section 4.7): load CSV files in the Annex B schema, then validate.

    python scripts/load_data.py data/history --reset        # the past data: 5 quarters, empty the store first
    python scripts/load_data.py data/demo_new_quarter       # add a new quarter on top of the history
    python scripts/load_data.py data/sample --reset         # the small synthetic pack used by tests and eval
    python scripts/load_data.py path/to/tariff.csv --table tariff
    python scripts/load_data.py data/late_drop --api http://localhost:8000   # while the API is running

DuckDB allows one writer, so while the API is running use --api (it uploads through POST /ingest).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cleaning import get_dataset          # noqa: E402
from app.config import load_config            # noqa: E402
from app.loader import find_files, load_csv_bytes, load_dir  # noqa: E402
from app.store import TABLES, Store           # noqa: E402
from app.validation import as_text, validate  # noqa: E402


def via_api(a) -> int:
    import httpx
    p = Path(a.path)
    files = find_files(str(p)) if p.is_dir() else [(a.table or next((t for t in TABLES if p.stem.startswith(t)), p.stem), p)]
    if not files:
        print("no Annex B files found")
        return 1
    for table, f in files:
        r = httpx.post(a.api.rstrip("/") + "/ingest", files={"file": (f.name, f.read_bytes())}, data={"table": table}, timeout=300)
        print(json.dumps(r.json(), indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="folder with Annex B CSV files, or one CSV file")
    ap.add_argument("--table", help="table name when path is a single file")
    ap.add_argument("--reset", action="store_true", help="delete stored data first")
    ap.add_argument("--if-empty", action="store_true", help="do nothing when data is already loaded")
    ap.add_argument("--api", help="base URL of the running API; files are sent to POST /ingest")
    a = ap.parse_args()
    if a.api:
        return via_api(a)
    try:
        store, cfg = Store(), load_config()
    except Exception as e:
        print(f"Could not open the data store ({type(e).__name__}). If the API is running it holds the store: "
              f"re-run with --api http://localhost:8000")
        return 3
    if a.if_empty and store.count("meter_15min"):
        print("data already loaded; skipping")
        return 0
    if a.reset:
        store.clear()
    p = Path(a.path)
    if p.is_dir():
        results = load_dir(store, str(p))
        card = p / "data_card.json"
    else:
        table = a.table or next((t for t in TABLES if p.stem.startswith(t)), p.stem)
        results = [load_csv_bytes(store, table, p.read_bytes(), str(p))]
        card = p.parent / "data_card.json"
    for r in results:
        print(json.dumps(r))
    if not results:
        print("no Annex B files found (expected meter_15min*.csv, tariff*.csv, production*.csv, weather*.csv)")
        return 1
    report = validate(store, get_dataset(store, cfg), str(card) if p.is_dir() else None)
    print(as_text(report))
    return 0 if report["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
