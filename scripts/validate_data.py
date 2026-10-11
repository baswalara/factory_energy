"""Validation script (guide section 4.5): runs the six checks on whatever is loaded.

    python scripts/validate_data.py [--card data/sample/data_card.json] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cleaning import get_dataset          # noqa: E402
from app.config import load_config            # noqa: E402
from app.store import Store                   # noqa: E402
from app.validation import as_text, validate  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    store = Store()
    report = validate(store, get_dataset(store, load_config()), a.card)
    print(json.dumps(report, indent=2) if a.json else as_text(report))
    raise SystemExit(0 if report["status"] != "FAIL" else 2)
