import os
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_tmp = tempfile.mkdtemp()
os.environ["DB_PATH"] = str(Path(_tmp) / "test.duckdb")     # must be set before app.api is imported
os.environ["MOCK_LLM"] = "true"


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app import api
    from app.loader import load_dir
    api.store.clear()
    load_dir(api.store, str(ROOT / "data" / "sample"))
    return TestClient(api.app)


@pytest.fixture(scope="session")
def ctx():
    """The untouched sample data in its own store, so API tests that ingest corrections cannot affect it."""
    from app.cleaning import build_dataset
    from app.config import load_config
    from app.loader import load_dir
    from app.store import Store
    store, cfg = Store(str(Path(_tmp) / "pristine.duckdb")), load_config()
    load_dir(store, str(ROOT / "data" / "sample"))
    return store, cfg, build_dataset(store, cfg)


def tiny_store(tmp_path, tariff_rows, days=2, kwh=1.0):
    """A store with a constant main meter and one line meter, for hand-checkable sums."""
    import pandas as pd

    from app.loader import load_frame
    from app.store import Store
    st = Store(str(tmp_path / "tiny.duckdb"))
    ts = pd.date_range("2026-07-01", periods=96 * days, freq="15min").strftime("%Y-%m-%dT%H:%M:%SZ")
    meter = pd.concat([pd.DataFrame({"meter_id": m, "ts": ts, "kwh": str(kwh)}) for m in ("main", "line_1")])
    load_frame(st, "meter_15min", meter)
    load_frame(st, "tariff", pd.DataFrame(tariff_rows, columns=["valid_from", "valid_to", "period", "price_per_kwh"]).astype(str))
    return st
