"""Loads CSV files in the Annex B schema. Used by the CLI loader and by POST /ingest.

Extra columns are ignored. Missing required columns reject the file. Bad rows are
rejected and counted, never silently repaired.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

from .store import Store, TABLES

REQUIRED = {
    "meter_15min": ["meter_id", "ts", "kwh"],
    "tariff": ["valid_from", "valid_to", "period", "price_per_kwh"],
    "production": ["date", "product", "units", "shift_pattern", "is_holiday"],
    "weather": ["date", "avg_temp_c", "cooling_degree_days", "heating_degree_days"],
}
KEYS = {"meter_15min": ["meter_id", "ts"], "tariff": ["valid_from", "period"],
        "production": ["date", "product"], "weather": ["date"]}
TRUE, FALSE = {"true", "1", "yes", "y", "t"}, {"false", "0", "no", "n", "f", ""}


def _utc_naive(s: pd.Series) -> pd.Series:
    """Parse to UTC and drop the zone. Values without an offset are taken as UTC (Annex B)."""
    return pd.to_datetime(s, utc=True, errors="coerce", format="mixed").dt.tz_localize(None).astype("datetime64[us]")


def _offsets(s: pd.Series) -> list[str]:
    """Distinct UTC offsets written in the raw timestamps, for the 'one time zone' check."""
    tail = s.dropna().astype(str).str.strip().str.extract(r"(Z|[+-]\d{2}:?\d{2})$", flags=re.I)[0]
    return sorted(tail.fillna("none").str.upper().replace({"+00:00": "Z", "+0000": "Z"}).unique().tolist())


def normalise(table: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Returns (clean rows, stats). Raises ValueError when the file cannot be this table."""
    raw = raw.rename(columns=lambda c: str(c).strip().lower())
    missing = [c for c in REQUIRED[table] if c not in raw.columns]
    if missing:
        raise ValueError(f"missing required column(s) for {table}: {missing}")
    df = raw[REQUIRED[table]].copy()
    stats: dict = {"rows_in_file": len(df)}

    if table == "meter_15min":
        stats["tz_offsets"] = _offsets(df.ts)
        df["meter_id"] = df.meter_id.astype("string").str.strip().str.lower()
        df["ts"] = _utc_naive(df.ts)
        df["kwh"] = pd.to_numeric(df.kwh, errors="coerce")
        ok = df.meter_id.notna() & (df.meter_id != "") & df.ts.notna() & df.kwh.notna()
        stats["negative_readings"] = int((df.kwh[ok] < 0).sum())       # kept: cleaning treats them as resets
        stats["off_grid_timestamps"] = int((df.ts[ok].dt.minute % 15 != 0).sum() + (df.ts[ok].dt.second != 0).sum())
    elif table == "tariff":
        df["valid_from"], df["valid_to"] = _utc_naive(df.valid_from), _utc_naive(df.valid_to)
        df["period"] = df.period.astype("string").str.strip().str.lower().str.replace("-", "_")
        df["price_per_kwh"] = pd.to_numeric(df.price_per_kwh, errors="coerce")
        ok = (df.valid_from.notna() & df.valid_to.notna() & (df.valid_to > df.valid_from)
              & df.period.isin(["peak", "off_peak", "flat"]) & (df.price_per_kwh > 0))
    elif table == "production":
        df["date"] = pd.to_datetime(df.date, errors="coerce").dt.date
        df["product"] = df["product"].astype("string").str.strip()
        df["units"] = pd.to_numeric(df.units, errors="coerce")
        df["shift_pattern"] = df.shift_pattern.astype("string").str.strip().fillna("")
        flag = df.is_holiday.astype("string").str.strip().str.lower().fillna("")
        ok = (df.date.notna() & df["product"].notna() & (df["product"] != "") & df.units.notna()
              & (df.units >= 0) & flag.isin(TRUE | FALSE))
        df["is_holiday"] = flag.isin(TRUE)
    else:  # weather
        df["date"] = pd.to_datetime(df.date, errors="coerce").dt.date
        for c in REQUIRED[table][1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        ok = df.notna().all(axis=1) & (df.cooling_degree_days >= 0) & (df.heating_degree_days >= 0)

    good = df[ok.fillna(False)].copy()
    stats["rows_rejected"] = int(len(df) - len(good))
    before = len(good)
    good = good.drop_duplicates(KEYS[table], keep="last")
    stats["duplicates_collapsed"] = int(before - len(good))
    if table == "production":
        good["units"] = good.units.round().astype("int64")
    for c in good.select_dtypes("string").columns:
        good[c] = good[c].astype(object)
    return good, stats


def load_frame(store: Store, table: str, raw: pd.DataFrame, source: str = "") -> dict:
    if table not in TABLES:
        return {"table": table, "rows_loaded": 0, "rows_rejected": 0, "status": "rejected",
                "detail": f"unknown table; expected one of {TABLES}"}
    try:
        good, stats = normalise(table, raw)
    except ValueError as e:
        return {"table": table, "rows_loaded": 0, "rows_rejected": len(raw), "status": "rejected", "detail": str(e)}
    if len(good):
        store.upsert(table, good)
    result = {"table": table, "rows_loaded": int(len(good)), "rows_rejected": stats["rows_rejected"],
              "status": "loaded" if stats["rows_rejected"] == 0 and len(good) else
                        ("loaded_with_rejects" if len(good) else "rejected"),
              "duplicates_collapsed": stats["duplicates_collapsed"], "source": source}
    store.set_meta(f"load:{table}", {**stats, **result})
    return result


def load_csv_bytes(store: Store, table: str, content: bytes, source: str = "") -> dict:
    try:
        raw = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    except Exception as e:  # unreadable file: report, do not crash the API
        return {"table": table, "rows_loaded": 0, "rows_rejected": 0, "status": "rejected", "detail": f"not a readable CSV: {e}"}
    return load_frame(store, table, raw, source)


def find_files(folder: str) -> list[tuple[str, Path]]:
    """Annex B files in a folder: '<table>.csv', or any CSV whose name starts with the table name
    (e.g. meter_15min_5q_messy.csv, tariff_5q.csv, weather_2026q4.csv). Meter files go first."""
    out = []
    for table in TABLES:
        exact = Path(folder) / f"{table}.csv"
        found = [exact] if exact.exists() else sorted(Path(folder).glob(f"{table}*.csv"))
        out += [(table, f) for f in found]
    return out


def load_dir(store: Store, folder: str) -> list[dict]:
    """Loads every Annex B file found in the folder (exact or prefixed names)."""
    return [load_csv_bytes(store, table, path.read_bytes(), str(path)) for table, path in find_files(folder)]
