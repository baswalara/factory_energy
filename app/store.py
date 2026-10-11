"""DuckDB store: the four Annex B tables, run traces, saved bridge results and small metadata."""
from __future__ import annotations

import json
import threading

import duckdb
import pandas as pd

from .config import db_path

SCHEMA = {
    "meter_15min": ("meter_id VARCHAR, ts TIMESTAMP, kwh DOUBLE", ["meter_id", "ts"]),
    "tariff": ("valid_from TIMESTAMP, valid_to TIMESTAMP, period VARCHAR, price_per_kwh DOUBLE",
               ["valid_from", "period"]),
    "production": ("date DATE, product VARCHAR, units BIGINT, shift_pattern VARCHAR, is_holiday BOOLEAN",
                   ["date", "product"]),
    "weather": ("date DATE, avg_temp_c DOUBLE, cooling_degree_days DOUBLE, heating_degree_days DOUBLE",
                ["date"]),
}
TABLES = list(SCHEMA)


def merge_tariff(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """A corrected tariff replaces older prices only inside the time span it covers.
    Old rows are trimmed to the part outside that span (same price label, or 'flat' on either side)."""
    pieces = []
    for row in old.itertuples(index=False):
        clash = new[(new.period == row.period) | (new.period == "flat") | (row.period == "flat")]
        if clash.empty:
            pieces.append(row._asdict())
            continue
        lo, hi = clash.valid_from.min(), clash.valid_to.max()
        if row.valid_from < lo:
            pieces.append({**row._asdict(), "valid_to": min(row.valid_to, lo)})
        if row.valid_to > hi:
            pieces.append({**row._asdict(), "valid_from": max(row.valid_from, hi)})
    kept = pd.DataFrame(pieces, columns=new.columns)
    return new.copy() if kept.empty else pd.concat([kept, new], ignore_index=True)


class Store:
    def __init__(self, path: str | None = None):
        self.path = path or db_path()
        self.lock = threading.RLock()
        self.con = duckdb.connect(self.path)
        for name, (cols, key) in SCHEMA.items():
            self.con.execute(f"CREATE TABLE IF NOT EXISTS {name} ({cols}, PRIMARY KEY ({', '.join(key)}))")
        self.con.execute("CREATE TABLE IF NOT EXISTS meta (key VARCHAR PRIMARY KEY, value VARCHAR)")
        self.con.execute("CREATE TABLE IF NOT EXISTS traces (run_id VARCHAR PRIMARY KEY, created TIMESTAMP, record VARCHAR)")
        self.con.execute("CREATE TABLE IF NOT EXISTS bridge_results (run_id VARCHAR, created TIMESTAMP, "
                         "data_version BIGINT, period VARCHAR, compare_to VARCHAR, result VARCHAR)")

    # ---- data tables -----------------------------------------------------
    def upsert(self, table: str, df: pd.DataFrame) -> None:
        """Rows with the same key replace the stored ones. A tariff file is also
        authoritative for the time span it covers, so older prices inside that span go."""
        cols = [c.split()[0] for c in SCHEMA[table][0].split(", ")]
        df = df[cols]
        with self.lock:
            if table == "tariff" and len(df):
                df = merge_tariff(self.read("tariff"), df)[cols]
                self.con.execute("DELETE FROM tariff")
            self.con.register("incoming", df)
            self.con.execute(f"INSERT OR REPLACE INTO {table} SELECT {', '.join(cols)} FROM incoming")
            self.con.unregister("incoming")
            self.set_meta("data_version", self.data_version() + 1)

    def read(self, table: str) -> pd.DataFrame:
        with self.lock:
            return self.con.execute(f"SELECT * FROM {table}").df()

    def count(self, table: str) -> int:
        with self.lock:
            return self.con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    def clear(self) -> None:
        with self.lock:
            for t in TABLES:
                self.con.execute(f"DELETE FROM {t}")
            self.con.execute("DELETE FROM meta WHERE key LIKE 'load:%'")
            self.set_meta("data_version", self.data_version() + 1)

    # ---- metadata ----------------------------------------------------------
    def set_meta(self, key: str, value) -> None:
        with self.lock:
            self.con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", [key, json.dumps(value, default=str)])

    def get_meta(self, key: str, default=None):
        with self.lock:
            row = self.con.execute("SELECT value FROM meta WHERE key = ?", [key]).fetchone()
        return json.loads(row[0]) if row else default

    def data_version(self) -> int:
        return int(self.get_meta("data_version", 0))

    # ---- traces and results ------------------------------------------------
    def save_trace(self, run_id: str, record: dict) -> None:
        with self.lock:
            self.con.execute("INSERT OR REPLACE INTO traces VALUES (?, now()::TIMESTAMP, ?)",
                             [run_id, json.dumps(record, default=str)])

    def get_trace(self, run_id: str) -> dict | None:
        with self.lock:
            row = self.con.execute("SELECT record FROM traces WHERE run_id = ?", [run_id]).fetchone()
        return json.loads(row[0]) if row else None

    def save_bridge(self, run_id: str, period: str, compare_to: str, result: dict) -> None:
        with self.lock:
            self.con.execute("INSERT INTO bridge_results VALUES (?, now()::TIMESTAMP, ?, ?, ?, ?)",
                             [run_id, self.data_version(), period, compare_to, json.dumps(result, default=str)])

    def previous_bridge(self, period: str, compare_to: str, current: dict | None = None) -> dict | None:
        """A saved bridge for these periods from an OLDER data version: the most recent one that differs
        from `current` (so a later no-effect load does not hide an earlier correction), else the most recent."""
        with self.lock:
            rows = self.con.execute(
                "SELECT result, data_version FROM bridge_results WHERE period = ? AND compare_to = ? "
                "AND data_version < ? ORDER BY data_version DESC, created DESC",
                [period, compare_to, self.data_version()]).fetchall()
        found = [{**json.loads(r[0]), "data_version": r[1]} for r in rows]
        if not found:
            return None
        if current:
            key = lambda b: ([r["pct_points"] for r in b["bridge"]], b["bill"]["change_pct"])
            for j, b in enumerate(found):
                if key(b) != key(current):
                    # step back over older identical results, so every load since then is listed
                    while j + 1 < len(found) and key(found[j + 1]) == key(b):
                        j += 1
                    return {**b, "data_version": found[j]["data_version"]}
        return found[0]

    def log_ingest(self, text: str) -> None:
        self.set_meta("ingest_log", self.get_meta("ingest_log", []) + [{"data_version": self.data_version(), "what": text}])

    def ingests_since(self, version: int) -> str:
        items = [i["what"] for i in self.get_meta("ingest_log", []) if i["data_version"] > version]
        return "; ".join(items) if items else "not recorded"
