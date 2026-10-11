"""Builds an example NEW quarter (2026-Q4) to upload on top of the history, for demos and tests.

The history in data/history ends with 2026-Q3. A real user uploads the next quarter's four files;
this script makes a stand-in from the same calendar quarter one year earlier (2025-Q4), shifted by
52 weeks so weekdays line up, then applies a few stated changes:

  - production volume x --volume (default 1.04)
  - the compressed-air leak seen in 2026-Q3 keeps running: +--leak-kw on compressors (and main),
    all hours, until --leak-fixed-on (default: never fixed)
  - the tariff continues at the last prices in the history, in the same block structure
    (weekday peak 07:00-22:00, off-peak otherwise, weekends flat)

    python scripts/make_demo_quarter.py                         # -> data/demo_new_quarter/
    python scripts/make_demo_quarter.py --days 30 --out data/demo_partial_quarter   # an in-progress quarter
    python scripts/make_demo_quarter.py --leak-fixed-on 2026-11-01

The output is SYNTHETIC: a demo input, not a forecast and not real data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SHIFT = pd.Timedelta(days=364)                  # 52 weeks: same weekday one year later
START, END = pd.Timestamp("2026-10-01"), pd.Timestamp("2027-01-01")


def first(folder: Path, table: str) -> Path:
    exact = folder / f"{table}.csv"
    return exact if exact.exists() else sorted(folder.glob(f"{table}*.csv"))[0]


def block_tariff(start: pd.Timestamp, end: pd.Timestamp, peak: float, off: float, flat: float) -> pd.DataFrame:
    rows = []
    for d in pd.date_range(start, end - pd.Timedelta(days=1), freq="D"):
        if d.dayofweek >= 5:
            rows.append((d, d + pd.Timedelta(days=1), "flat", flat))
        else:
            rows += [(d, d + pd.Timedelta(hours=7), "off_peak", off),
                     (d + pd.Timedelta(hours=7), d + pd.Timedelta(hours=22), "peak", peak),
                     (d + pd.Timedelta(hours=22), d + pd.Timedelta(days=1), "off_peak", off)]
    t = pd.DataFrame(rows, columns=["valid_from", "valid_to", "period", "price_per_kwh"])
    for c in ("valid_from", "valid_to"):
        t[c] = t[c].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", default=str(ROOT / "data" / "history"))
    ap.add_argument("--out", default=str(ROOT / "data" / "demo_new_quarter"))
    ap.add_argument("--volume", type=float, default=1.04)
    ap.add_argument("--leak-kw", type=float, default=60.0)
    ap.add_argument("--leak-fixed-on", default=None, help="date the leak is repaired, e.g. 2026-11-01")
    ap.add_argument("--days", type=int, default=None, help="write only the first N days (a quarter in progress)")
    a = ap.parse_args()
    hist, out = Path(a.history), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    end = START + pd.Timedelta(days=a.days) if a.days else END

    # meter: same quarter last year, 52 weeks later
    m = pd.read_csv(first(hist, "meter_15min"))
    m["ts"] = pd.to_datetime(m.ts) + SHIFT
    m = m[(m.ts >= START) & (m.ts < end)].drop_duplicates(["meter_id", "ts"], keep="last")
    m = m[m.kwh >= 0].copy()
    leak_until = pd.Timestamp(a.leak_fixed_on) if a.leak_fixed_on else end
    leaking = (m.ts < leak_until) & m.meter_id.isin(["compressors", "main"])
    m.loc[leaking, "kwh"] += a.leak_kw / 4
    m["kwh"] = m.kwh.round(3)
    m = m.sort_values(["ts", "meter_id"])
    m["ts"] = m.ts.dt.strftime("%Y-%m-%dT%H:%M:%S")
    m.to_csv(out / "meter_15min_2026q4.csv", index=False)

    # production and weather: same days last year
    p = pd.read_csv(first(hist, "production"), dtype={"is_holiday": str})
    p["date"] = pd.to_datetime(p.date) + SHIFT
    p = p[(p.date >= START) & (p.date < end)].drop_duplicates(["date", "product"], keep="last").copy()
    p["units"] = np.round(p.units * a.volume).astype(int)
    p["date"] = p.date.dt.strftime("%Y-%m-%d")
    p.to_csv(out / "production_2026q4.csv", index=False)

    w = pd.read_csv(first(hist, "weather"))
    w["date"] = pd.to_datetime(w.date) + SHIFT
    w = w[(w.date >= START) & (w.date < end)].drop_duplicates("date", keep="last").copy()
    w["date"] = w.date.dt.strftime("%Y-%m-%d")
    w.to_csv(out / "weather_2026q4.csv", index=False)

    # tariff: last prices in the history, same block structure
    t = pd.read_csv(first(hist, "tariff"), parse_dates=["valid_from"]).sort_values("valid_from")
    last = t.groupby("period").price_per_kwh.last().to_dict()
    tariff = block_tariff(START, end, last["peak"], last["off_peak"], last["flat"])
    tariff.to_csv(out / "tariff_2026q4.csv", index=False)

    card = {"name": "demo-new-quarter", "synthetic": True,
            "description": "Example new quarter built by scripts/make_demo_quarter.py from 2025-Q4 shifted 52 weeks. "
                           "Not real data and not a forecast.",
            "date_range": [str(START.date()), str((end - pd.Timedelta(days=1)).date())],
            "changes": {"volume_factor": a.volume, "compressor_leak_kw": a.leak_kw, "leak_fixed_on": a.leak_fixed_on,
                        "tariff": {k: float(v) for k, v in last.items()}},
            "row_counts": {"meter_15min": int(len(m)), "tariff": int(len(tariff)), "production": int(len(p)),
                           "weather": int(len(w))}}
    (out / "data_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
