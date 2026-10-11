"""Energy by sub-meter (R5) and idle load (R6)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def period_mask(ds, period: str) -> np.ndarray:
    return (ds.quarter_of_interval() == period).values


def asset_breakdown(ds, period: str) -> dict:
    m = period_mask(ds, period)
    main = float(ds.wide["main"].values[m].sum())
    rows = [{"meter": s, "kwh": round(float(ds.wide[s].values[m].sum())),
             "share_of_main_pct": round(float(ds.wide[s].values[m].sum()) / main * 100, 1)} for s in ds.submeters]
    rows.sort(key=lambda r: -r["kwh"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    sub_total = sum(r["kwh"] for r in rows)
    gap_pct = round((main - sub_total) / main * 100, 2) if main else 0.0
    return {"period": period, "main_kwh": round(main), "assets": rows,
            "submeter_total_kwh": round(sub_total), "unmetered_kwh": round(main - sub_total),
            "unmetered_pct_of_main": gap_pct, "tolerance_pct": ds.cfg["submeter_tolerance_pct"],
            "within_tolerance": abs(gap_pct) <= ds.cfg["submeter_tolerance_pct"]}


def _hour_ranges(hours: list[int]) -> str:
    """[22,23,0,1,...,5] -> '22:00-06:00'."""
    if not hours:
        return "none"
    if len(hours) == 24:
        return "all day"
    flag = [h in hours for h in range(24)]
    starts = [h for h in range(24) if flag[h] and not flag[h - 1]]
    out = []
    for s in starts:
        e = s
        while flag[(e + 1) % 24] and (e + 1) % 24 != s:
            e += 1
        out.append(f"{s:02d}:00-{(e + 1) % 24:02d}:00")
    return ", ".join(out)


def idle_load(ds, period: str) -> dict:
    """Energy used while the production lines are not running, by sub-meter."""
    m = period_mask(ds, period)
    idle = m & ~ds.productive.values
    work = m & ds.productive.values
    idle_h, work_h = idle.sum() / 4, work.sum() / 4
    main = ds.wide["main"].values

    day = ds.local.normalize()
    working_day = ds.daily.working.reindex(day).fillna(False).values
    on_work_days = m & working_day
    hour = np.asarray(ds.local.hour)
    idle_share = pd.Series(~ds.productive.values[on_work_days]).groupby(hour[on_work_days]).mean()
    window = _hour_ranges([int(h) for h, v in idle_share.items() if v >= 0.5])
    d = ds.daily[ds.daily.quarter == period]

    rows = []
    for s in ds.submeters:
        v = ds.wide[s].values
        idle_kwh, total = float(v[idle].sum()), float(v[m].sum())
        idle_kw = idle_kwh / idle_h if idle_h else 0.0
        work_kw = float(v[work].sum()) / work_h if work_h else 0.0
        rows.append({"meter": s, "idle_kwh": round(idle_kwh),
                     "idle_share_of_meter_pct": round(idle_kwh / total * 100, 1) if total else 0.0,
                     "avg_idle_kw": round(idle_kw, 1), "avg_working_kw": round(work_kw, 1),
                     "idle_vs_working_pct": round(idle_kw / work_kw * 100, 1) if work_kw else None})
    rows.sort(key=lambda r: -r["idle_kwh"])
    main_idle = float(main[idle].sum())
    return {"period": period, "idle_definition": ds.log.get("idle_basis", ""),
            "idle_hours": round(float(idle_h), 1), "working_hours": round(float(work_h), 1),
            "idle_window_on_working_days": window,
            "non_working_days": int((~d.working).sum()), "working_days": int(d.working.sum()),
            "main_idle_kwh": round(main_idle),
            "main_idle_share_pct": round(main_idle / float(main[m].sum()) * 100, 1) if m.any() else 0.0,
            "assets": rows}
