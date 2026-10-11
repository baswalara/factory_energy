"""Leak and fault detection (R7). Transparent rules, no black-box anomaly model.

A sub-meter is flagged only when its load in idle hours (production lines stopped) is
well above its own pattern in the HISTORY quarters (the same window the baseline learns from,
up to 4 quarters) AFTER adjusting for weather. A year of history includes summer and winter, so a
hot spell on the cooling meter, or a furnace that holds temperature in winter, is not called waste.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bridge import baseline_quarters


def action_cfg(cfg: dict, meter: str) -> dict:
    for a in cfg["actions"]:
        if a["match"] in meter.lower():
            return a
    return cfg["default_action"]


def _daily_idle(ds, meter: str) -> pd.DataFrame:
    """Per day: average kW of this meter while lines are stopped, and while they run."""
    day = ds.local.normalize()
    idle = ~ds.productive.values
    v = ds.wide[meter].values
    g = pd.DataFrame({"day": day, "idle_kwh": np.where(idle, v, 0.0), "idle_n": idle.astype(int),
                      "work_kwh": np.where(idle, 0.0, v), "work_n": (~idle).astype(int)}).groupby("day").sum()
    g["idle_h"] = g.idle_n / 4
    g["idle_kw"] = g.idle_kwh / g.idle_h.where(g.idle_h >= 2)       # needs at least 2 idle hours in the day
    g["work_kw"] = g.work_kwh / (g.work_n / 4).where(g.work_n >= 8)
    return g.join(ds.daily[["cdd", "hdd", "quarter"]])


def find_waste(ds, cfg: dict, new: str, ref: str) -> dict:
    w = cfg["waste"]
    lines = [m for m in ds.submeters if any(m.startswith(p) for p in cfg["production_meter_prefixes"])]
    findings, checked = [], []
    window, _ = baseline_quarters(ds, cfg, new, ref)
    hist = ", ".join(window) or "none"
    for meter in [m for m in ds.submeters if m not in lines]:
        g = _daily_idle(ds, meter)
        r, n = g[g.quarter.isin(window) & g.idle_kw.notna()], g[(g.quarter == new) & g.idle_kw.notna()]
        if len(r) < 10 or len(n) < 10:
            checked.append({"meter": meter, "flagged": False, "reason": "too few idle hours to judge"})
            continue
        # history pattern: idle kW explained by weather only
        cols = ["cdd", "hdd"]
        keep = [c for c in cols if r[c].nunique() > 3]
        Xr = np.column_stack([np.ones(len(r))] + [r[c].values for c in keep])
        coef = np.linalg.lstsq(Xr, r.idle_kw.values, rcond=None)[0]
        resid_std = float(np.std(r.idle_kw.values - Xr @ coef, ddof=len(coef)))
        Xn = np.column_stack([np.ones(len(n))] + [n[c].values for c in keep])
        expected = pd.Series(Xn @ coef, index=n.index)
        excess = n.idle_kw - expected

        recent = excess.index >= excess.index.max() - pd.Timedelta(days=w["recent_window_days"] - 1)
        ex_recent, exp_recent = float(excess[recent].mean()), float(expected[recent].mean())
        threshold = max(w["min_excess_pct"] / 100 * abs(exp_recent), w["min_z"] * resid_std)
        raw_rise = float(n.idle_kw.mean() - r.idle_kw.mean())
        work_kw_new = float(n.work_kw.mean()) if n.work_kw.notna().any() else 0.0
        base = {"meter": meter,
                "idle_kw_reference": round(float(r.idle_kw.mean()), 1), "idle_kw_new": round(float(n.idle_kw.mean()), 1),
                "idle_kw_recent": round(float(n.idle_kw[recent].mean()), 1),
                "expected_idle_kw_recent": round(exp_recent, 1), "weather_adjusted": bool(keep),
                "reference_quarters": hist,
                "evidence_refs": [f"meter_15min:{meter}:{new}"] + [f"meter_15min:{meter}:{q}" for q in window]}

        flagged = ex_recent > threshold
        if flagged:
            over = (excess > threshold).values
            sustained = [i for i in range(len(over)) if over[i:i + 3].all() and len(over[i:i + 3]) == 3]
            onset = excess.index[sustained[0]] if sustained else excess.index[recent][0]
            since = excess.index >= onset
            days_since = int((excess.index.max() - onset).days) + 1
            idle_kwh = float((excess[since] * n.idle_h[since]).sum())
            findings.append({**base, "kind": "idle_growth", "excess_kw": round(ex_recent, 1),
                             "threshold_kw": round(threshold, 1),
                             "onset": onset.strftime("%Y-%m-%d"), "days_since_onset": days_since,
                             "excess_idle_kwh_in_period": round(idle_kwh),
                             "excess_all_hours_kwh_in_period": round(float(excess[since].mean()) * 24 * days_since),
                             "strength_sd": round(ex_recent / resid_std, 1) if resid_std else None})
        # standing idle load: only for asset types where a high idle load is not normal
        a = action_cfg(cfg, meter)
        idle_base = float(n.idle_kw.mean()) - (max(float(excess.mean()), 0.0) if flagged else 0.0)
        ratio = idle_base / work_kw_new if work_kw_new else 0.0
        standing_kw = idle_base - a["max_normal_idle_ratio"] * work_kw_new
        if a["max_normal_idle_ratio"] < 1.0 and ratio > a["max_normal_idle_ratio"] and standing_kw > 0:
            findings.append({**base, "kind": "standing_idle", "excess_kw": round(standing_kw, 1),
                             "idle_vs_working_pct": round(ratio * 100, 1),
                             "normal_idle_vs_working_pct": round(a["max_normal_idle_ratio"] * 100, 1),
                             "avg_working_kw": round(work_kw_new, 1)})
        if not flagged:
            if raw_rise > threshold and keep:
                reason = "idle load rose, but the weather explains it"
            else:
                reason = "idle load is in line with its history"
            checked.append({**base, "flagged": False, "reason": reason, "raw_rise_kw": round(raw_rise, 1)})
        else:
            checked.append({**base, "flagged": True, "reason": "idle load rose beyond its weather-adjusted reference"})

    return {"period": new, "compare_to": ref, "history": window, "findings": findings, "meters_checked": checked,
            "rule": (f"flag when idle-hour load in the last {w['recent_window_days']} days exceeds the weather-adjusted "
                     f"pattern of {hist} by more than {w['min_excess_pct']}% and {w['min_z']} standard deviations")}


def cross_check(bridge: dict, waste: dict) -> dict:
    """Does a named meter back up the waste slice of the bill? If not, it is unexplained, not waste."""
    slice_ = next(r for r in bridge["bridge"] if r["cause"] == "waste")
    growth = [f for f in waste["findings"] if f["kind"] == "idle_growth"]
    low = sum(f["excess_idle_kwh_in_period"] for f in growth)
    high = sum(f["excess_all_hours_kwh_in_period"] for f in growth)
    kwh = slice_["kwh"]
    if not growth:
        status = ("no_meter_evidence" if slice_["range"][0] > 0 else "no_waste_growth_found")
    elif kwh <= 0:
        status = "meter_pattern_without_bill_leftover"
    else:
        status = "supported_by_meters"
    return {"waste_slice_kwh": kwh, "waste_slice_pct_points": slice_["pct_points"],
            "meter_backed_kwh_idle_hours_only": round(low), "meter_backed_kwh_if_all_hours": round(high),
            "unexplained_kwh": round(max(kwh - high, 0)) if kwh > 0 else 0,
            "meters": [f["meter"] for f in growth], "status": status}
