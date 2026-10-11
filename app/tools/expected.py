"""Expected vs actual energy (baseline check).

The baseline learned from the history quarters (baseline.py) predicts how much energy each day
should have used given its production and weather. Comparing that with the metered energy shows
whether the current quarter is above or below normal, month by month, before any bill split.

Works on a complete quarter and on a partial one (a quarter still in progress): only days with
real meter readings, production and weather are compared.
"""
from __future__ import annotations

import numpy as np

from ..cleaning import period_status
from .baseline import bootstrap_betas, predict_daily, totals
from .bridge import expected_vs_actual, fit_for


def usable_days(ds, period: str):
    d = ds.daily[ds.daily.quarter == period]
    if period_status(ds).get(period, {}).get("complete"):
        return d
    return d[d.has_meter & ~d.production_missing & ~d.weather_missing & d.priced]


def baseline_check(ds, cfg: dict, new: str, ref: str | None) -> dict:
    fit = fit_for(ds, cfg, new, ref)
    if not fit["ok"]:
        return fit
    beta, use_mix = fit["beta"], fit["use_mix"]
    betas = bootstrap_betas(fit, cfg["baseline"]["bootstrap_samples"], cfg["baseline"]["seed"])
    status = period_status(ds).get(new, {})
    d = usable_days(ds, new)
    if d.empty:
        return {"ok": False, "reason": f"{new} has no day with meter, production and weather data"}
    tot = totals(ds, new, ds.products, d)
    cur = expected_vs_actual(beta, float(d.kwh_main.sum()), tot, betas)
    cur["days"] = int(len(d))

    exp_day = predict_daily(beta, d, ds.products, use_mix)
    months = []
    for m, idx in d.groupby(d.index.to_period("M")).groups.items():
        a, e = float(d.loc[idx, "kwh_main"].sum()), float(exp_day.loc[idx].sum())
        months.append({"month": str(m), "days": int(len(idx)), "actual_kwh": round(a), "expected_kwh": round(e),
                       "gap_kwh": round(a - e), "gap_pct": round((a - e) / e * 100, 2) if e else None})
    gap = d.kwh_main - exp_day
    weekly = gap.groupby(((d.index - d.index.min()).days // 7)).sum()
    worst = d.index.min() + np.timedelta64(int(weekly.idxmax()) * 7, "D") if len(weekly) else None

    # what the expected energy is made of
    units = sum(beta[c] * tot.get(c, 0.0) for c in beta if c.startswith("units_"))
    parts = {"base_load_kwh": round(beta["const"] * tot["const"]), "working_days_kwh": round(beta["working"] * tot["working"]),
             "production_kwh": round(units), "weather_kwh": round(beta["cdd"] * tot["cdd"] + beta["hdd"] * tot["hdd"])}

    out = {"ok": True, "period": new, "compare_to": ref, "partial": not status.get("complete", False),
           "days_in_quarter": status.get("days_in_quarter"), "new": cur, "months": months, "expected_breakdown": parts,
           "largest_weekly_gap": ({"week_starting": worst.strftime("%Y-%m-%d"), "gap_kwh": round(float(weekly.max()))}
                                  if worst is not None else None),
           "baseline": {"model": "linear regression on daily main-meter kWh", "fitted_on": ", ".join(fit["quarters"]),
                        "history_quarters": len(fit["quarters"]), "days_used": fit["n_days"], "r2": round(fit["r2"], 3),
                        "cv_rmse_pct": fit["cv_rmse_pct"], "nmbe_pct": fit["nmbe_pct"],
                        "kwh_per_unit": ({p: round(beta[f"units_{p}"], 3) for p in ds.products} if use_mix
                                         else {"all_products": round(beta["units_total"], 3)}),
                        "base_kwh_per_day": round(beta["const"]), "working_day_kwh": round(beta["working"]),
                        "kwh_per_cooling_degree_day": round(beta["cdd"], 1),
                        "kwh_per_heating_degree_day": round(beta["hdd"], 1)},
           "limitations": list(fit["warnings"])}
    if ref:
        rd = usable_days(ds, ref)
        out["reference"] = expected_vs_actual(beta, float(rd.kwh_main.sum()), totals(ds, ref, ds.products, rd), betas)
        out["reference"]["days"] = int(len(rd))
    if out["partial"]:
        out["limitations"].append(f"{new} is incomplete: {cur['days']} of {status.get('days_in_quarter')} days have meter, "
                                  f"production and weather data, so only those days are compared")
    lo, hi = cur.get("expected_kwh_range", [None, None])
    out["verdict"] = ("above" if lo is not None and cur["actual_kwh"] > hi else
                      "below" if lo is not None and cur["actual_kwh"] < lo else "within")
    return out
