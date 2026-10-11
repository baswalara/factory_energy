"""Savings (R8) and ranking (R9). Every figure is a low-high range (Annex A).

kWh   = avoidable kW x hours per year x recoverable share (share range from config.yaml)
money = kWh x current price in the hours where the waste happens
CO2e  = kWh x emission factor
payback = action cost / yearly money saving (cost range from config.yaml)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .assets import period_mask
from .waste import action_cfg

EASE_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def current_prices(ds, cfg: dict, period: str) -> dict:
    """Average price over the last days of the period, i.e. after any tariff change."""
    m = period_mask(ds, period)
    last = ds.local[m].max()
    recent = m & np.asarray(ds.local > last - pd.Timedelta(days=cfg["waste"]["recent_window_days"]))
    idle = recent & ~ds.productive.values
    p = ds.price.values
    return {"all_hours": float(p[recent].mean()), "idle_hours": float(p[idle].mean()) if idle.any() else float(p[recent].mean()),
            "idle_hours_per_year": float(idle.sum() / recent.sum() * 8760)}


def recommend(ds, cfg: dict, waste: dict, check: dict | None = None) -> dict:
    period = waste["period"]
    pr = current_prices(ds, cfg, period)
    ef = cfg["emission_factor_kg_per_kwh"]
    recs = []
    for f in waste["findings"]:
        a = action_cfg(cfg, f["meter"])
        lo_share, hi_share = a["recoverable_share"]
        kw = f["excess_kw"]
        if f["kind"] == "idle_growth":
            # low: the excess only runs in idle hours. high: it runs all hours (a leak does not stop for production)
            kwh = [kw * pr["idle_hours_per_year"] * lo_share, kw * 8760 * hi_share]
            money = [kwh[0] * pr["idle_hours"], kwh[1] * pr["all_hours"]]
            action = a["growth_action"].format(meter=f["meter"])
            rationale = (f"Idle-hour load on {f['meter']} is {f['idle_kw_recent']} kW against {f['expected_idle_kw_recent']} kW "
                         f"expected from its history ({f['reference_quarters']}) and weather: {f['excess_kw']} kW extra since {f['onset']}.")
            backed = bool(check and check["status"] == "supported_by_meters")
            confidence = "high" if backed and (f["strength_sd"] or 0) >= 2 * cfg["waste"]["min_z"] else "medium"
        else:
            kwh = [kw * pr["idle_hours_per_year"] * lo_share, kw * pr["idle_hours_per_year"] * hi_share]
            money = [kwh[0] * pr["idle_hours"], kwh[1] * pr["idle_hours"]]
            action = a["standing_action"].format(meter=f["meter"])
            rationale = (f"{f['meter']} draws {f['idle_vs_working_pct']}% of its working load while lines are stopped; "
                         f"up to {f['normal_idle_vs_working_pct']}% is assumed normal, leaving {f['excess_kw']} kW avoidable. "
                         f"Present in both quarters, so it is not part of the bill increase.")
            confidence = "low"
        co2 = [k * ef / 1000 for k in kwh]
        cost = a["cost"]
        payback = [cost[0] / money[1] * 12, cost[1] / money[0] * 12] if money[0] > 0 else [None, None]
        rnd = lambda xs, nd=0: [round(x, nd) if nd else int(round(x)) for x in xs]
        rec = {"action": action,
               "saving_kwh_per_year": rnd(kwh), "saving_money_per_year": rnd(money),
               "saving_co2e_t_per_year": rnd(co2, 1),
               "payback_months": rnd(payback, 1) if payback[0] is not None else None,
               "ease": a["ease"], "rationale": rationale, "evidence_refs": f["evidence_refs"],
               "confidence": confidence, "requires_approval": True,
               "finding": f["kind"], "meter": f["meter"], "avoidable_kw": kw,
               "assumed_cost": cost, "assumed_recoverable_share": a["recoverable_share"]}
        rec["estimated_impact"] = {"kwh_per_year": rec["saving_kwh_per_year"], "money_per_year": rec["saving_money_per_year"],
                                   "co2e_t_per_year": rec["saving_co2e_t_per_year"]}
        recs.append(rec)

    mid = lambda r: sum(r["payback_months"]) / 2 if r["payback_months"] else 1e9
    recs.sort(key=lambda r: (mid(r), EASE_ORDER.get(r["ease"], 1)))
    for i, r in enumerate(recs, 1):
        r["rank"] = i
    money_label = cfg.get("currency_label") or "the tariff's money unit"
    return {
        "period": period, "recommendations": recs,
        "prices_used": {"all_hours_avg": round(pr["all_hours"], 4), "idle_hours_avg": round(pr["idle_hours"], 4),
                        "idle_hours_per_year": round(pr["idle_hours_per_year"])},
        "ranking_rule": "shortest payback (midpoint of the range) first, then ease",
        "assumptions": [
            f"Emission factor {ef} kg CO2e per kWh (config value; the data files do not contain one)",
            f"Action costs, ease and recoverable shares are our stated assumptions in config.yaml, in {money_label}; no file provides them",
            f"Yearly savings scale the pattern of the last {cfg['waste']['recent_window_days']} days of {period} to a full year",
            "Savings are priced at the tariff in force at the end of the period",
        ],
    }
