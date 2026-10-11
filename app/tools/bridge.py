"""Bill decomposition (R3). Order from Annex A: tariff, volume, product mix, weather, then waste.

The current quarter (new) is compared with the latest past quarter (reference). The baseline that
values volume, mix and weather is learned from the history: up to 4 complete quarters before the
current one (see baseline.py).

With B = bill, E = kWh, p0 = reference average price (B0 / E0):

    B1 - B0 = (B1 - E1 x p0)          tariff slice: new energy at new prices minus new energy at the old price
            + (E1 - E0) x p0          energy change at the old price, split into:
                volume   = calendar days + extra units at the reference mix
                mix      = this quarter's units at this quarter's mix minus the same units at the reference mix
                weather  = change in degree-days
                waste    = whatever the baseline cannot explain (growth in unexplained energy)

Each slice is divided by B0, so the five percentage points sum to the total change by construction.

Because the baseline is linear, volume + mix + weather = expected(new) - expected(reference), so
    waste = (actual_new - expected_new) - (actual_ref - expected_ref)
i.e. the growth in energy the baseline cannot explain. Both expected-vs-actual gaps are reported.
"""
from __future__ import annotations

import numpy as np

from ..cleaning import available_periods, history_window
from .baseline import bootstrap_betas, fit_baseline, predict_total, totals
from .billing import bill

CAUSES = ["tariff", "volume", "product_mix", "weather", "waste"]


def energy_slices(beta: dict, t0: dict, t1: dict, products: list[str], use_mix: bool, mix_first: bool = False) -> dict:
    """kWh explained by volume, mix and weather when inputs move from reference (t0) to new (t1)."""
    calendar = beta["const"] * (t1["const"] - t0["const"]) + beta["working"] * (t1["working"] - t0["working"])
    V0, V1 = t0["units_total"], t1["units_total"]
    if use_mix and V0 > 0 and V1 > 0:
        r = np.array([beta[f"units_{p}"] for p in products])
        s0 = np.array([t0[f"units_{p}"] for p in products]) / V0
        s1 = np.array([t1[f"units_{p}"] for p in products]) / V1
        if mix_first:
            mix, volume = V0 * float(r @ (s1 - s0)), (V1 - V0) * float(r @ s1)
        else:
            volume, mix = (V1 - V0) * float(r @ s0), V1 * float(r @ (s1 - s0))
    elif use_mix:    # one quarter made nothing: shares are undefined, so all of it is volume
        volume, mix = sum(beta[f"units_{p}"] * (t1[f"units_{p}"] - t0[f"units_{p}"]) for p in products), 0.0
    else:
        volume, mix = beta["units_total"] * (V1 - V0), 0.0
    weather = beta["cdd"] * (t1["cdd"] - t0["cdd"]) + beta["hdd"] * (t1["hdd"] - t0["hdd"])
    return {"volume": calendar + volume, "product_mix": mix, "weather": weather}


def _round_to_total(values: dict, total: float, nd: int = 2) -> dict:
    """Round each slice; put the rounding remainder (at most a few hundredths) on the largest slice,
    so the printed numbers add up exactly to the printed total."""
    out = {k: round(float(v), nd) for k, v in values.items()}
    gap = round(round(float(total), nd) - sum(out.values()), nd)
    if gap:
        big = max(out, key=lambda k: abs(out[k]))
        out[big] = round(out[big] + gap, nd)
    return out


def baseline_quarters(ds, cfg: dict, new: str, ref: str | None) -> tuple[list[str], list[str]]:
    """History the baseline learns from, plus limitation notes. Falls back to the comparison quarter
    only when nothing earlier than `new` is loaded (for example when comparing backwards)."""
    want = cfg["baseline"].get("history_quarters", 4)
    window = history_window(ds, new, want)
    notes = []
    if not window and ref and ref in available_periods(ds):
        window = [ref]
        notes.append(f"No complete quarter before {new} is loaded, so the baseline is learned from {ref}, "
                     f"the quarter it is compared with")
    elif 0 < len(window) < want:
        notes.append(f"The baseline is learned from {len(window)} quarter(s) ({', '.join(window)}); {want} are configured, "
                     f"so seasonal effects are learned from less history than intended")
    return window, notes


def fit_for(ds, cfg: dict, new: str, ref: str | None) -> dict:
    window, notes = baseline_quarters(ds, cfg, new, ref)
    fit = fit_baseline(ds, cfg, window)
    if fit["ok"]:
        fit["warnings"] = notes + fit["warnings"]
    return fit


def expected_vs_actual(beta: dict, actual: float, tot: dict, runs: list[dict] | None = None) -> dict:
    exp = predict_total(beta, tot)
    out = {"actual_kwh": round(actual), "expected_kwh": round(exp), "gap_kwh": round(actual - exp),
           "gap_pct": round((actual - exp) / exp * 100, 2) if exp else None}
    if runs:
        sims = np.array([predict_total(b, tot) for b in runs])
        out["expected_kwh_range"] = [round(float(np.percentile(sims, 5))), round(float(np.percentile(sims, 95)))]
    return out


def decompose(ds, cfg: dict, new: str, ref: str, price=None, scenario: str | None = None) -> dict:
    """Full bridge for new vs ref. `price` replaces the interval prices of the new quarter (what-if)."""
    b0, b1 = bill(ds, ref), bill(ds, new, price)
    B0, B1, E0, E1, p0 = b0["bill"], b1["bill"], b0["kwh"], b1["kwh"], b0["avg_price"]
    if B0 <= 0 or E0 <= 0:
        return {"ok": False, "reason": f"no billed energy in {ref}"}
    fit = fit_for(ds, cfg, new, ref)
    if not fit["ok"]:
        return fit

    use_mix = fit["use_mix"]
    t0, t1 = totals(ds, ref, ds.products), totals(ds, new, ds.products)
    pp = lambda kwh: kwh * p0 / B0 * 100            # kWh at the old price -> percentage points of the old bill

    def all_slices(beta: dict, mix_first: bool = False) -> dict:
        s = energy_slices(beta, t0, t1, ds.products, use_mix, mix_first)
        s["waste"] = (E1 - E0) - sum(s.values())
        return s

    point = all_slices(fit["beta"])
    kwh = {"tariff": 0.0, **point}
    money = {"tariff": B1 - E1 * p0, **{k: v * p0 for k, v in point.items()}}
    pct = {k: v / B0 * 100 for k, v in money.items()}
    total_pct = (B1 - B0) / B0 * 100

    # ranges: resampled baseline fits x both orders of volume and mix
    betas = bootstrap_betas(fit, cfg["baseline"]["bootstrap_samples"], cfg["baseline"]["seed"])
    runs = [all_slices(b, mf) for b in betas for mf in (False, True)] + [all_slices(fit["beta"], True)]
    rng = {}
    for k in point:
        vals = np.array([pp(r[k]) for r in runs])
        rng[k] = [min(float(np.percentile(vals, 5)), pct[k]), max(float(np.percentile(vals, 95)), pct[k])]
    # tariff range: price change valued on the new quarter's energy (ours) or the reference quarter's
    alt = E0 * (b1["avg_price"] - p0) / B0 * 100
    rng["tariff"] = [min(pct["tariff"], alt), max(pct["tariff"], alt)]

    shown = _round_to_total(pct, total_pct)
    bridge = [{"cause": c, "pct_points": shown[c],
               "range": [round(float(min(rng[c][0], shown[c])), 2), round(float(max(rng[c][1], shown[c])), 2)],
               "kwh": int(round(kwh[c])), "money": round(float(money[c]), 2)} for c in CAUSES]

    rates = ({p: round(fit["beta"][f"units_{p}"], 3) for p in ds.products} if use_mix
             else {"all_products": round(fit["beta"]["units_total"], 3)})
    limitations = list(fit["warnings"])
    if not use_mix:
        limitations.append("Baseline run without product mix: a mix shift cannot be separated and lands in waste")
    hist = [totals(ds, q, ds.products) for q in fit["quarters"]]
    for c, word in (("cdd", "hotter"), ("hdd", "colder")):
        most = max((h[c] / h["const"] for h in hist if h["const"]), default=0.0)
        if t1[c] > 0 and t1[c] / t1["const"] > 1.2 * most:
            limitations.append(f"{new} was {word} than any quarter the baseline learned from ({', '.join(fit['quarters'])}): "
                               f"the weather rate is stretched beyond the days it was learned on")
    new_only = [p for p in ds.products if t1[f"units_{p}"] > 0 and all(h[f"units_{p}"] == 0 for h in hist)]
    if new_only:
        limitations.append(f"Product(s) {new_only} were not made in the baseline history; their energy cannot be learned "
                           f"and falls into waste")
    expected = {"baseline_quarters": fit["quarters"],
                "reference": expected_vs_actual(fit["beta"], E0, t0, betas),
                "new": expected_vs_actual(fit["beta"], E1, t1, betas)}

    return {
        "ok": True, "period": new, "compare_to": ref, "scenario": scenario,
        "bill": {"reference": {k: round(v, 4 if k == "avg_price" else 2) for k, v in b0.items() if k != "period"},
                 "new": {k: round(v, 4 if k == "avg_price" else 2) for k, v in b1.items() if k != "period"},
                 "change": round(B1 - B0, 2), "change_pct": round(total_pct, 2),
                 "kwh_change": round(E1 - E0), "kwh_change_pct": round((E1 - E0) / E0 * 100, 2),
                 "avg_price_change_pct": round((b1["avg_price"] - p0) / p0 * 100, 2)},
        "bridge": bridge, "total_pct_points": round(sum(r["pct_points"] for r in bridge), 2),
        "expected_vs_actual": expected,
        "baseline": {"model": "linear regression on daily main-meter kWh", "fitted_on": ", ".join(fit["quarters"]),
                     "history_quarters": len(fit["quarters"]),
                     "r2": round(fit["r2"], 3), "cv_rmse_pct": fit["cv_rmse_pct"], "nmbe_pct": fit["nmbe_pct"],
                     "days_used": fit["n_days"], "days_excluded": fit["days_excluded"],
                     "kwh_per_unit": rates, "base_kwh_per_day": round(fit["beta"]["const"]),
                     "working_day_kwh": round(fit["beta"]["working"]),
                     "kwh_per_cooling_degree_day": round(fit["beta"]["cdd"], 1),
                     "kwh_per_heating_degree_day": round(fit["beta"]["hdd"], 1),
                     "expected_kwh_new": round(predict_total(fit["beta"], t1)),
                     "expected_kwh_reference": round(predict_total(fit["beta"], t0))},
        "inputs": {"units_reference": round(t0["units_total"]), "units_new": round(t1["units_total"]),
                   "working_days_reference": int(t0["working"]), "working_days_new": int(t1["working"]),
                   "cdd_reference": round(t0["cdd"], 1), "cdd_new": round(t1["cdd"], 1)},
        "limitations": limitations,
    }
