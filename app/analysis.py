"""Runs the code tools for one labelled question and returns facts. No LLM in this file.

Every handler returns (answer_type, facts, steps). `facts` is the only thing the brief
writer sees, and the only source of numbers the number check accepts.
"""
from __future__ import annotations

import re

from .cleaning import Dataset, available_periods, describe_periods, partial_periods, parse_period, period_status
from .intents import find_meter
from .tools.assets import asset_breakdown, idle_load, period_mask
from .tools.billing import bill, prices_as_of, reprice_at
from .tools.bridge import baseline_quarters, decompose
from .tools.expected import baseline_check
from .tools.savings import recommend
from .tools.waste import cross_check, find_waste


NAMED_ASSET = re.compile(r"\b(?:the|our|my)\s+([a-z][a-z0-9_ -]{1,30}?)\s+(?:use|uses|used|consume|consumes|consumed|draw|draws|waste|wastes)\b", re.I)
GENERIC_ASSETS = {"factory", "plant", "site", "line", "lines", "machine", "machines", "asset", "assets", "meter", "meters",
                  "equipment", "building", "company", "production", "production lines", "whole factory", "main meter"}


def tariff_assumption(ds: Dataset) -> str:
    c, mode = ds.cfg, ds.log.get("pricing", {}).get("mode")
    if mode == "block":
        return "Each 15-minute reading is priced by the tariff time block it falls in (the tariff file gives dated peak, off-peak and flat blocks)"
    if mode == "flat":
        return "Each 15-minute reading is priced at the flat tariff in force at that moment"
    return (f"Peak price applies {c['peak_hours']['start']:02d}:00-{c['peak_hours']['end']:02d}:00 local time for tariff rows "
            f"that name a period but not its hours")


def base_assumptions(ds: Dataset) -> list[str]:
    c = ds.cfg
    return [
        "The main meter is the billed total; sub-meters are used for attribution only",
        tariff_assumption(ds),
        f"Factory time zone {c['factory_timezone']} is used to match meter readings to daily production and weather",
        f"Idle means the production lines are not running ({ds.log.get('idle_basis', 'n/a')})",
    ]


def baseline_assumption(ds: Dataset, new: str, ref: str | None) -> str:
    window, _ = baseline_quarters(ds, ds.cfg, new, ref)
    return (f"Expected energy (the baseline) is a linear regression of daily main-meter kWh on working days, units per "
            f"product, cooling and heating degree-days, learned from the history quarters {', '.join(window) or 'none'} "
            f"(never from {new} itself)")


def resolve_periods(ds: Dataset, question: str, period: str | None, compare_to: str | None, allow_partial: bool = False) -> dict:
    """Which quarters to compare: the current quarter (latest loaded unless named) against the latest past quarter.
    Never guesses: unknown text asks back, unloaded or incomplete quarters say not_in_data.
    allow_partial: the current quarter may be incomplete (expected-vs-actual check only)."""
    complete = available_periods(ds)
    partial = partial_periods(ds) if allow_partial else []
    have = sorted(set(complete) | set(partial))
    if len(have) == 0:
        status = describe_periods(ds) if ds.ok else ""
        return {"exit": "not_in_data", "message": "No quarter has enough meter, production and weather data to analyse yet. "
                                                  f"{status} Load data with POST /ingest or scripts/load_data.py."}
    asked = {}
    for name, text in (("period", period), ("compare_to", compare_to)):
        if text:
            p = parse_period(text)
            if not p:
                return {"exit": "clarification_needed",
                        "message": f"I could not read '{text}' as a quarter. Use a label like {have[-1]}. Loaded quarters: {', '.join(have)}."}
            asked[name] = p
    in_q = re.findall(r"(?:20\d{2}\s*[-_/ ]?\s*Q\s*[1-4]|Q\s*[1-4]\s*[-_/ ]?\s*20\d{2})", question, re.I)
    named = [parse_period(x) for x in in_q]
    for bare in re.findall(r"(?<![\w-])Q([1-4])\b(?!\s*[-_/ ]?\s*20\d{2})", question, re.I):
        match = [h for h in have if h.endswith(f"Q{bare}")]
        named.append(match[0] if len(match) == 1 else f"Q{bare}")
    named = list(dict.fromkeys(named))
    if "period" not in asked and named:
        asked["period"] = max(named) if len(named) > 1 else named[0]
        if len(named) > 1 and "compare_to" not in asked:
            asked["compare_to"] = min(named)
    st = period_status(ds)
    for name, p in asked.items():
        ok = p in complete or (name == "period" and p in partial)
        if not ok:
            if p in st:
                r = st[p]
                hint = (" I can still compare its energy with the baseline for the days loaded: ask "
                        f"'Is energy above the baseline in {p}?'" if r["partial"] else "")
                return {"exit": "not_in_data",
                        "message": f"{p} is incomplete: {r['usable_days']} of {r['days_in_quarter']} days have meter, production, "
                                   f"weather and tariff data" + (f" (missing: {', '.join(r['missing'])})" if r["missing"] else "")
                                   + f". Comparing bills needs at least {ds.cfg['min_quarter_coverage']:.0%} of the quarter.{hint}"}
            return {"exit": "not_in_data",
                    "message": f"{p} is not in the loaded data. {describe_periods(ds)} I will not estimate a quarter that is not loaded."}
    new = asked.get("period", have[-1])
    if "compare_to" in asked:
        ref = asked["compare_to"]
    else:
        earlier = [h for h in complete if h < new]
        ref = earlier[-1] if earlier else None
    if ref == new:
        return {"exit": "clarification_needed", "message": f"Both periods are {new}. Which quarter should it be compared with? Loaded: {', '.join(have)}."}
    note = [] if period and compare_to else [f"Periods used: {new} compared with {ref} (current quarter = latest loaded, against the latest "
                                             f"past quarter, unless the request names them)"]
    if new in partial:
        note.append(f"{new} is still incomplete; only its loaded days are compared with the baseline")
    skipped = [q for q in partial_periods(ds) if q > new and q not in partial]
    if skipped and not period:
        note.append(f"{', '.join(skipped)} is loaded but incomplete, so it is not used for bill comparisons yet "
                    f"({describe_periods(ds)})")
    return {"new": new, "ref": ref, "available": complete, "partial": partial, "note": note}


def _step(tool: str, output: dict) -> dict:
    return {"agent": "analyst", "tool": tool, "output": output}


def _bridge_bundle(ds, cfg, new, ref, price=None, scenario=None):
    """Bridge + waste evidence + cross-check + actions: the full chain behind any 'why' or 'what to do' answer."""
    b = decompose(ds, cfg, new, ref, price, scenario)
    if not b.get("ok"):
        return b, None, None, None, []
    w = find_waste(ds, cfg, new, ref)
    chk = cross_check(b, w)
    rec = recommend(ds, cfg, w, chk)
    steps = [
        _step("build_baseline", {"fitted_on": b["baseline"]["fitted_on"], "r2": b["baseline"]["r2"],
                                 "cv_rmse_pct": b["baseline"]["cv_rmse_pct"], "days_used": b["baseline"]["days_used"],
                                 "kwh_per_unit": b["baseline"]["kwh_per_unit"]}),
        _step("expected_vs_actual", b["expected_vs_actual"]),
        _step("decompose_change", {**{r["cause"]: r["pct_points"] for r in b["bridge"]}, "total": b["bill"]["change_pct"]}),
        _step("find_idle_load", {"flagged": [{k: f.get(k) for k in ("meter", "kind", "excess_kw", "onset")} for f in w["findings"]],
                                 "cleared": [c["meter"] + ": " + c["reason"] for c in w["meters_checked"] if not c["flagged"]]}),
        _step("cross_check_waste", chk),
        _step("estimate_savings", {"actions": [{"action": r["action"], "kwh": r["saving_kwh_per_year"],
                                                "payback_months": r["payback_months"]} for r in rec["recommendations"]]}),
    ]
    return b, w, chk, rec, steps


def _bridge_facts(b, w, chk) -> dict:
    waste_pp = next(r["pct_points"] for r in b["bridge"] if r["cause"] == "waste")
    return {"period": b["period"], "compare_to": b["compare_to"], "bill": b["bill"], "bridge": b["bridge"],
            "explained_pct_points": round(b["total_pct_points"] - waste_pp, 2), "baseline": b["baseline"],
            "expected_vs_actual": b["expected_vs_actual"],
            "inputs": b["inputs"], "waste_check": chk,
            "evidence": [f for f in w["findings"] if f["kind"] == "idle_growth"],
            "meters_cleared": [{"meter": c["meter"], "reason": c["reason"]} for c in w["meters_checked"] if not c["flagged"]]}


def _waste_limits(chk: dict) -> list[str]:
    out = ["The waste slice is what the baseline cannot explain, so it also contains baseline error; see its range"]
    if chk["status"] == "supported_by_meters":
        out.append("Meter evidence covers idle hours only; whether the excess also runs during production is assumed in the high end of the savings range")
    return out


def run(intent: str, params: dict, ds: Dataset, cfg: dict, store, new: str, ref: str | None, question: str) -> dict:
    """Returns {answer_type, facts, steps, bridge, recommendations, assumptions, limitations} or an early exit."""
    out = {"bridge": [], "recommendations": [], "assumptions": base_assumptions(ds), "limitations": [], "steps": []}
    meter = find_meter(question, ds.submeters)
    if intent == "baseline_check":
        r = baseline_check(ds, cfg, new, ref)
        if not r.get("ok"):
            return {"exit": "not_in_data", "message": r["reason"]}
        out["assumptions"].append(baseline_assumption(ds, new, ref))
        out["limitations"] += r.pop("limitations")
        out.update(answer_type="calculated", facts=r, steps=[
            _step("build_baseline", {k: r["baseline"][k] for k in ("fitted_on", "r2", "cv_rmse_pct", "nmbe_pct", "days_used")}),
            _step("expected_vs_actual", {"new": r["new"], "reference": r.get("reference"), "verdict": r["verdict"]})])
        return out
    if new in partial_periods(ds):
        return {"exit": "not_in_data", "message": f"{new} is incomplete, so only the expected-vs-actual check can run on it. "
                                                  f"{describe_periods(ds)}"}
    if intent in ("top_users", "idle_waste") and not meter and re.search(r"\b(this|that|the) (asset|machine|meter|equipment)\b", question, re.I):
        return {"exit": "clarification_needed", "message": f"Which asset do you mean? Meters in the data: {', '.join(ds.submeters)}."}
    named = NAMED_ASSET.search(question)
    if intent in ("top_users", "idle_waste") and not meter and named and named.group(1).strip().lower() not in GENERIC_ASSETS:
        return {"exit": "clarification_needed",
                "message": f"There is no meter called '{named.group(1).strip()}' in the data. Which one do you mean? "
                           f"Meters: {', '.join(ds.submeters)}."}

    need_ref = intent not in ("top_users", "idle_waste", "bill_total")
    if ref is None and need_ref:
        return {"exit": "not_in_data", "message": f"Only {new} is loaded, so there is no earlier quarter to compare with."}

    if intent == "top_users":
        a = asset_breakdown(ds, new)
        facts = {**a, "focus": next((x for x in a["assets"] if x["meter"] == meter), None)}
        out.update(answer_type="calculated", facts=facts, steps=[_step("asset_breakdown", {
            "top": [(x["meter"], x["share_of_main_pct"]) for x in a["assets"][:3]], "unmetered_pct": a["unmetered_pct_of_main"]})])
        if not a["within_tolerance"]:
            out["limitations"].append(f"Sub-meters differ from the main meter by {a['unmetered_pct_of_main']}%, above the {a['tolerance_pct']}% tolerance")
        return out

    if intent == "idle_waste":
        i = idle_load(ds, new)
        w = find_waste(ds, cfg, new, ref) if ref else {"findings": [], "meters_checked": []}
        facts = {**i, "findings": w["findings"], "focus": next((x for x in i["assets"] if x["meter"] == meter), None),
                 "meters_cleared": [{"meter": c["meter"], "reason": c["reason"]} for c in w["meters_checked"] if not c["flagged"]]}
        out.update(answer_type="calculated", facts=facts, steps=[
            _step("find_idle_load", {"main_idle_kwh": i["main_idle_kwh"], "idle_hours": i["idle_hours"],
                                     "top": [(x["meter"], x["idle_kwh"]) for x in i["assets"][:3]]}),
            _step("find_waste_patterns", {"flagged": [(f["meter"], f["kind"], f["excess_kw"]) for f in w["findings"]]})])
        out["limitations"].append("Idle energy includes loads that must stay on (for example a furnace holding temperature); only flagged meters are treated as waste")
        return out

    if intent == "bill_total":
        r = lambda b: {k: round(v, 4 if k == "avg_price" else 2) for k, v in b.items() if k != "period"}
        if ref is None:
            facts = {"period": new, "compare_to": None, "bill": {"new": r(bill(ds, new))}}
        else:
            b = decompose(ds, cfg, new, ref)
            if not b.get("ok"):
                return {"exit": "not_in_data", "message": b["reason"]}
            facts = {"period": new, "compare_to": ref, "bill": b["bill"]}
        out.update(answer_type="calculated", facts=facts,
                   steps=[_step("price_intervals", {"bill": facts["bill"]["new"]["bill"], "kwh": facts["bill"]["new"]["kwh"]})])
        return out

    # everything below needs the full chain
    price, scenario = None, None
    if intent == "what_if":
        if params.get("tariff_unchanged"):
            end_ref = ds.wide.index[period_mask(ds, ref)].max()
            old = prices_as_of(end_ref, ds.tariff)
            price = reprice_at(ds, old)
            scenario = f"tariff stays at the {ref} prices for all of {new}"
        elif "tariff_change_pct" in params:
            pct = params["tariff_change_pct"]
            price = ds.price * (1 + pct / 100)
            scenario = f"every {new} price is {abs(pct):g}% {'lower' if pct < 0 else 'higher'} than actual"
        else:
            return {"exit": "clarification_needed",
                    "message": "Which what-if do you mean? Supported: a tariff that is N% lower or higher, or a tariff that had not changed. "
                               "For the effect of fixing waste, ask what we should do first."}

    b, w, chk, rec, steps = _bridge_bundle(ds, cfg, new, ref)
    if not b.get("ok"):
        return {"exit": "not_in_data", "message": b["reason"]}
    out["limitations"] += b["limitations"] + _waste_limits(chk)
    out["assumptions"].append(baseline_assumption(ds, new, ref))
    cleaned = {m: s for m, s in ds.log["meters"].items() if s["profile_filled"] or s["interpolated"]}
    if "main" in cleaned:
        out["limitations"].append(f"Main-meter gaps were filled ({cleaned['main']['interpolated']} intervals interpolated, "
                                  f"{cleaned['main']['profile_filled']} from the typical daily profile)")
    d = ds.daily[ds.daily.quarter.isin([new, ref])]
    if d.weather_missing.any() or d.production_missing.any():
        out["limitations"].append(f"{int(d.weather_missing.sum())} day(s) have no weather row (nearest value used) and "
                                  f"{int(d.production_missing.sum())} day(s) have no production row (treated as no production)")
    out["assumptions"] += rec["assumptions"]
    out["recommendations"] = rec["recommendations"]
    facts = _bridge_facts(b, w, chk)

    if intent in ("savings", "ranking"):
        facts = {"period": new, "compare_to": ref, "recommendations": rec["recommendations"], "ranking_rule": rec["ranking_rule"],
                 "prices_used": rec["prices_used"], "waste_check": chk, "bill_change_pct": b["bill"]["change_pct"]}
        out.update(answer_type="recommendation", facts=facts, steps=steps, bridge=b["bridge"])
        return out

    if intent == "what_if":
        s = decompose(ds, cfg, new, ref, price, scenario)
        ws = find_waste(ds, cfg, new, ref)
        facts = _bridge_facts(s, ws, cross_check(s, ws))
        t = lambda x: next(r["pct_points"] for r in x["bridge"] if r["cause"] == "tariff")
        facts["what_if"] = {"scenario": scenario, "bill_actual": b["bill"]["new"]["bill"], "bill_scenario": s["bill"]["new"]["bill"],
                            "bill_difference": round(s["bill"]["new"]["bill"] - b["bill"]["new"]["bill"], 2),
                            "change_pct_actual": b["bill"]["change_pct"], "change_pct_scenario": s["bill"]["change_pct"],
                            "tariff_points_actual": t(b), "tariff_points_scenario": t(s)}
        steps.insert(2, _step("reprice_scenario", facts["what_if"]))
        out["assumptions"].append("What-if: only prices change; energy use is kept exactly as measured")
        out.update(answer_type="decomposition", facts=facts, steps=steps, bridge=s["bridge"])
        return out

    if intent == "data_change":
        prev = store.previous_bridge(new, ref, b)
        if not prev:
            return {"exit": "not_in_data", "message": "There is no earlier result to compare with: no data has been corrected or added "
                                                      f"since the first analysis of {new} against {ref}."}
        before = {r["cause"]: r["pct_points"] for r in prev["bridge"]}
        rows = [{"cause": r["cause"], "before": before.get(r["cause"]), "after": r["pct_points"],
                 "delta": round(r["pct_points"] - before.get(r["cause"], 0), 2)} for r in b["bridge"]]
        facts["comparison"] = {"rows": rows, "total_before": prev["bill"]["change_pct"], "total_after": b["bill"]["change_pct"],
                               "bill_new_before": prev["bill"]["new"]["bill"], "bill_new_after": b["bill"]["new"]["bill"],
                               "what_was_loaded": store.ingests_since(prev["data_version"])}
        steps.append(_step("compare_with_previous_run", facts["comparison"]))

    out.update(answer_type="decomposition", facts=facts, steps=steps, bridge=b["bridge"])
    out["_bridge_result"] = b
    return out
