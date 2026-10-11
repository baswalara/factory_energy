"""Evaluation (guide section 7). Method: exact match with a stated tolerance; no LLM judge.

1. Runs every case in eval/cases.json through POST /analyze on a fresh copy of the sample data.
   Expected numbers are recomputed here from the raw CSVs with plain pandas, not with app code.
2. Scores the bill split against the planted split (eval/truth.json), before and after the late drop.
3. Compares two configurations (baseline with and without product mix) on several generated datasets.

    python scripts/run_eval.py            # writes eval/report.md and eval/results.json

Uses MOCK_LLM unless LLM settings are present in the environment.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP = tempfile.mkdtemp()
os.environ["DB_PATH"] = str(Path(TMP) / "eval.duckdb")
os.environ.setdefault("MOCK_LLM", "true")

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
from fastapi.testclient import TestClient            # noqa: E402

from app import api                                  # noqa: E402
from app.cleaning import build_dataset               # noqa: E402
from app.config import with_overrides                # noqa: E402
from app.loader import load_dir                      # noqa: E402
from app.numcheck import check_numbers               # noqa: E402
from app.store import Store                          # noqa: E402
from app.tools.bridge import decompose               # noqa: E402
from app.tools.savings import recommend              # noqa: E402
from app.tools.waste import cross_check, find_waste  # noqa: E402

CAUSES = ["tariff", "volume", "product_mix", "weather", "waste"]


def independent_expected(folder: Path, truth: dict) -> dict:
    """Expected values straight from the raw files (resets removed, no gap filling), without app code."""
    m = pd.read_csv(folder / "meter_15min.csv", parse_dates=["ts"]).drop_duplicates(["meter_id", "ts"], keep="last")
    w = m.pivot(index="ts", columns="meter_id", values="kwh")
    w = w.where((w >= 0) & (w < 10000))
    w.index = w.index.tz_localize(None)
    t = pd.read_csv(folder / "tariff.csv", parse_dates=["valid_from", "valid_to"])
    for c in ("valid_from", "valid_to"):
        t[c] = t[c].dt.tz_localize(None)
    peak = (w.index.hour >= 6) & (w.index.hour < 22)

    def prices(rows):
        p = pd.Series(np.nan, index=w.index)
        for r in rows.itertuples():
            sel = (w.index >= r.valid_from) & (w.index < r.valid_to) & (peak if r.period == "peak" else ~peak)
            p[sel] = r.price_per_kwh
        return p

    q3, q2 = w.index.month >= 7, w.index.month < 7
    price = prices(t)
    old = t[t.valid_from == t.valid_from.min()].assign(valid_to=pd.Timestamp("2027-01-01"))
    bill_q3 = float((w.main * price)[q3].sum())
    lines = w[[c for c in w.columns if c.startswith("line")]].sum(axis=1)
    idle = lines <= 0.2 * lines[lines > 0].quantile(0.95)
    subs = w.drop(columns="main")
    return {"bill_q3": bill_q3, "kwh_q2": float(w.main[q2].sum()), "furnace_kwh_q3": float(w.furnace[q3].sum()),
            "idle_kwh_q3": float(w.main[q3 & idle].sum()), "top_asset_q3": subs[q3].sum().idxmax(),
            "unmetered_pct_q3": float((w.main[q3].sum() - subs[q3].sum().sum()) / w.main[q3].sum() * 100),
            "bill_q3_minus10": bill_q3 * 0.9, "bill_q3_plus5": bill_q3 * 1.05,
            "bill_q3_old_prices": float((w.main * prices(old))[q3].sum()),
            "leak_meter": truth["waste"]["meter"]}


def dig(obj, path: str):
    for part in path.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def run_check(c: dict, d: dict, exp: dict, truth: dict) -> tuple[bool, str]:
    t = c["type"]
    rows = {r["cause"]: r["pct_points"] for r in d.get("bridge", [])}
    try:
        if t == "answer_type":
            return d["answer_type"] == c["is"], f"answer_type {d['answer_type']} (expected {c['is']})"
        if t == "adds_up":
            total = d["data"]["bill"]["change_pct"]
            return abs(sum(rows.values()) - total) <= 0.005, f"slices {sum(rows.values()):.2f} vs total {total}"
        if t == "largest_cause":
            top = max(rows, key=rows.get)
            return top == c["is"], f"largest cause {top} (expected {c['is']})"
        if t == "cause_gt":
            return rows[c["a"]] > rows[c["b"]], f"{c['a']} {rows[c['a']]} > {c['b']} {rows[c['b']]}"
        if t == "cause_positive":
            return rows[c["a"]] > 0, f"{c['a']} {rows[c['a']]} > 0"
        if t == "numeric":
            got, want = float(dig(d, c["path"])), float(exp[c["expected"]])
            ok = abs(got - want) <= c["tol_abs"] if "tol_abs" in c else abs(got - want) <= abs(want) * c["tol_pct"] / 100
            return ok, f"{c['path']} = {got:,.2f}, independent value {want:,.2f}"
        if t == "equals":
            got = dig(d, c["path"])
            return got == exp[c["expected"]], f"{c['path']} = {got} (expected {exp[c['expected']]})"
        if t == "not_in_summary":
            hit = [x for x in c["text"] if x in d["summary"]]
            return not hit, f"injected number in summary: {hit or 'no'}"
        if t == "truth_saving_in_range":
            lo, hi = next(r for r in d["recommendations"] if r["meter"] == truth["waste"]["meter"])["saving_kwh_per_year"]
            want = truth["waste"]["true_annual_saving_kwh"]
            return lo <= want <= hi, f"planted saving {want:,.0f} kWh/yr vs range [{lo:,.0f}, {hi:,.0f}]"
        if t == "ranges_everywhere":
            ok = all(len(r[k]) == 2 and r[k][0] <= r[k][1] for r in d["recommendations"]
                     for k in ("saving_kwh_per_year", "saving_money_per_year", "saving_co2e_t_per_year", "payback_months"))
            return ok, "every saving and payback is a low-high pair"
        if t == "payback_sorted":
            mids = [sum(r["payback_months"]) / 2 for r in d["recommendations"]]
            return mids == sorted(mids), f"payback midpoints {mids}"
        if t in ("flagged", "not_flagged"):
            flagged = {f["meter"] for f in d["data"]["findings"]}
            ok = (c["meter"] in flagged) == (t == "flagged")
            return ok, f"flagged meters {sorted(flagged)}"
    except Exception as e:
        return False, f"check could not run: {type(e).__name__}: {e}"
    return False, f"unknown check {t}"


def split_error(bridge: dict, true: dict) -> dict:
    got = {r["cause"]: r for r in bridge["bridge"]}
    err = {c: round(got[c]["pct_points"] - true["split_pct_points"][c], 2) for c in CAUSES}
    # 0.02 allowance: slices are reported to two decimals
    inside = {c: got[c]["range"][0] - 0.02 <= true["split_pct_points"][c] <= got[c]["range"][1] + 0.02 for c in CAUSES}
    return {"estimate": {c: got[c]["pct_points"] for c in CAUSES}, "truth": {c: round(true["split_pct_points"][c], 2) for c in CAUSES},
            "error_pp": err, "mean_abs_error_pp": round(float(np.mean([abs(v) for v in err.values()])), 3),
            "max_abs_error_pp": max(abs(v) for v in err.values()), "truth_inside_range": inside,
            "total_estimate": bridge["bill"]["change_pct"], "total_truth": round(true["total_change_pct"], 2),
            "reconciles": abs(sum(got[c]["pct_points"] for c in CAUSES) - bridge["bill"]["change_pct"]) <= 0.005}


def seed_study(cfg: dict, seeds: list[int]) -> list[dict]:
    """Both configurations on freshly generated datasets (different random seeds)."""
    out = []
    for seed in seeds:
        d = Path(TMP) / f"seed{seed}"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_data.py"), "--out", str(d / "data"), "--late", str(d / "late"),
                        "--truth", str(d / "truth.json"), "--seed", str(seed)], check=True, capture_output=True)
        truth = json.loads((d / "truth.json").read_text())
        st = Store(str(d / "db.duckdb"))
        load_dir(st, str(d / "data"))
        ds = build_dataset(st, cfg)
        row = {"seed": seed}
        for name, c in (("with_mix", cfg), ("without_mix", with_overrides(cfg, use_product_mix=False))):
            b = decompose(ds, c, truth["new"], truth["reference"])
            row[name] = split_error(b, truth["before_late_drop"])
        w = find_waste(ds, cfg, truth["new"], truth["reference"])
        growth = [f for f in w["findings"] if f["kind"] == "idle_growth"]
        rec = recommend(ds, cfg, w, cross_check(decompose(ds, cfg, truth["new"], truth["reference"]), w))["recommendations"]
        leak = next((r for r in rec if r["meter"] == truth["waste"]["meter"] and r["finding"] == "idle_growth"), None)
        row["waste"] = {"leak_found": [f["meter"] for f in growth] == [truth["waste"]["meter"]],
                        "false_claims": [f["meter"] for f in w["findings"] if f["meter"] in truth["no_waste_meters"]],
                        "leak_kw_estimate": growth[0]["excess_kw"] if growth else None, "leak_kw_truth": truth["waste"]["leak_kw"],
                        "onset_estimate": growth[0]["onset"] if growth else None, "onset_truth": truth["waste"]["leak_start"],
                        "saving_in_range": bool(leak and leak["saving_kwh_per_year"][0] <= truth["waste"]["true_annual_saving_kwh"]
                                                <= leak["saving_kwh_per_year"][1])}
        out.append(row)
    return out


def main() -> None:
    truth = json.loads((ROOT / "eval" / "truth.json").read_text())
    cases = json.loads((ROOT / "eval" / "cases.json").read_text())
    sample = ROOT / "data" / "sample"
    api.store.clear()
    load_dir(api.store, str(sample))
    client = TestClient(api.app)
    exp = independent_expected(sample, truth)

    results, latencies, calls, tokens, faithful = [], [], [], [], []
    for case in cases:
        body = {k: case[k] for k in ("question", "period", "compare_to") if k in case}
        d = client.post("/analyze", json=body).json()
        tr = client.get(f"/trace/{d['run_id']}").json()
        latencies.append(tr["latency_ms"]); calls.append(tr["llm_calls"]); tokens.append(tr["tokens"])
        checks = [run_check(c, d, exp, truth) for c in case["checks"]]
        if d["data"] is not None:
            f = check_numbers(d["summary"], d["data"])
            faithful.append(f["ok"])
            checks.append((f["ok"], f"number faithfulness: {f['numbers_checked']} numbers, unmatched {f['unmatched']}"))
        results.append({"id": case["id"], "category": case["category"], "question": case["question"],
                        "answer_type": d["answer_type"], "passed": all(ok for ok, _ in checks),
                        "checks": [{"ok": ok, "detail": msg} for ok, msg in checks]})

    ds = build_dataset(api.store, api.cfg)
    before = split_error(decompose(ds, api.cfg, truth["new"], truth["reference"]), truth["before_late_drop"])
    load_dir(api.store, str(ROOT / "data" / "late_drop"))
    after = split_error(decompose(build_dataset(api.store, api.cfg), api.cfg, truth["new"], truth["reference"]), truth["after_late_drop"])
    study = seed_study(api.cfg, [42, 1, 2, 3, 4, 5, 6, 7])

    def agg(name):
        return {"mean_abs_error_pp": round(statistics.mean(r[name]["mean_abs_error_pp"] for r in study), 3),
                "max_abs_error_pp": round(max(r[name]["max_abs_error_pp"] for r in study), 2),
                "mix_abs_error_pp": round(statistics.mean(abs(r[name]["error_pp"]["product_mix"]) for r in study), 3),
                "waste_abs_error_pp": round(statistics.mean(abs(r[name]["error_pp"]["waste"]) for r in study), 3),
                "truth_inside_range_pct": round(100 * statistics.mean(v for r in study for v in r[name]["truth_inside_range"].values()), 1),
                "all_reconcile": all(r[name]["reconciles"] for r in study)}

    lat = sorted(latencies)
    summary = {
        "llm": api.llm.name, "cases": len(results), "cases_passed": sum(r["passed"] for r in results),
        "by_category": {c: f"{sum(r['passed'] for r in results if r['category'] == c)}/{sum(1 for r in results if r['category'] == c)}"
                        for c in dict.fromkeys(r["category"] for r in results)},
        "decomposition_before_late_drop": before, "decomposition_after_late_drop": after,
        "configurations": {"with_product_mix": agg("with_mix"), "without_product_mix": agg("without_mix")},
        "waste_detection": {"datasets": len(study), "leak_found": sum(r["waste"]["leak_found"] for r in study),
                            "false_waste_claims": sum(len(r["waste"]["false_claims"]) for r in study),
                            "leak_kw_mean_abs_error": round(statistics.mean(abs(r["waste"]["leak_kw_estimate"] - r["waste"]["leak_kw_truth"])
                                                                           for r in study if r["waste"]["leak_kw_estimate"] is not None), 2),
                            "onset_exact": sum(r["waste"]["onset_estimate"] == r["waste"]["onset_truth"] for r in study),
                            "true_saving_inside_range": sum(r["waste"]["saving_in_range"] for r in study)},
        "number_faithfulness": f"{sum(faithful)}/{len(faithful)} briefs with every number matched to a tool result",
        "latency_ms": {"p50": round(lat[len(lat) // 2], 1), "p95": round(lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))], 1)},
        "llm_calls_per_request": round(statistics.mean(calls), 2), "tokens_per_request": round(statistics.mean(tokens), 1),
    }
    (ROOT / "eval" / "results.json").write_text(json.dumps({"summary": summary, "cases": results, "seed_study": study}, indent=2))

    row = lambda name, e: f"| {name} | " + " | ".join(f"{e['estimate'][c]:.2f} / {e['truth'][c]:.2f} ({e['error_pp'][c]:+.2f})" for c in CAUSES) + f" | {e['mean_abs_error_pp']} |"
    cfgs = summary["configurations"]
    wd = summary["waste_detection"]
    md = [
        "# Evaluation report", "",
        f"Generated by `python scripts/run_eval.py`. LLM setting for this run: **{summary['llm']}**.", "",
        "**Method.** Exact match with a stated tolerance; no LLM judge. Answer types must match exactly. Numeric answers are compared with "
        "values recomputed in the evaluation script from the raw CSV files using plain pandas (tolerance shown per case in `eval/cases.json`). "
        "The bill split is compared with the split planted by the data generator (`eval/truth.json`), which the application never reads.", "",
        "**Read this first.** The data is our own synthetic data, built from the same kind of model the baseline assumes "
        "(energy linear in units and degree-days, plus a cooling effect that depends on whether the day is a working day, and random day-to-day noise). "
        "Scores on it are therefore optimistic. They show the code is correct and how it behaves under noise; they do not predict accuracy on HCLTech's hidden data.", "",
        "## 1. Test cases", "",
        f"**{summary['cases_passed']} of {summary['cases']} cases passed.** By category: "
        + ", ".join(f"{k} {v}" for k, v in summary["by_category"].items()) + ".", "",
        "| ID | Category | Question | Answer type | Result | Detail |", "|---|---|---|---|---|---|"]
    for r in results:
        detail = "; ".join(c["detail"] for c in r["checks"] if not c["ok"]) or r["checks"][-1]["detail"] if not r["passed"] else \
                 "; ".join(c["detail"] for c in r["checks"][1:2]) or "answer type matches"
        md.append(f"| {r['id']} | {r['category']} | {r['question']} | {r['answer_type']} | {'PASS' if r['passed'] else 'FAIL'} | {detail} |")
    md += ["", "## 2. Decomposition error (percentage points)", "",
           "Each cell: our estimate / planted truth (error).", "",
           "| Data | " + " | ".join(CAUSES) + " | Mean abs error |", "|---|" + "---|" * (len(CAUSES) + 1),
           row("Sample data", before), row("After late drop", after), "",
           f"Total change: {before['total_estimate']}% estimated against {before['total_truth']}% planted "
           f"(the gap comes from the {ds.log['meters']['main']['missing_intervals']} main-meter intervals missing from the file, which cleaning fills with typical values). "
           f"After the late drop: {after['total_estimate']}% against {after['total_truth']}%.",
           f"The planted split leaves {truth['before_late_drop']['random_noise_pct_points']} points of pure random noise that belong to no cause; "
           "our method has to put them somewhere, and most of them land in the waste slice.", "",
           "## 3. Total reconciliation", "",
           f"Slices sum to the total change in every run: sample {before['reconciles']}, after late drop {after['reconciles']}, "
           f"all {len(study)} generated datasets in both configurations {cfgs['with_product_mix']['all_reconcile'] and cfgs['without_product_mix']['all_reconcile']}.", "",
           "## 4. Two configurations compared", "",
           f"Baseline with a separate energy rate per product (our choice) against one rate for all units, on {len(study)} generated datasets (seeds {[r['seed'] for r in study]}).", "",
           "| Configuration | Mean abs error | Worst slice error | Mix slice error | Waste slice error | Truth inside our range |", "|---|---|---|---|---|---|",
           *[f"| {n} | {c['mean_abs_error_pp']} | {c['max_abs_error_pp']} | {c['mix_abs_error_pp']} | {c['waste_abs_error_pp']} | {c['truth_inside_range_pct']}% |"
             for n, c in (("With product mix", cfgs["with_product_mix"]), ("Without product mix", cfgs["without_product_mix"]))], "",
           "Without product mix, the mix shift cannot be separated: it is reported as waste, which blames machines for a change in what was made. "
           "That is the reason we keep the per-product baseline even though its rates are less precisely estimated.", "",
           "## 5. Waste detection and savings", "",
           f"- Planted compressor leak found as the only idle-growth finding: {wd['leak_found']} of {wd['datasets']} datasets.",
           f"- False waste claims on meters with no planted waste (furnace, hvac, utilities): {wd['false_waste_claims']}.",
           f"- Leak size: mean absolute error {wd['leak_kw_mean_abs_error']} kW against a planted {truth['waste']['leak_kw']} kW. Start date exact in {wd['onset_exact']} of {wd['datasets']}.",
           f"- Planted yearly saving inside our low-high range: {wd['true_saving_inside_range']} of {wd['datasets']}. "
           f"The planted saving assumes a repair recovers {truth['waste']['repair_effectiveness_assumed']:.0%} of the leak; that share is our assumption in the generator.", "",
           "## 6. Number faithfulness", "",
           f"{summary['number_faithfulness']}. Checked by `app/numcheck.py` on every answer that carries data. "
           + ("This run used the mock LLM, so the briefs were written by the template; the check on real model output is exercised by "
              "`tests/test_api.py::test_invented_numbers_never_reach_the_answer` with a stand-in model that invents numbers. "
              "**Re-run this script with the real model before quoting this figure.**" if api.llm.mock else ""), "",
           "## 7. Latency and cost", "",
           f"p50 {summary['latency_ms']['p50']} ms, p95 {summary['latency_ms']['p95']} ms per request (in-process, this machine). "
           f"LLM calls per request {summary['llm_calls_per_request']}, tokens per request {summary['tokens_per_request']}."
           + (" Mock LLM: no model calls were made, so these are the code-only figures. With a real model expect up to 2 calls per request (3 if the brief is retried)." if api.llm.mock else ""), ""]
    (ROOT / "eval" / "report.md").write_text("\n".join(md))
    print(json.dumps(summary, indent=2))
    failed = [r for r in results if not r["passed"]]
    for r in failed:
        print("FAILED", r["id"], r["question"], [c["detail"] for c in r["checks"] if not c["ok"]])


if __name__ == "__main__":
    main()
