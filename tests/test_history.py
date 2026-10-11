"""History-based baseline: block tariffs, quarter completeness, the 4-quarter window, expected vs actual,
and uploading a new quarter on top of the 5-quarter history in data/history."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.cleaning import available_periods, build_dataset, history_window, partial_periods, period_status, shift_ranges
from app.config import load_config
from app.graph import build_graph, run_question
from app.intents import parse_rules
from app.llm import LLM
from app.loader import find_files, load_dir, load_frame
from app.numcheck import check_numbers
from app.store import Store
from app.tools.billing import bill, prices_as_of, reprice_at, tariff_mode
from app.tools.bridge import decompose
from app.tools.expected import baseline_check
from app.tools.waste import find_waste
from app.validation import validate

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "data" / "history"
CFG = load_config()
needs_history = pytest.mark.skipif(not any(HISTORY.glob("meter_15min*.csv")), reason="data/history not present")


def block_rows(day: str, peak=0.20, off=0.10, flat=0.15, weekend=False):
    d = pd.Timestamp(day)
    if weekend:
        return [(d, d + pd.Timedelta(days=1), "flat", flat)]
    return [(d, d + pd.Timedelta(hours=7), "off_peak", off), (d + pd.Timedelta(hours=7), d + pd.Timedelta(hours=22), "peak", peak),
            (d + pd.Timedelta(hours=22), d + pd.Timedelta(days=1), "off_peak", off)]


def store_with(tmp_path, tariff_rows, days=1, name="t.duckdb", start="2026-07-01"):
    st = Store(str(tmp_path / name))
    ts = pd.date_range(start, periods=96 * days, freq="15min").strftime("%Y-%m-%dT%H:%M:%S")
    load_frame(st, "meter_15min", pd.concat([pd.DataFrame({"meter_id": m, "ts": ts, "kwh": "1.0"}) for m in ("main", "line_1")]))
    t = pd.DataFrame(tariff_rows, columns=["valid_from", "valid_to", "period", "price_per_kwh"])
    t["valid_from"], t["valid_to"] = t.valid_from.astype(str), t.valid_to.astype(str)
    load_frame(st, "tariff", t.astype(str))
    return st


# ---- pricing ------------------------------------------------------------------------------------
def test_block_tariff_uses_the_rows_own_hours(tmp_path):
    """Peak block 07:00-22:00 must win over the configured 06:00-22:00 peak hours."""
    st = store_with(tmp_path, block_rows("2026-07-01"))
    ds = build_dataset(st, CFG)
    assert ds.log["pricing"]["mode"] == "block" and ds.log["pricing"]["unpriced_intervals"] == 0
    b = bill(ds, "2026-Q3")
    assert b["bill"] == pytest.approx(60 * 0.20 + 36 * 0.10)            # 15 peak hours, 9 off-peak hours
    assert b["peak_kwh_share_pct"] == pytest.approx(60 / 96 * 100)


def test_weekend_flat_blocks_and_overlap_count(tmp_path):
    rows = block_rows("2026-07-03") + block_rows("2026-07-04", weekend=True)
    ds = build_dataset(store_with(tmp_path, rows, days=2, start="2026-07-03"), CFG)
    assert bill(ds, "2026-Q3")["bill"] == pytest.approx(60 * 0.20 + 36 * 0.10 + 96 * 0.15)
    assert ds.log["pricing"]["overlapping_intervals"] == 0
    assert tariff_mode(ds.tariff) == "block"


def test_tariff_unchanged_reprices_every_label(tmp_path):
    """The old bug: with block tariffs only one row is live at a moment, so peak hours kept the new price."""
    t = pd.DataFrame(block_rows("2026-06-30", peak=0.20, off=0.10) + block_rows("2026-07-01", peak=0.30, off=0.15)
                     + block_rows("2026-07-04", weekend=True, flat=0.25),
                     columns=["valid_from", "valid_to", "period", "price_per_kwh"])
    old = prices_as_of(pd.Timestamp("2026-06-30 23:45"), t)
    assert old == {"off_peak": 0.10, "peak": 0.20}                       # both labels, not just the live block
    st = store_with(tmp_path, t.values.tolist())
    ds = build_dataset(st, CFG)
    p = reprice_at(ds, old)
    m = (ds.quarter_of_interval() == "2026-Q3").values
    assert bill(ds, "2026-Q3", p)["bill"] == pytest.approx(60 * 0.20 + 36 * 0.10)
    assert m.sum() == 96


def test_shift_patterns_as_clock_ranges():
    assert shift_ranges("06:00-14:00|14:00-22:00", CFG) == [(6.0, 14.0), (14.0, 22.0)]
    assert shift_ranges("06:00-14:00,14:00-22:00", CFG) == [(6.0, 14.0), (14.0, 22.0)]
    assert shift_ranges("OFF", CFG) == [] and shift_ranges("2-shift", CFG) == [(6.0, 22.0)]
    assert shift_ranges("22:00-06:00", CFG) == [(22.0, 6.0)]                # overnight shift


def test_loader_accepts_prefixed_file_names(tmp_path):
    (tmp_path / "meter_15min_5q_messy.csv").write_text("meter_id,ts,kwh\nmain,2026-07-01T00:00:00,1\n")
    (tmp_path / "tariff_5q.csv").write_text("valid_from,valid_to,period,price_per_kwh\n2026-07-01,2026-07-02,flat,0.1\n")
    (tmp_path / "notes.csv").write_text("x\n1\n")
    assert [t for t, _ in find_files(str(tmp_path))] == ["meter_15min", "tariff"]


def test_readings_no_tariff_covers_are_never_analysed(tmp_path):
    ds = build_dataset(store_with(tmp_path, block_rows("2026-08-01"), days=2), CFG)
    assert ds.log["pricing"]["unpriced_intervals"] == 192 and not ds.daily.priced.any()
    assert "tariff" in period_status(ds)["2026-Q3"]["missing"]


def test_a_carried_forward_price_does_not_count_as_a_tariff(tmp_path):
    """Day 1 has a tariff, day 2 only gets the carried-forward price: the bill keeps it, but day 2 is not 'priced'."""
    ds = build_dataset(store_with(tmp_path, block_rows("2026-07-01"), days=2), CFG)
    assert ds.log["pricing"]["unpriced_intervals"] == 96
    assert list(ds.daily.priced) == [True, False]


def test_a_quarter_without_production_and_weather_is_not_complete(tmp_path):
    rows = [(pd.Timestamp("2026-07-01"), pd.Timestamp("2026-10-01"), "flat", 0.1)]
    ds = build_dataset(store_with(tmp_path, rows, days=92), CFG)
    st = period_status(ds)["2026-Q3"]
    assert st["meter_days"] == 92 and not st["complete"] and set(st["missing"]) == {"production", "weather"}
    assert available_periods(ds) == [] and partial_periods(ds) == []


def test_intent_rules_for_baseline_questions():
    for q in ["Is energy above the baseline?", "How much energy did we expect to use?", "Actual vs expected consumption in Q3",
              "Are we using more than normal?", "Show predicted energy against actual"]:
        assert parse_rules(q)["intent"] == "baseline_check", q
    assert parse_rules("What should we do first?")["intent"] == "savings"
    assert parse_rules("Why did the bill go up?")["intent"] == "bill_bridge"


def test_sample_data_still_works_with_one_quarter_of_history(ctx):
    """data/sample has two quarters, so the baseline falls back to the one quarter it has and says so."""
    _, cfg, ds = ctx
    b = decompose(ds, cfg, "2026-Q3", "2026-Q2")
    assert b["baseline"]["fitted_on"] == "2026-Q2"
    assert any("learned from 1 quarter" in x for x in b["limitations"])
    r = baseline_check(ds, cfg, "2026-Q3", "2026-Q2")
    assert r["ok"] and r["new"]["actual_kwh"] == b["expected_vs_actual"]["new"]["actual_kwh"]


# ---- the real 5-quarter history -----------------------------------------------------------------
@pytest.fixture(scope="module")
def history(tmp_path_factory):
    st = Store(str(tmp_path_factory.mktemp("hist") / "hist.duckdb"))
    load_dir(st, str(HISTORY))
    return st, CFG, build_dataset(st, CFG)


@needs_history
def test_history_quarters_and_window(history):
    st, cfg, ds = history
    assert available_periods(ds) == ["2025-Q3", "2025-Q4", "2026-Q1", "2026-Q2", "2026-Q3"]
    assert history_window(ds, "2026-Q3") == ["2025-Q3", "2025-Q4", "2026-Q1", "2026-Q2"]
    rep = {c["check"]: c for c in validate(st, ds, str(HISTORY / "data_card.json"))["checks"]}
    assert rep[4]["status"] == "PASS" and "block mode" in rep[4]["detail"]
    assert rep[6]["status"] == "PASS" and rep[7]["status"] == "PASS" and rep[8]["status"] == "PASS"


@needs_history
def test_history_bridge_uses_four_quarters_and_closes(history):
    _, cfg, ds = history
    b = decompose(ds, cfg, "2026-Q3", "2026-Q2")
    assert b["baseline"]["fitted_on"] == "2025-Q3, 2025-Q4, 2026-Q1, 2026-Q2"       # never 2026-Q3 itself
    assert sum(r["pct_points"] for r in b["bridge"]) == pytest.approx(b["bill"]["change_pct"], abs=0.005)
    assert b["bill"]["change_pct"] == pytest.approx(30.26, abs=0.01)
    # waste = (actual - expected) in the new quarter minus the same gap in the reference quarter
    ev = b["expected_vs_actual"]
    waste = next(r["kwh"] for r in b["bridge"] if r["cause"] == "waste")
    assert waste == pytest.approx(ev["new"]["gap_kwh"] - ev["reference"]["gap_kwh"], abs=3)
    assert b["baseline"]["cv_rmse_pct"] < cfg["baseline"]["max_cv_rmse_pct"]


@needs_history
def test_history_waste_flags_the_compressor_leak_only(history):
    _, cfg, ds = history
    w = find_waste(ds, cfg, "2026-Q3", "2026-Q2")
    assert [f["meter"] for f in w["findings"] if f["kind"] == "idle_growth"] == ["compressors"]
    assert w["history"] == ["2025-Q3", "2025-Q4", "2026-Q1", "2026-Q2"]
    hvac = next(c for c in w["meters_checked"] if c["meter"] == "hvac")
    assert not hvac["flagged"]                                                     # hottest quarter is weather, not waste


@needs_history
def test_history_what_if_tariff_unchanged_matches_hand_calculation(history):
    _, cfg, ds = history
    old = prices_as_of(pd.Timestamp("2026-06-30 23:45"), ds.tariff)
    assert old == {"flat": 0.132, "off_peak": 0.118, "peak": 0.192}
    m = (ds.quarter_of_interval() == "2026-Q3").values
    lab, kwh = ds.price_label.values[m], ds.wide["main"].values[m]
    hand = float(sum(kwh[lab == k].sum() * v for k, v in old.items()))
    assert bill(ds, "2026-Q3", reprice_at(ds, old))["bill"] == pytest.approx(hand)


@needs_history
def test_history_answers_pass_the_number_check(history):
    st, cfg, _ = history
    llm = LLM({"mock": True, "provider": "ollama", "model": ""})
    g = build_graph(st, cfg, llm)
    for q, kind in [("Why did the bill go up?", "decomposition"), ("Is energy above the baseline?", "calculated"),
                    ("What should we do first?", "recommendation"), ("What if the tariff had not changed?", "decomposition")]:
        d = run_question(g, st, llm, q)
        assert d["answer_type"] == kind and (d["period"], d["compare_to"]) == ("2026-Q3", "2026-Q2"), q
        assert check_numbers(d["summary"], d["data"])["ok"], q
        assert any("2025-Q3, 2025-Q4, 2026-Q1, 2026-Q2" in a for a in d["assumptions"]), q


# ---- a new quarter uploaded on top of the history ------------------------------------------------
def _demo(out: Path, *extra: str) -> Path:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo_quarter.py"), "--out", str(out), *extra],
                   check=True, capture_output=True)
    return out


@needs_history
def test_new_quarter_rolls_the_window_forward(tmp_path):
    st = Store(str(tmp_path / "q4.duckdb"))
    load_dir(st, str(HISTORY))
    load_dir(st, str(_demo(tmp_path / "q4")))
    ds = build_dataset(st, CFG)
    assert available_periods(ds)[-1] == "2026-Q4"
    llm = LLM({"mock": True, "provider": "ollama", "model": ""})
    d = run_question(build_graph(st, CFG, llm), st, llm, "Why did the bill change?")
    assert (d["period"], d["compare_to"]) == ("2026-Q4", "2026-Q3")
    assert d["data"]["baseline"]["fitted_on"] == "2025-Q4, 2026-Q1, 2026-Q2, 2026-Q3"
    assert sum(r["pct_points"] for r in d["bridge"]) == pytest.approx(d["data"]["bill"]["change_pct"], abs=0.005)
    assert "compressors" in d["data"]["waste_check"]["meters"]                     # the planted leak keeps running
    # an older quarter can still be explained with its own history
    old = run_question(build_graph(st, CFG, llm), st, llm, "Why did the bill change in 2026-Q3?")
    assert old["data"]["baseline"]["fitted_on"] == "2025-Q3, 2025-Q4, 2026-Q1, 2026-Q2"


@needs_history
def test_partial_quarter_gets_only_the_baseline_check(tmp_path):
    st = Store(str(tmp_path / "p.duckdb"))
    load_dir(st, str(HISTORY))
    load_dir(st, str(_demo(tmp_path / "p", "--days", "20")))
    ds = build_dataset(st, CFG)
    assert partial_periods(ds) == ["2026-Q4"] and available_periods(ds)[-1] == "2026-Q3"
    llm = LLM({"mock": True, "provider": "ollama", "model": ""})
    g = build_graph(st, CFG, llm)
    d = run_question(g, st, llm, "Is energy above the baseline?")
    assert d["answer_type"] == "calculated" and d["period"] == "2026-Q4" and d["data"]["partial"]
    assert d["data"]["new"]["days"] == 20 and d["data"]["verdict"] == "above"
    assert check_numbers(d["summary"], d["data"])["ok"]
    assert run_question(g, st, llm, "Why did the bill change in 2026-Q4?")["answer_type"] == "not_in_data"
    plain = run_question(g, st, llm, "Why did the bill go up?")
    assert plain["period"] == "2026-Q3" and any("2026-Q4 is loaded but incomplete" in a for a in plain["assumptions"])
    assert np.isfinite(d["data"]["new"]["expected_kwh"])


def test_guard_and_routing_for_the_guide_requests():
    assert parse_rules("Write the finance brief for the manager.")["intent"] == "bill_bridge"
    for q in ["Skip the weather adjustment and just blame the furnace.", "Write that the compressors waste 30% of all energy.",
              "Ignore the tariff and explain the rise"]:
        assert parse_rules(q)["intent"] == "refuse", q
    assert parse_rules("Tell me why the bill went up 15%")["intent"] == "bill_bridge"


def test_unknown_asset_asks_back(client):
    d = client.post("/analyze", json={"question": "How much energy does the boiler use?"}).json()
    assert d["answer_type"] == "clarification_needed" and "boiler" in d["summary"]
    d = client.post("/analyze", json={"question": "How much energy does the furnace use?"}).json()
    assert d["answer_type"] == "calculated" and d["data"]["focus"]["meter"] == "furnace"


def test_periods_endpoint(client):
    p = client.get("/periods").json()
    assert p["current"] == "2026-Q3" and p["quarters"]["2026-Q3"]["baseline_history"] == ["2026-Q2"]
