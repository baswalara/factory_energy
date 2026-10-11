"""Tests for the key functions: pricing, the bridge, cleaning, waste rules, number check."""
import numpy as np
import pandas as pd
import pytest

from app.cleaning import build_dataset, clean_meter, parse_period
from app.config import load_config, with_overrides
from app.numcheck import check_numbers
from app.store import merge_tariff
from app.tools.billing import bill
from app.tools.bridge import decompose, energy_slices
from app.tools.waste import cross_check, find_waste
from tests.conftest import tiny_store

CFG = load_config()


def test_price_change_in_the_middle_of_a_period(tmp_path):
    st = tiny_store(tmp_path, [("2026-07-01", "2026-07-02", "flat", 0.10), ("2026-07-02", "2026-07-03", "flat", 0.20)])
    b = bill(build_dataset(st, CFG), "2026-Q3")
    assert b["kwh"] == pytest.approx(192.0)
    assert b["bill"] == pytest.approx(96 * 0.10 + 96 * 0.20)        # each reading at the price in force


def test_peak_and_off_peak_hours(tmp_path):
    st = tiny_store(tmp_path, [("2026-07-01", "2026-07-02", "peak", 0.30), ("2026-07-01", "2026-07-02", "off_peak", 0.10)], days=1)
    b = bill(build_dataset(st, CFG), "2026-Q3")
    assert b["bill"] == pytest.approx(64 * 0.30 + 32 * 0.10)        # 06:00-22:00 is 64 intervals
    assert b["peak_kwh_share_pct"] == pytest.approx(64 / 96 * 100)


def test_unpriced_intervals_are_reported_not_dropped(tmp_path):
    st = tiny_store(tmp_path, [("2026-07-01", "2026-07-02", "flat", 0.10)])
    ds = build_dataset(st, CFG)
    assert ds.log["pricing"]["unpriced_intervals"] == 96
    assert bill(ds, "2026-Q3")["bill"] == pytest.approx(192 * 0.10)


def test_bridge_adds_up_to_the_total(ctx):
    _, cfg, ds = ctx
    b = decompose(ds, cfg, "2026-Q3", "2026-Q2")
    total = sum(r["pct_points"] for r in b["bridge"])
    assert total == pytest.approx(b["bill"]["change_pct"], abs=0.005)
    assert [r["cause"] for r in b["bridge"]] == ["tariff", "volume", "product_mix", "weather", "waste"]
    for r in b["bridge"]:
        assert r["range"][0] <= r["pct_points"] <= r["range"][1]
    assert sum(r["money"] for r in b["bridge"]) == pytest.approx(b["bill"]["change"], abs=0.05)


def test_bridge_adds_up_in_every_configuration(ctx):
    _, cfg, ds = ctx
    for c, price in [(with_overrides(cfg, use_product_mix=False), None), (cfg, ds.price * 0.9), (cfg, ds.price * 1.3)]:
        b = decompose(ds, c, "2026-Q3", "2026-Q2", price)
        assert sum(r["pct_points"] for r in b["bridge"]) == pytest.approx(b["bill"]["change_pct"], abs=0.005)
    b = decompose(ds, cfg, "2026-Q2", "2026-Q3")                    # reversed direction also reconciles
    assert sum(r["pct_points"] for r in b["bridge"]) == pytest.approx(b["bill"]["change_pct"], abs=0.005)


def test_heavier_mix_is_mix_not_volume():
    beta = {"const": 0, "working": 0, "units_A": 2.0, "units_B": 10.0, "cdd": 0, "hdd": 0}
    t0 = {"const": 90, "working": 60, "units_A": 100, "units_B": 100, "units_total": 200, "cdd": 0, "hdd": 0}
    t1 = {**t0, "units_A": 50, "units_B": 150}                      # same volume, heavier mix
    s = energy_slices(beta, t0, t1, ["A", "B"], True)
    assert s["volume"] == pytest.approx(0.0)
    assert s["product_mix"] == pytest.approx(50 * 10 - 50 * 2)
    t2 = {**t0, "units_A": 200, "units_B": 200, "units_total": 400}  # double volume, same mix
    s = energy_slices(beta, t0, t2, ["A", "B"], True)
    assert s["product_mix"] == pytest.approx(0.0)
    assert s["volume"] == pytest.approx(100 * 2 + 100 * 10)


def test_cleaning_handles_resets_and_gaps():
    idx = pd.date_range("2026-07-01", periods=96 * 14, freq="15min")
    s = pd.Series(10.0, index=idx)
    s.iloc[5] = -500.0                      # reset: negative
    s.iloc[50] = 90000.0                    # reset: register dump
    s.iloc[100:102] = np.nan                # short gap
    s.iloc[300:340] = np.nan                # long gap
    clean, stats, long = clean_meter(s, CFG)
    assert stats["negative_resets"] == 1 and stats["spike_resets"] == 1
    assert stats["interpolated"] == 4 and stats["profile_filled"] == 40
    assert clean.min() >= 0 and clean.max() == pytest.approx(10.0) and not clean.isna().any()
    assert long.sum() == 40


def test_corrected_tariff_replaces_only_the_span_it_covers():
    ts = pd.Timestamp
    old = pd.DataFrame({"valid_from": [ts("2026-04-01"), ts("2026-08-01")], "valid_to": [ts("2026-08-01"), ts("2026-10-01")],
                        "period": ["flat", "flat"], "price_per_kwh": [0.10, 0.12]})
    new = pd.DataFrame({"valid_from": [ts("2026-07-20")], "valid_to": [ts("2026-10-01")], "period": ["flat"], "price_per_kwh": [0.11]})
    m = merge_tariff(old, new).sort_values("valid_from").reset_index(drop=True)
    assert len(m) == 2
    assert m.valid_to[0] == ts("2026-07-20") and m.price_per_kwh[0] == 0.10      # old row trimmed, not deleted
    assert m.price_per_kwh[1] == 0.11


def test_waste_rule_flags_the_leak_and_nothing_else(ctx):
    _, cfg, ds = ctx
    w = find_waste(ds, cfg, "2026-Q3", "2026-Q2")
    growth = [f["meter"] for f in w["findings"] if f["kind"] == "idle_growth"]
    assert growth == ["compressors"]
    cleared = {c["meter"]: c["reason"] for c in w["meters_checked"] if not c["flagged"]}
    assert "weather explains" in cleared["hvac"]                    # hot spell is not waste
    assert "furnace" in cleared                                     # holding temperature is not waste
    chk = cross_check(decompose(ds, cfg, "2026-Q3", "2026-Q2"), w)
    assert chk["status"] == "supported_by_meters"


def test_holiday_logged_as_shift_is_not_a_working_day(ctx):
    _, _, ds = ctx
    day = ds.daily.loc["2026-08-26"]
    assert day.holiday_as_shift and not day.working


def test_number_check():
    facts = {"a": 16.07, "b": [91980, 306600], "when": "2026-08-10", "share": 21.9}
    assert check_numbers("Up 16.07%, saving 91,980 to 306,600 kWh since 2026-08-10 (21.9%).", facts)["ok"]
    assert check_numbers("Up 16.1%.", facts)["ok"]                  # same value, fewer decimals
    bad = check_numbers("The bill went up 25% and we save 500,000 kWh.", facts)
    assert not bad["ok"] and bad["unmatched"] == [25.0, 500000.0]


def test_template_brief_passes_its_own_number_check(client):
    for q in ["Why did the bill go up?", "Which assets use the most energy?", "How much energy is used when nothing is produced?",
              "What should we do first?", "What if the tariff were 10% lower?", "What was the total bill?"]:
        d = client.post("/analyze", json={"question": q}).json()
        assert check_numbers(d["summary"], d["data"])["ok"], q


def test_parse_period():
    assert parse_period("2026-Q3") == "2026-Q3" and parse_period("q2 2026") == "2026-Q2" and parse_period("last summer") is None
