"""Synthetic factory data generator (seeded). No real data is used.

Writes the four Annex B files, a late-drop pack, a machine-readable data card
and the planted answer key. The answer key goes to eval/truth.json and is read
ONLY by scripts/run_eval.py, never by the application.

    python scripts/generate_data.py --out data/sample --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ---- planted parameters (the "true" factory) --------------------------------
START, END = "2026-04-01", "2026-09-30"            # Q2 (reference) and Q3 (new)
RATE = {"A": 4.0, "B": 7.0, "C": 14.0}             # kWh per unit, by product family
LINE_OF = {"A": "line_1", "B": "line_2", "C": "line_3"}
UNITS = {"Q2": {"A": 1500, "B": 1000, "C": 400},   # per 2-shift day
         "Q3": {"A": 1470, "B": 1020, "C": 480}}   # +2% volume, mix shifts to heavy product C
SPLIT = {"line": 0.6, "compressors": 0.2, "furnace": 0.2}  # where product energy is metered
COOL_KWH_PER_CDD = 70.0
COOL_NONWORK_FACTOR = 0.6                          # empty building needs less cooling
LEAK_KW, LEAK_START = 35.0, "2026-08-10"           # hidden waste: compressed-air leak
REPAIR_EFFECTIVENESS = 0.8                         # share of the leak a repair really recovers
LIGHT_ON_KW, LIGHT_IDLE_KW = 80.0, 50.0            # standing waste: lights stay on, both quarters
PEAK = (6, 22)
DAY_NOISE = 0.025                                  # day-to-day efficiency variation (sd)
TARIFF = [("2026-04-01", "2026-08-01", 0.130, 0.085),
          ("2026-08-01", "2026-10-01", 0.1456, 0.0952)]     # +12% part-way through Q3
TARIFF_CORRECTED = [("2026-04-01", "2026-07-20", 0.130, 0.085),
                    ("2026-07-20", "2026-10-01", 0.1440, 0.0952)]
HOLIDAY_OK = "2026-05-01"        # holiday logged correctly
HOLIDAY_TRAP = "2026-08-26"      # holiday logged as a normal 2-shift day
HVAC_CALIBRATION = 0.93          # late drop: hvac sub-meter read 7% high in Q3


def calendar(rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = pd.date_range(START, END, freq="D")
    doy = np.arange(len(days))
    temp = np.where(days.month <= 6, 21 + 6 * doy / 91, 30 - 2 * np.clip(doy - 150, 0, None) / 33)
    temp = temp + rng.normal(0, 1.8, len(days))
    hot = (days >= "2026-08-03") & (days <= "2026-08-16")   # hot spell
    temp = temp + np.where(hot, 4.0, 0.0)
    weather = pd.DataFrame({"date": days.date, "avg_temp_c": temp.round(1)})
    weather["cooling_degree_days"] = np.clip(weather.avg_temp_c - 18.0, 0, None).round(1)
    weather["heating_degree_days"] = np.clip(15.5 - weather.avg_temp_c, 0, None).round(1)

    rows = []
    for d in days:
        q = "Q2" if d.month <= 6 else "Q3"
        iso = d.strftime("%Y-%m-%d")
        if iso in (HOLIDAY_OK, HOLIDAY_TRAP):
            shift, factor, hol = ("off" if iso == HOLIDAY_OK else "2-shift"), 0.0, True
        elif d.dayofweek < 5:
            shift, factor, hol = "2-shift", 1.0, False
        elif d.dayofweek == 5 and d.isocalendar().week % 2 == 0:
            shift, factor, hol = "1-shift", 0.5, False
        else:
            shift, factor, hol = "off", 0.0, False
        for p, base in UNITS[q].items():
            units = int(round(base * factor * rng.normal(1, 0.15))) if factor else 0
            rows.append({"date": d.date(), "product": p, "units": units,
                         "shift_pattern": shift, "is_holiday": hol})
    return pd.DataFrame(rows), weather


def simulate(prod: pd.DataFrame, weather: pd.DataFrame, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (wide meter frame in kWh per 15 min, per-interval energy tagged by driver)."""
    ts = pd.date_range(START, pd.Timestamp(END) + pd.Timedelta(hours=23, minutes=45), freq="15min")
    day = ts.normalize()
    hour = ts.hour + ts.minute / 60
    units = prod.pivot(index="date", columns="product", values="units")
    units.index = pd.to_datetime(units.index)
    worked = units.sum(axis=1) > 0
    shift = prod.drop_duplicates("date").set_index("date").shift_pattern
    shift.index = pd.to_datetime(shift.index)
    end_hour = np.where(shift.reindex(day).values == "1-shift", 14, 22)
    running = worked.reindex(day).values & (hour >= 6) & (hour < end_hour)
    n_run = pd.Series(running, index=day).groupby(level=0).transform("sum").values.clip(min=1)

    kw = {}                                   # everything below is kW, converted at the end
    prod_kw = {}
    for p, line in LINE_OF.items():
        day_kwh = (units[p] * RATE[p]).reindex(day).values
        prod_kw[p] = np.where(running, day_kwh / n_run * 4, 0.0)
        kw[line] = 3.0 + SPLIT["line"] * prod_kw[p]
    total_prod = sum(prod_kw.values())

    cdd = weather.set_index(pd.to_datetime(weather.date)).cooling_degree_days.reindex(day).values
    work_day = worked.reindex(day).values
    shape = 0.5 + np.clip(np.sin((hour - 7) / 24 * 2 * np.pi), 0, None)     # afternoon peak
    shape_sum = pd.Series(shape, index=day).groupby(level=0).transform("sum").values
    cool_kw = COOL_KWH_PER_CDD * cdd * np.where(work_day, 1.0, COOL_NONWORK_FACTOR) * shape / shape_sum * 4
    leak_kw = np.where(ts >= LEAK_START, LEAK_KW, 0.0)

    kw["compressors"] = 30.0 + SPLIT["compressors"] * total_prod + leak_kw
    kw["furnace"] = 120.0 + SPLIT["furnace"] * total_prod
    kw["hvac"] = 40.0 + cool_kw
    kw["lighting"] = np.where(running, LIGHT_ON_KW, LIGHT_IDLE_KW)
    kw["utilities"] = np.full(len(ts), 150.0)

    # interval noise plus a day-to-day efficiency wobble, so the baseline is not a perfect fit
    day_factor = pd.Series(rng.normal(1, DAY_NOISE, day.nunique()), index=day.unique()).reindex(day).values
    subs = pd.DataFrame({m: v / 4 * rng.normal(1, 0.03, len(ts)) * day_factor for m, v in kw.items()}, index=ts)
    unmetered = subs.sum(axis=1) * 0.03 * rng.normal(1, 0.1, len(ts))     # sub-meters do not add up
    wide = subs.copy()
    wide.insert(0, "main", subs.sum(axis=1) + unmetered)

    drivers = pd.DataFrame({"product_" + p: v / 4 for p, v in prod_kw.items()}, index=ts)
    drivers["weather"] = cool_kw / 4
    drivers["leak"] = leak_kw / 4
    drivers["fixed"] = sum(kw.values()) / 4 - drivers.sum(axis=1)      # noise-free standing loads
    return wide, drivers


def add_faults(wide: pd.DataFrame, rng) -> tuple[pd.DataFrame, dict]:
    long = wide.reset_index(names="ts").melt(id_vars="ts", var_name="meter_id", value_name="kwh")
    long = long[["meter_id", "ts", "kwh"]]

    def window(meter, start, hours):
        t0 = pd.Timestamp(start)
        return (long.meter_id == meter) & (long.ts >= t0) & (long.ts < t0 + pd.Timedelta(hours=hours))

    gaps = [("main", "2026-04-14 09:00", 0.75), ("hvac", "2026-06-03 01:00", 6), ("main", "2026-07-21 10:00", 5)]
    drop = np.zeros(len(long), bool)
    for m, s, h in gaps:
        drop |= window(m, s, h).values
    long = long[~drop].copy()
    long.loc[(long.meter_id == "furnace") & (long.ts == "2026-05-20 10:00"), "kwh"] = -812.4   # reset
    long.loc[(long.meter_id == "line_2") & (long.ts == "2026-09-02 13:15"), "kwh"] = 250000.0  # reset dump
    dups = long[long.meter_id == "lighting"].sample(40, random_state=int(rng.integers(1_000_000)))
    long = pd.concat([long, dups]).sort_values(["meter_id", "ts"], kind="stable").reset_index(drop=True)
    info = {"missing_intervals": int(drop.sum()), "gaps": gaps, "duplicate_rows": 40,
            "resets": ["furnace 2026-05-20T10:00Z negative", "line_2 2026-09-02T13:15Z spike"]}
    return long, info


def tariff_frame(table) -> pd.DataFrame:
    rows = []
    for a, b, peak, off in table:
        rows.append({"valid_from": f"{a}T00:00:00Z", "valid_to": f"{b}T00:00:00Z", "period": "peak", "price_per_kwh": peak})
        rows.append({"valid_from": f"{a}T00:00:00Z", "valid_to": f"{b}T00:00:00Z", "period": "off_peak", "price_per_kwh": off})
    return pd.DataFrame(rows)


def price_series(ts: pd.DatetimeIndex, table) -> np.ndarray:
    price = np.zeros(len(ts))
    is_peak = (ts.hour >= PEAK[0]) & (ts.hour < PEAK[1])
    for a, b, peak, off in table:
        m = (ts >= a) & (ts < b)
        price[m] = np.where(is_peak[m], peak, off)
    return price


def true_split(wide, drivers, prod, table) -> dict:
    """The planted split, built from the tagged drivers (same attribution order as Annex A)."""
    ts = wide.index
    q3 = ts.month >= 7
    price = price_series(ts, table)
    E0, E1 = wide.main[~q3].sum(), wide.main[q3].sum()
    B0, B1 = (wide.main * price)[~q3].sum(), (wide.main * price)[q3].sum()
    p0 = B0 / E0
    u = prod.assign(q=np.where(pd.to_datetime(prod.date).dt.month >= 7, "Q3", "Q2")).groupby(["q", "product"]).units.sum().unstack()
    V0, V1 = u.loc["Q2"].sum(), u.loc["Q3"].sum()
    s0 = u.loc["Q2"] / V0
    unmetered_gain = 1.03                       # product energy also carries the 3% unmetered share
    volume_kwh = sum(RATE[p] * s0[p] * (V1 - V0) for p in RATE) * unmetered_gain
    mix_kwh = sum(RATE[p] * (u.loc["Q3", p] - s0[p] * V1) for p in RATE) * unmetered_gain
    weather_kwh = (drivers.weather[q3].sum() - drivers.weather[~q3].sum()) * unmetered_gain
    waste_kwh = (drivers.leak[q3].sum() - drivers.leak[~q3].sum()) * unmetered_gain
    # calendar effect (one more day, different working days) is counted inside volume
    volume_kwh += (drivers.fixed[q3].sum() - drivers.fixed[~q3].sum()) * unmetered_gain
    noise_kwh = (E1 - E0) - volume_kwh - mix_kwh - weather_kwh - waste_kwh   # random variation, no cause
    to_pp = lambda kwh: round(float(kwh * p0 / B0 * 100), 3)
    return {
        "bill_ref": round(float(B0), 2), "bill_new": round(float(B1), 2),
        "kwh_ref": round(float(E0), 1), "kwh_new": round(float(E1), 1),
        "total_change_pct": round(float((B1 - B0) / B0 * 100), 3),
        "split_pct_points": {"tariff": round(float((B1 - E1 * p0) / B0 * 100), 3), "volume": to_pp(volume_kwh),
                             "product_mix": to_pp(mix_kwh), "weather": to_pp(weather_kwh), "waste": to_pp(waste_kwh)},
        "random_noise_pct_points": to_pp(noise_kwh),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "sample"))
    ap.add_argument("--late", default=str(ROOT / "data" / "late_drop"))
    ap.add_argument("--truth", default=str(ROOT / "eval" / "truth.json"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out, late = Path(a.out), Path(a.late)
    out.mkdir(parents=True, exist_ok=True)
    late.mkdir(parents=True, exist_ok=True)

    prod, weather = calendar(rng)
    wide, drivers = simulate(prod, weather, rng)
    meter, faults = add_faults(wide, rng)
    tariff = tariff_frame(TARIFF)

    meter_out = meter.assign(ts=meter.ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ"), kwh=meter.kwh.round(3))
    meter_out.to_csv(out / "meter_15min.csv", index=False)
    tariff.to_csv(out / "tariff.csv", index=False)
    prod.to_csv(out / "production.csv", index=False)
    weather.to_csv(out / "weather.csv", index=False)

    # late drop: corrected tariff table + calibration fix for one sub-meter
    tariff_frame(TARIFF_CORRECTED).to_csv(late / "tariff.csv", index=False)
    fix = meter_out[(meter_out.meter_id == "hvac") & (meter_out.ts >= "2026-07-01")].copy()
    fix["kwh"] = (fix.kwh * HVAC_CALIBRATION).round(3)
    fix.to_csv(late / "meter_15min.csv", index=False)

    card = {"name": "synthetic-factory", "version": "1.0", "seed": a.seed, "generator": "scripts/generate_data.py",
            "date_range": [START, END],
            "row_counts": {"meter_15min": len(meter_out), "tariff": len(tariff), "production": len(prod), "weather": len(weather)},
            "meters": sorted(wide.columns), "faults_planted": faults,
            "privacy": "Fully synthetic. No real company or personal data."}
    (out / "data_card.json").write_text(json.dumps(card, indent=2, default=str))

    truth = {"seed": a.seed, "reference": "2026-Q2", "new": "2026-Q3",
             "before_late_drop": true_split(wide, drivers, prod, TARIFF),
             "after_late_drop": true_split(wide, drivers, prod, TARIFF_CORRECTED),
             "waste": {"meter": "compressors", "leak_kw": LEAK_KW, "leak_start": LEAK_START,
                       "true_annual_saving_kwh": LEAK_KW * 8760 * REPAIR_EFFECTIVENESS,
                       "repair_effectiveness_assumed": REPAIR_EFFECTIVENESS},
             "standing_waste": {"meter": "lighting", "idle_kw": LIGHT_IDLE_KW, "on_kw": LIGHT_ON_KW},
             "no_waste_meters": ["furnace", "hvac", "utilities"],
             "holiday_trap": HOLIDAY_TRAP}
    Path(a.truth).parent.mkdir(parents=True, exist_ok=True)
    Path(a.truth).write_text(json.dumps(truth, indent=2))
    print(json.dumps({"written_to": str(out), "row_counts": card["row_counts"],
                      "true_split": truth["before_late_drop"]}, indent=2))


if __name__ == "__main__":
    main()
