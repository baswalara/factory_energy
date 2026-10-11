"""Cleaning (R1): put the meters on one regular 15-minute grid, handle resets, duplicates
and gaps, align with tariff, production and weather, and build the daily table.

Everything that was changed is counted in Dataset.log, and shown by validation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .store import Store
from .tools.billing import interval_prices

OFF_SHIFTS = {"", "off", "none", "closed", "holiday", "0", "no shift", "nan"}


@dataclass
class Dataset:
    cfg: dict
    version: int
    wide: pd.DataFrame            # cleaned kWh per 15 min; index = UTC, columns = meters
    local: pd.DatetimeIndex       # the same instants in factory local time
    price: pd.Series              # price per kWh for each interval
    price_label: pd.Series        # tariff label of each interval: peak, off_peak or flat
    productive: pd.Series         # True while production lines are running
    daily: pd.DataFrame           # one row per local day
    products: list[str]
    submeters: list[str]
    tariff: pd.DataFrame
    production: pd.DataFrame
    weather: pd.DataFrame
    log: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return "main" in self.wide.columns and len(self.wide) > 0

    def quarter_of_interval(self) -> pd.Series:
        return pd.Series(quarter_label(self.local), index=self.wide.index)


def quarter_label(idx: pd.DatetimeIndex) -> np.ndarray:
    return (idx.year.astype(str) + "-Q" + idx.quarter.astype(str)).values


def parse_period(text: str | None) -> str | None:
    """'2026-Q3', '2026 q3', 'Q3 2026' -> '2026-Q3'. None when it is not a quarter label."""
    if not text:
        return None
    m = re.search(r"(20\d{2})\s*[-_/ ]?\s*Q\s*([1-4])", text, re.I) or None
    if m:
        return f"{m.group(1)}-Q{m.group(2)}"
    m = re.search(r"Q\s*([1-4])\s*[-_/ ]?\s*(20\d{2})", text, re.I)
    return f"{m.group(2)}-Q{m.group(1)}" if m else None


_RANGE = re.compile(r"(\d{1,2})(?::(\d{2}))?\s*-\s*(\d{1,2})(?::(\d{2}))?")


def shift_ranges(pattern, cfg: dict) -> list[tuple[float, float]]:
    """Local working hours of a shift_pattern value. Accepts labels from config ('2-shift'), clock ranges
    ('06:00-14:00|14:00-22:00', '06:00-14:00,14:00-22:00') and off words ('OFF', 'none'). Unknown labels use
    default_shift_hours."""
    text = str(pattern if pattern is not None else "").strip().lower()
    if text in OFF_SHIFTS:
        return []
    if text in cfg["shift_hours"]:
        h = cfg["shift_hours"][text]
        return [(float(h[0]), float(h[1]))] if h else []
    found = _RANGE.findall(text)
    if found:
        return [(int(a) + int(b or 0) / 60, int(c) + int(d or 0) / 60) for a, b, c, d in found]
    h = cfg["default_shift_hours"]
    return [(float(h[0]), float(h[1]))]


def clean_meter(s: pd.Series, cfg: dict) -> tuple[pd.Series, dict, pd.Series]:
    """One meter on the full grid -> (clean series, counts, mask of long-gap-filled intervals)."""
    c = cfg["cleaning"]
    missing = int(s.isna().sum())
    neg = s < 0
    q99 = s[s >= 0].quantile(0.99) if s.notna().any() else np.nan
    spike = (s > c["spike_factor"] * q99) if q99 and q99 > 0 else pd.Series(False, index=s.index)
    s = s.mask(neg | spike)

    nan = s.isna()
    run = (nan != nan.shift()).cumsum()
    run_len = nan.groupby(run).transform("sum")
    short = nan & (run_len <= c["max_interp_intervals"])
    s = s.where(~short, s.interpolate(limit_area="inside"))

    long = s.isna()
    if long.any():      # long gaps: typical value for that time of day on that kind of day
        key = [s.index.dayofweek >= 5, s.index.hour * 4 + s.index.minute // 15]
        s = s.where(~long, s.groupby(key).transform("median"))
        s = s.interpolate().bfill().ffill().fillna(0.0)
    stats = {"missing_intervals": missing, "negative_resets": int(neg.sum()), "spike_resets": int(spike.sum()),
             "interpolated": int(short.sum()), "profile_filled": int(long.sum())}
    return s, stats, long


def build_dataset(store: Store, cfg: dict) -> Dataset:
    meter, tariff = store.read("meter_15min"), store.read("tariff")
    production, weather = store.read("production"), store.read("weather")
    log: dict = {"meters": {}, "notes": []}

    if meter.empty:
        empty = pd.DataFrame()
        return Dataset(cfg, store.data_version(), empty, pd.DatetimeIndex([]), pd.Series(dtype=float),
                       pd.Series(dtype=object), pd.Series(dtype=bool), empty, [], [], tariff, production, weather, log)

    meter["ts"] = pd.to_datetime(meter.ts).dt.floor("15min")           # snap stray seconds to the grid
    raw = meter.groupby(["ts", "meter_id"]).kwh.last().unstack("meter_id")
    grid = pd.date_range(raw.index.min(), raw.index.max(), freq="15min")
    raw = raw.reindex(grid)
    wide, long_fill = {}, {}
    for m in raw.columns:
        first, last = raw[m].first_valid_index(), raw[m].last_valid_index()
        s, stats, lf = clean_meter(raw[m].loc[first:last], cfg)
        wide[m], long_fill[m] = s.reindex(grid).fillna(0.0), lf.reindex(grid).fillna(False)
        log["meters"][m] = stats
    wide = pd.DataFrame(wide, index=grid)
    wide.index.name = "ts"

    tz = cfg["factory_timezone"]
    local = grid.tz_localize("UTC").tz_convert(tz).tz_localize(None)
    price, price_label, covered, price_stats = interval_prices(grid, local, tariff, cfg)
    log["pricing"] = price_stats

    submeters = [m for m in wide.columns if m != "main"]
    lines = [m for m in submeters if any(m.startswith(p) for p in cfg["production_meter_prefixes"])]
    day = local.normalize()

    # production and weather as daily tables
    prod = production.copy()
    prod["date"] = pd.to_datetime(prod.date)
    products = sorted(prod["product"].unique().tolist()) if len(prod) else []
    units = prod.pivot_table(index="date", columns="product", values="units", aggfunc="sum").fillna(0.0)
    units.columns = [f"units_{p}" for p in units.columns]
    cal = prod.groupby("date").agg(shift_pattern=("shift_pattern", "first"), is_holiday=("is_holiday", "max"))
    wx = weather.copy()
    wx["date"] = pd.to_datetime(wx.date)
    wx = wx.set_index("date")[["avg_temp_c", "cooling_degree_days", "heating_degree_days"]]

    days = pd.DatetimeIndex(sorted(set(day)))
    daily = pd.DataFrame(index=days)
    daily.index.name = "date"
    if "main" in wide:
        daily["kwh_main"] = wide["main"].groupby(day).sum().reindex(days).values
        daily["gap_hours"] = long_fill["main"].groupby(day).sum().reindex(days).values / 4
        daily["intervals"] = wide["main"].groupby(day).size().reindex(days).values
        daily["main_readings"] = raw["main"].notna().groupby(day).sum().reindex(days).fillna(0).values
    else:
        daily["kwh_main"], daily["gap_hours"], daily["intervals"], daily["main_readings"] = 0.0, 24.0, 0, 0
    daily = daily.join(units).join(cal).join(wx)
    daily["production_missing"] = daily.shift_pattern.isna()
    daily["weather_missing"] = daily.cooling_degree_days.isna()
    unit_cols = [f"units_{p}" for p in products]
    daily[unit_cols] = daily[unit_cols].fillna(0.0)
    daily["units_total"] = daily[unit_cols].sum(axis=1) if unit_cols else 0.0
    daily["shift_pattern"] = daily.shift_pattern.fillna("").astype(str)
    daily["is_holiday"] = daily.is_holiday.eq(True)
    for c in ["cooling_degree_days", "heating_degree_days"]:
        daily[c] = daily[c].interpolate().bfill().ffill().fillna(0.0)
    daily = daily.rename(columns={"cooling_degree_days": "cdd", "heating_degree_days": "hdd"})

    # productive intervals: from the production-line meters only
    if lines:
        load = wide[lines].sum(axis=1)
        level = load[load > 0].quantile(0.95) if (load > 0).any() else 0.0
        productive = load > cfg["productive_threshold"] * level
        log["idle_basis"] = f"production-line meters {lines}"
        active = productive.groupby(day).sum().reindex(days).values >= 4          # at least 1 hour of running
        daily["lines_active"] = active
    else:
        shift = daily.shift_pattern.reindex(day).values
        hour = np.asarray(local.hour + local.minute / 60)
        on = np.zeros(len(grid), dtype=bool)
        for pattern in set(shift):
            sel = shift == pattern
            for a, b in shift_ranges(pattern, cfg):
                on |= sel & (((hour >= a) & (hour < b)) if a < b else ((hour >= a) | (hour < b)))
        worked = daily.units_total.reindex(day).values > 0
        productive = pd.Series(worked & on, index=grid)
        log["idle_basis"] = "shift_pattern hours from config (no production-line meters found)"
        daily["lines_active"] = daily.units_total > 0
    productive = pd.Series(np.asarray(productive), index=grid)

    daily["logged_but_idle"] = (daily.units_total > 0) & ~daily.lines_active
    daily["holiday_as_shift"] = daily.is_holiday & ~daily.shift_pattern.str.lower().isin(OFF_SHIFTS)
    daily["working"] = (daily.units_total > 0) & daily.lines_active
    daily["quarter"] = quarter_label(days)
    # a day is priced when tariff rows really cover it (>= 95% of intervals), not when a price was carried forward
    daily["priced"] = (covered.groupby(day).mean().reindex(days).fillna(0.0) >= 0.95).values
    daily["has_meter"] = daily.main_readings >= 48                    # at least half a day of real readings
    daily["full_day"] = daily.main_readings >= 96 - 4 * cfg["cleaning"]["max_gap_hours_for_fit"]

    return Dataset(cfg, store.data_version(), wide, local, price, price_label, productive, daily, products, submeters,
                   tariff, production, weather, log)


_cache: dict = {}


def get_dataset(store: Store, cfg: dict) -> Dataset:
    """Cleaned data, rebuilt only when new data has been loaded."""
    key = (store.path, store.data_version())
    if _cache.get("key") != key:
        _cache["key"], _cache["ds"] = key, build_dataset(store, cfg)
    return _cache["ds"]


def quarter_days(q: str) -> int:
    year, qn = int(q[:4]), int(q[-1])
    return sum(pd.Period(f"{year}-{m:02d}", freq="M").days_in_month for m in range(3 * qn - 2, 3 * qn + 1))


def period_status(ds: Dataset) -> dict:
    """Per quarter: how many of its days have real main-meter readings, production rows, weather rows and a tariff.
    Gap-filled meter values never count, so a quarter that was never uploaded cannot look complete."""
    if not ds.ok:
        return {}
    cov, min_days = ds.cfg["min_quarter_coverage"], ds.cfg.get("min_partial_days", 7)
    out = {}
    for q, d in ds.daily.groupby("quarter"):
        n = quarter_days(q)
        usable = d.has_meter & ~d.production_missing & ~d.weather_missing & d.priced
        row = {"days_in_quarter": n, "meter_days": int(d.has_meter.sum()), "tariff_days": int((d.has_meter & d.priced).sum()),
               "production_days": int((d.has_meter & ~d.production_missing).sum()),
               "weather_days": int((d.has_meter & ~d.weather_missing).sum()), "usable_days": int(usable.sum())}
        row["complete"] = row["usable_days"] / n >= cov
        row["partial"] = not row["complete"] and row["usable_days"] >= min_days
        missing = [k for k in ("meter_days", "production_days", "weather_days", "tariff_days") if row[k] / n < cov]
        row["missing"] = [k.replace("_days", "") for k in missing]
        out[q] = row
    return out


def available_periods(ds: Dataset) -> list[str]:
    """Complete quarters (meter, production and weather data on enough days), oldest first."""
    return sorted(q for q, r in period_status(ds).items() if r["complete"])


def partial_periods(ds: Dataset) -> list[str]:
    """Quarters with some usable days but not enough to compare bills (for example a quarter in progress)."""
    return sorted(q for q, r in period_status(ds).items() if r["partial"])


def history_window(ds: Dataset, new: str, n: int | None = None) -> list[str]:
    """Up to n complete quarters just before `new`: the data the baseline learns 'normal' from."""
    n = n or ds.cfg["baseline"].get("history_quarters", 4)
    return [q for q in available_periods(ds) if q < new][-n:]


def describe_periods(ds: Dataset) -> str:
    """One line for messages: what is loaded and what is incomplete."""
    st = period_status(ds)
    full = [q for q, r in st.items() if r["complete"]]
    other = [f"{q} ({r['usable_days']} of {r['days_in_quarter']} days usable"
             + (f"; missing {', '.join(r['missing'])}" if r["missing"] else "") + ")"
             for q, r in st.items() if not r["complete"]]
    text = f"Complete quarters: {', '.join(full) or 'none'}."
    return text + (f" Incomplete: {'; '.join(other)}." if other else "")
