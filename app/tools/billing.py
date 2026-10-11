"""Billing (R4): every 15-minute reading is priced at the tariff in force at that moment,
so a price change part-way through a quarter is handled without special cases.

Two tariff shapes are accepted, row by row:
  - block rows: a peak/off_peak row shorter than one day is a dated time block (e.g. 07:00-22:00);
    its own span and label are used as written;
  - long rows: a peak/off_peak row of a day or more covers only the configured peak (or off-peak) hours.
A 'flat' row applies to every hour it spans.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ONE_DAY = pd.Timedelta(days=1)


def is_peak(local: pd.DatetimeIndex, cfg: dict) -> np.ndarray:
    """Configured peak hours (used only for long-row tariffs)."""
    start, end = cfg["peak_hours"]["start"], cfg["peak_hours"]["end"]
    return np.asarray((local.hour >= start) & (local.hour < end))


def tariff_mode(tariff: pd.DataFrame) -> str:
    """'block' when every peak/off_peak row is a dated time block, 'map' when none is, else 'mixed'."""
    tou = tariff[tariff.period != "flat"]
    if tou.empty:
        return "flat"
    short = (tou.valid_to - tou.valid_from) < ONE_DAY
    return "block" if short.all() else ("map" if not short.any() else "mixed")


def interval_prices(utc: pd.DatetimeIndex, local: pd.DatetimeIndex, tariff: pd.DataFrame, cfg: dict):
    """Price per kWh and price label for each interval. Validity is matched in UTC.
    Returns (prices, labels, covered, stats); covered = a tariff row really matched the interval (not filled)."""
    n = len(utc)
    price = np.full(n, np.nan)
    label = np.full(n, "", dtype=object)
    hits = np.zeros(n, dtype=np.int16)
    peak = is_peak(local, cfg)
    t = tariff.sort_values("valid_from")
    starts, ends = utc.searchsorted(t.valid_from.values), utc.searchsorted(t.valid_to.values)
    for row, a, b in zip(t.itertuples(index=False), starts, ends):
        if a >= b:
            continue
        sl = slice(a, b)
        m = np.ones(b - a, dtype=bool)
        if row.period != "flat" and (row.valid_to - row.valid_from) >= ONE_DAY:
            m = peak[sl] if row.period == "peak" else ~peak[sl]
        idx = np.arange(a, b)[m]
        price[idx] = row.price_per_kwh
        label[idx] = row.period
        hits[idx] += 1
    s = pd.Series(price, index=utc)
    covered = pd.Series(~np.isnan(price), index=utc)
    unpriced = int(s.isna().sum())
    # never drop energy from the bill: use the nearest known price and report it. If no row covers any
    # reading at all, prices stay missing and the quarter is reported as having no tariff (never analysed).
    if unpriced:
        s = s.ffill().bfill()
        have = np.flatnonzero(label != "")
        if len(have):      # nearest earlier label (later one at the very start)
            pos = np.maximum.accumulate(np.where(label != "", np.arange(n), -1))
            label = label[np.where(pos >= 0, pos, have[0])]
        else:
            label[:] = "flat"
    return s, pd.Series(label, index=utc), covered, {
        "unpriced_intervals": unpriced, "overlapping_intervals": int((hits > 1).sum()),
        "tariff_rows": int(len(tariff)), "mode": tariff_mode(tariff) if len(tariff) else "none"}


def prices_as_of(when: pd.Timestamp, tariff: pd.DataFrame) -> dict:
    """The latest price of each label that had started by `when` (used for 'what if the tariff had not changed').
    Works for block tariffs too, where only one block is live at any moment."""
    started = tariff[tariff.valid_from <= when].sort_values("valid_from")
    if started.empty:
        started = tariff.sort_values("valid_from").head(3)
    last = started.groupby("period").tail(1)
    return {r.period: float(r.price_per_kwh) for r in last.itertuples(index=False)}


def reprice_at(ds, prices: dict) -> pd.Series:
    """Interval prices if each label had the given price. Labels missing from `prices` keep the actual price."""
    lab = ds.price_label.values
    vals = np.array([prices.get(x, np.nan) for x in lab], dtype=float)
    if "flat" in prices and not ({"peak", "off_peak"} & set(prices)):
        vals = np.where(np.isnan(vals), prices["flat"], vals)      # an all-flat old tariff covers every hour
    return pd.Series(vals, index=ds.wide.index).fillna(ds.price)


def bill(ds, period: str, price: pd.Series | None = None, mask: np.ndarray | None = None) -> dict:
    """Bill and energy for one quarter from the main meter (the bill-authoritative total)."""
    if mask is None:
        mask = (ds.quarter_of_interval() == period).values
    kwh = ds.wide["main"].values[mask]
    p = (ds.price if price is None else price).values[mask]
    peak = ds.price_label.values[mask] == "peak"
    energy, money = float(kwh.sum()), float((kwh * p).sum())
    return {"period": period, "kwh": energy, "bill": money,
            "avg_price": money / energy if energy else 0.0,
            "peak_kwh_share_pct": float(kwh[peak].sum() / energy * 100) if energy else 0.0,
            "days": int(mask.sum() / 96)}
