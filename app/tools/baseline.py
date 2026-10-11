"""Baseline (R2): expected daily energy from production volume, product mix and weather.

    daily kWh = base + working-day overhead + sum(rate_p x units_p) + k_cool x CDD + k_heat x HDD

Learned by linear regression on the HISTORY quarters: up to `baseline.history_quarters` (default 4)
complete quarters just before the quarter being explained. The quarter being explained is never in
the fit, otherwise the model would learn its waste as normal and hide it. A year of history covers
summer and winter, so the cooling and heating rates are learned from real variation instead of being
stretched beyond one quarter. Linear on purpose: each term is one cause, so the effects separate
cleanly and add up.

The fitted model gives the EXPECTED energy for any day from that day's production and weather;
"actual minus expected" is the energy the drivers cannot explain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def feature_names(products: list[str], use_mix: bool) -> list[str]:
    units = [f"units_{p}" for p in products] if use_mix else ["units_total"]
    return ["const", "working"] + units + ["cdd", "hdd"]


def features(daily: pd.DataFrame, products: list[str], use_mix: bool) -> pd.DataFrame:
    X = pd.DataFrame(index=daily.index)
    X["const"] = 1.0
    X["working"] = daily.working.astype(float)
    for c in feature_names(products, use_mix)[2:]:
        X[c] = daily[c].astype(float)
    return X


def fit_days(ds, quarters: list[str] | str) -> pd.DataFrame:
    """History days clean enough to learn from."""
    quarters = [quarters] if isinstance(quarters, str) else list(quarters)
    d = ds.daily[ds.daily.quarter.isin(quarters)]
    ok = (d.full_day & (d.gap_hours <= ds.cfg["cleaning"]["max_gap_hours_for_fit"])
          & ~d.logged_but_idle & ~d.production_missing & ~d.weather_missing)
    return d[ok]


def _physical(names: list[str]) -> list[str]:
    """Rates that cannot be negative: energy per unit and energy per degree-day."""
    return [n for n in names if n.startswith("units_") or n in ("cdd", "hdd")]


def constrained_lstsq(X: np.ndarray, y: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Least squares where physical rates may not go below zero. A rate that comes out negative
    is fixed at zero and the fit is repeated without it (most negative first)."""
    active, dropped = list(range(len(names))), []
    while True:
        coef = np.linalg.lstsq(X[:, active], y, rcond=None)[0]
        neg = [(c, i) for c, i in zip(coef, active) if names[i] in _physical(names) and c < 0]
        if not neg:
            break
        worst = min(neg)[1]
        active.remove(worst)
        dropped.append(names[worst])
    full = np.zeros(len(names))
    full[active] = coef
    return full, dropped


def _week_ids(index: pd.DatetimeIndex, block_days: int) -> np.ndarray:
    return ((index - index.min()).days // max(int(block_days), 1)).to_numpy()


def quality(X: np.ndarray, y: np.ndarray, names: list[str], weeks: np.ndarray) -> dict:
    """Out-of-sample quality: each week is predicted by a model fitted without it (leave-one-week-out)."""
    err = np.full(len(y), np.nan)
    for w in np.unique(weeks):
        test = weeks == w
        if test.all() or (~test).sum() <= len(names):
            continue
        coef, _ = constrained_lstsq(X[~test], y[~test], names)
        err[test] = y[test] - X[test] @ coef
    ok = ~np.isnan(err)
    if ok.sum() < 5 or y.mean() == 0:
        return {"cv_rmse_pct": None, "nmbe_pct": None}
    m = float(y[ok].mean())
    return {"cv_rmse_pct": round(float(np.sqrt(np.mean(err[ok] ** 2)) / m * 100), 2),
            "nmbe_pct": round(float(err[ok].sum() / ((ok.sum() - len(names)) * m) * 100), 2)}


def fit_baseline(ds, cfg: dict, quarters: list[str] | str) -> dict:
    quarters = [quarters] if isinstance(quarters, str) else list(quarters)
    if not quarters:
        return {"ok": False, "reason": "no complete earlier quarter is loaded, so there is no history to learn the baseline from"}
    use_mix = cfg["baseline"]["use_product_mix"]
    label = ", ".join(quarters)
    days = fit_days(ds, quarters)
    X, y = features(days, ds.products, use_mix), days.kwh_main.astype(float)
    names = list(X.columns)
    # a column that never varies in the history (e.g. no heating days) cannot be learned
    keep = ["const"] + [c for c in names[1:] if X[c].nunique() > 1]
    warnings = [f"'{c}' did not vary in the history ({label}), so its effect cannot be estimated and is set to 0"
                for c in names if c not in keep]
    if len(days) < len(keep) + 5:
        return {"ok": False, "reason": f"only {len(days)} usable days in {label}; too few to fit a baseline"}

    _, negative = constrained_lstsq(X[keep].values, y.values, keep)
    for c in negative:
        warnings.append(f"the data gave a negative rate for '{c}', which is physically impossible; it is set to 0, "
                        f"so this effect cannot be separated and any real effect lands in waste")
    keep = [c for c in keep if c not in negative]
    model = sm.OLS(y.values, X[keep].values).fit()
    beta = dict.fromkeys(names, 0.0)
    beta.update(dict(zip(keep, model.params)))
    ci = model.conf_int(alpha=0.05)
    for c in ("cdd", "hdd"):
        lo, hi = dict(zip(keep, ci)).get(c, (0, 0))
        if c in keep and lo <= 0:
            warnings.append(f"the weather rate ('{c}') is weakly determined from {label}: 95% interval {lo:.0f} to {hi:.0f} kWh "
                            f"per degree-day includes zero, so the weather and waste slices have wide ranges")
    q = quality(X[keep].values, y.values, keep, _week_ids(days.index, 7))          # always scored week by week
    blocks = _week_ids(days.index, cfg["baseline"].get("bootstrap_block_days", 1))
    if q["cv_rmse_pct"] is not None and q["cv_rmse_pct"] > cfg["baseline"].get("max_cv_rmse_pct", 15):
        warnings.append(f"baseline CV(RMSE) is {q['cv_rmse_pct']}% (out of sample, by week), above the "
                        f"{cfg['baseline'].get('max_cv_rmse_pct', 15)}% limit: expected energy is uncertain")
    if q["nmbe_pct"] is not None and abs(q["nmbe_pct"]) > cfg["baseline"].get("max_abs_nmbe_pct", 5):
        warnings.append(f"baseline NMBE is {q['nmbe_pct']}%, outside +/-{cfg['baseline'].get('max_abs_nmbe_pct', 5)}%")
    total = ds.daily[ds.daily.quarter.isin(quarters)]
    return {
        "ok": True, "names": names, "keep": keep, "beta": beta, "use_mix": use_mix, "quarters": quarters,
        "r2": float(model.rsquared), "n_days": int(len(days)), "days_excluded": int(len(total) - len(days)),
        "resid_std_kwh": float(np.sqrt(model.scale)), **q,
        "conf_int": {k: [float(lo), float(hi)] for k, (lo, hi) in zip(keep, ci)},
        "X": X[keep].values, "y": y.values, "blocks": blocks, "warnings": warnings,
    }


def bootstrap_betas(fit: dict, n: int, seed: int) -> list[dict]:
    """Refit on resampled history days: how much the rates move if the sample were different. Days are drawn in
    blocks of baseline.bootstrap_block_days (1 = single days; 7 = whole weeks, for autocorrelated data)."""
    rng = np.random.default_rng(seed)
    X, y, out = fit["X"], fit["y"], []
    blocks = fit.get("blocks")
    if blocks is None:
        blocks = np.arange(len(y))
    groups = [np.flatnonzero(blocks == w) for w in np.unique(blocks)]
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        i = np.concatenate([groups[k] for k in pick])
        coef, _ = constrained_lstsq(X[i], y[i], fit["keep"])
        b = dict.fromkeys(fit["names"], 0.0)
        b.update(dict(zip(fit["keep"], coef)))
        out.append(b)
    return out


def totals(ds, period: str, products: list[str], days: pd.DataFrame | None = None) -> dict:
    """Totals of every baseline input over a quarter (or over the given days)."""
    d = ds.daily[ds.daily.quarter == period] if days is None else days
    out = {"const": float(len(d)), "working": float(d.working.sum()), "cdd": float(d.cdd.sum()),
           "hdd": float(d.hdd.sum()), "units_total": float(d.units_total.sum())}
    for p in products:
        out[f"units_{p}"] = float(d[f"units_{p}"].sum())
    return out


def predict_total(beta: dict, tot: dict) -> float:
    return float(sum(beta[k] * tot.get(k, 0.0) for k in beta))


def predict_daily(beta: dict, daily: pd.DataFrame, products: list[str], use_mix: bool) -> pd.Series:
    """Expected kWh for each day from its production and weather."""
    X = features(daily, products, use_mix)
    return pd.Series(X.values @ np.array([beta[c] for c in X.columns]), index=daily.index)
