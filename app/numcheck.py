"""Number-faithfulness check: every number in the brief must come from a tool result.

Runs in code after the LLM writes. A number passes when it equals some tool value
rounded to the same number of decimals. Anything else is listed as unmatched; the
brief is then rewritten once, and replaced by a fixed template if it fails again.
"""
from __future__ import annotations

import re

NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> list[tuple[float, int]]:
    """(value, decimals) for each number written in the text."""
    out = []
    for tok in NUMBER.findall(text):
        tok = tok.replace(",", "").lstrip("+-")
        if tok:
            out.append((float(tok), len(tok.split(".")[1]) if "." in tok else 0))
    return out


def allowed_values(payload) -> list[float]:
    """Every number in the tool results, including numbers inside their text fields (dates, hours, labels)."""
    vals: list[float] = []

    def walk(x):
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            vals.append(abs(float(x)))
        elif isinstance(x, str):
            vals.extend(v for v, _ in numbers_in(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(payload)
    return vals


def check_numbers(text: str, payload) -> dict:
    allowed = allowed_values(payload)
    found = numbers_in(text)
    unmatched = []
    for value, decimals in found:
        tol = 0.5 * 10 ** (-decimals) + 1e-9
        if not any(abs(a - value) <= tol for a in allowed):
            unmatched.append(value)
    return {"ok": not unmatched, "numbers_checked": len(found), "unmatched": unmatched}
