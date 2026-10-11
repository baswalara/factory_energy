"""FastAPI app: the four endpoints of the contract plus two admin helpers.

    POST /analyze           ask a question
    POST /ingest            add a new quarter, or correct data, while running (multipart: file + table)
    GET  /health            API, data store and LLM status
    GET  /trace/{run_id}    audit record of one answer
    GET  /validate          the validation checks on the loaded data
    GET  /periods           which quarters are complete or partial, and the history each one's baseline uses
    POST /admin/load        loader: load every Annex B CSV found in a server-side folder

/analyze and /ingest are plain `def`, so FastAPI runs them in its threadpool and the pandas work
never blocks /health.
"""
from __future__ import annotations

import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .cleaning import available_periods, describe_periods, get_dataset, partial_periods, period_status
from .config import llm_settings, load_config
from .graph import build_graph, run_question
from .llm import LLM
from .loader import load_csv_bytes, load_dir
from .schemas import AnalyzeRequest, AnalyzeResponse, IngestResponse
from .store import TABLES, Store
from .tools.bridge import baseline_quarters, decompose
from .validation import validate

app = FastAPI(title="Factory Energy Cost Decomposition and Savings Agent", version="1.0")
cfg = load_config()
store = Store()
llm = LLM(llm_settings())
graph = build_graph(store, cfg, llm)


def _bridge_now() -> dict | None:
    ds = get_dataset(store, cfg)
    have = available_periods(ds)
    if len(have) < 2:
        return None
    b = decompose(ds, cfg, have[-1], have[-2])
    return b if b.get("ok") else None


def _table_name(table: str | None, meta: str | None, filename: str | None) -> str:
    """The guide says 'a JSON naming the table'. Accept {"table": "..."} in either form field, a bare name, or the file name."""
    for raw in (meta, table):
        if raw:
            raw = raw.strip()
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and obj.get("table"):
                    return str(obj["table"]).strip().lower()
                if isinstance(obj, str):
                    return obj.strip().lower()
            except json.JSONDecodeError:
                return raw.lower()
    stem = (filename or "").rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
    return next((t for t in TABLES if stem.startswith(t)), stem)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> dict:
    return run_question(graph, store, llm, req.question, req.period, req.compare_to)


def _periods() -> dict:
    ds = get_dataset(store, cfg)
    st = period_status(ds)
    have = available_periods(ds)
    for q, row in st.items():
        prev = [h for h in have if h < q]
        row["compared_with"] = prev[-1] if prev else None
        row["baseline_history"] = baseline_quarters(ds, cfg, q, row["compared_with"])[0] if (row["complete"] or row["partial"]) else []
    return {"quarters": st, "complete": have, "partial": partial_periods(ds), "current": have[-1] if have else None,
            "summary": describe_periods(ds), "history_quarters_configured": cfg["baseline"].get("history_quarters", 4)}


@app.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...), table: str | None = Form(None), meta: str | None = Form(None)) -> dict:
    name = _table_name(table, meta, file.filename)
    if name not in TABLES:
        return {"rows_loaded": 0, "rows_rejected": 0, "status": "rejected", "table": name,
                "detail": f"unknown table '{name}'; expected one of {TABLES}"}
    before = _bridge_now()
    if before and not store.previous_bridge(before["period"], before["compare_to"]):
        store.save_bridge("pre-ingest", before["period"], before["compare_to"], before)   # keep a 'before' to compare with
    result = load_csv_bytes(store, name, file.file.read(), file.filename or "")
    if result["rows_loaded"]:
        store.log_ingest(f"{name} ({result['rows_loaded']} rows from {file.filename})")
    report = validate(store, get_dataset(store, cfg))
    result["validation"] = {"status": report["status"],
                            "problems": [c for c in report["checks"] if c["status"] in ("WARN", "FAIL")]}
    result["periods"] = _periods()
    after = _bridge_now() if result["rows_loaded"] else None
    if after:
        store.save_bridge("post-ingest", after["period"], after["compare_to"], after)
        result["rerun"] = {"period": after["period"], "compare_to": after["compare_to"],
                           "total_change_pct": {"before": before["bill"]["change_pct"] if before else None, "after": after["bill"]["change_pct"]},
                           "bridge": [{"cause": r["cause"], "after": r["pct_points"],
                                       "before": next((x["pct_points"] for x in before["bridge"] if x["cause"] == r["cause"]), None) if before else None}
                                      for r in after["bridge"]]}
    return result


@app.get("/health")
def health() -> dict:
    try:
        counts = {t: store.count(t) for t in TABLES}
        ds = get_dataset(store, cfg)
        have = available_periods(ds)
        data = {"status": "ok" if ds.ok else "empty", "rows": counts, "periods": have,
                "partial_periods": partial_periods(ds), "current_quarter": have[-1] if have else None,
                "baseline_history": baseline_quarters(ds, cfg, have[-1], have[-2] if len(have) > 1 else None)[0] if have else [],
                "meters": list(ds.wide.columns) if ds.ok else [], "data_version": store.data_version()}
    except Exception as e:
        data = {"status": f"error: {type(e).__name__}: {e}"}
    llm_status = llm.status()
    ok = data["status"] in ("ok", "empty") and not llm_status.startswith(("error", "missing", "unreachable"))
    return {"status": "ok" if ok else "degraded", "api": "ok", "data_store": data, "llm": llm_status}


@app.get("/trace/{run_id}")
def trace(run_id: str) -> dict:
    rec = store.get_trace(run_id)
    if not rec:
        raise HTTPException(404, f"no trace for run_id '{run_id}'")
    return rec


@app.get("/validate")
def validation() -> dict:
    return validate(store, get_dataset(store, cfg))


@app.get("/periods")
def periods() -> dict:
    return _periods()


@app.post("/admin/load")
def admin_load(folder: str, reset: bool = False) -> dict:
    if reset:
        store.clear()
    results = load_dir(store, folder)
    if not results:
        raise HTTPException(400, f"no Annex B CSV files found in '{folder}'")
    return {"loaded": results, "validation": validate(store, get_dataset(store, cfg), f"{folder}/data_card.json")}
