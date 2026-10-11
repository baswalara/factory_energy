"""The one LangGraph flow. Six nodes, at most two LLM calls (label the question, word the brief).

    parse -> periods -> analyse -> write -> check -> respond

    Early exits (refused, not_in_data, clarification_needed) go from periods or analyse straight to respond.
    check -> write happens once at most, when the brief has a number no tool produced; after that a fixed template is used.

Why a graph when the steps are fixed: it gives the routing by question type, the capped
retry branch and one place where every step is recorded for the trace.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from . import analysis
from .cleaning import get_dataset
from .intents import parse_llm, parse_rules
from .numcheck import check_numbers
from .writer import llm_brief, template_brief

REFUSED = ("I can't do that. Every number I report has to come from a calculation on the loaded data, "
           "and actions are suggestions that a manager approves. I can explain the bill change, show energy by asset, "
           "idle load, or savings actions with ranges.")
NOT_IN_DATA = ("This is not in the data. The loaded files hold meter readings, the tariff, production and weather for {periods}. "
               "They say nothing about this, so I will not guess.")
UNCLEAR = ("I am not sure what you are asking. I can answer: why the bill changed, expected (baseline) energy against actual, "
           "which assets use the most energy, how much is used when nothing is produced, what to fix first and how fast it "
           "pays back, or a tariff what-if. "
           "Loaded quarters: {periods}.")


class State(TypedDict, total=False):
    run_id: str
    question: str
    period: str | None
    compare_to: str | None
    intent: str
    params: dict
    new: str
    ref: str | None
    exit: str
    message: str
    result: dict
    brief: str
    brief_source: str
    check: dict
    bad_numbers: list
    attempts: int
    steps: list
    llm_calls: int
    tokens: int
    response: dict


def build_graph(store, cfg, llm):
    def timed(name, fn):
        def node(state: State) -> dict:
            t = time.perf_counter()
            out = fn(state) or {}
            ms = round((time.perf_counter() - t) * 1000, 1)
            for s in out.get("steps", [])[len(state.get("steps", [])):]:
                s.setdefault("node", name)
                s.setdefault("ms", ms)
            return out
        return node

    def parse(state: State) -> dict:
        steps = list(state.get("steps", []))
        guard = parse_rules(state["question"])
        calls, tokens, source = 0, 0, "rules"
        parsed = guard
        # rules first; the LLM labels only what the rules could not (unclear, or a what-if without a scenario)
        need_llm = guard["intent"] == "unclear" or (guard["intent"] == "what_if" and len(guard) == 1)
        if need_llm and not llm.mock:
            got, tokens = parse_llm(state["question"], llm)
            calls = 1
            if got and got["intent"] != "unclear":
                parsed, source = got, "llm"
            else:
                source = "rules (LLM label unusable)" if not got else "rules (LLM also unsure)"
        steps.append({"agent": "router", "tool": "label_question", "output": {**parsed, "source": source}})
        return {"intent": parsed["intent"], "params": {k: v for k, v in parsed.items() if k != "intent"},
                "steps": steps, "llm_calls": calls, "tokens": tokens}

    def periods(state: State) -> dict:
        ds = get_dataset(store, cfg)
        steps = list(state["steps"])
        have = analysis.available_periods(ds)
        if state["intent"] == "refuse":
            return {"exit": "refused", "message": REFUSED, "steps": steps}
        if state["intent"] == "out_of_data":
            return {"exit": "not_in_data", "message": NOT_IN_DATA.format(periods=", ".join(have) or "no period yet"), "steps": steps}
        r = analysis.resolve_periods(ds, state["question"], state.get("period"), state.get("compare_to"),
                                     allow_partial=state["intent"] == "baseline_check")
        steps.append({"agent": "router", "tool": "resolve_periods", "output": {k: v for k, v in r.items() if k != "note"}})
        if "exit" in r:
            return {"exit": r["exit"], "message": r["message"], "steps": steps}
        if state["intent"] == "unclear":
            return {"exit": "clarification_needed", "message": UNCLEAR.format(periods=", ".join(have)), "steps": steps}
        return {"new": r["new"], "ref": r["ref"], "steps": steps, "result": {"period_note": r["note"]}}

    def analyse(state: State) -> dict:
        ds = get_dataset(store, cfg)
        res = analysis.run(state["intent"], state["params"], ds, cfg, store, state["new"], state["ref"], state["question"])
        steps = list(state["steps"])
        if "exit" in res:
            return {"exit": res["exit"], "message": res["message"], "steps": steps}
        steps += res.pop("steps")
        res["assumptions"] += state["result"]["period_note"]
        bridge_result = res.pop("_bridge_result", None)
        if bridge_result is not None and state["intent"] in ("bill_bridge", "data_change"):
            store.save_bridge(state["run_id"], state["new"], state["ref"], bridge_result)
        return {"result": res, "steps": steps}

    def write(state: State) -> dict:
        facts, intent = state["result"]["facts"], state["intent"]
        attempts = state.get("attempts", 0) + 1
        calls, tokens = state.get("llm_calls", 0), state.get("tokens", 0)
        if llm.mock:
            return {"brief": template_brief(intent, facts), "brief_source": "template (mock LLM)", "attempts": attempts}
        try:
            text, used = llm_brief(llm, state["question"], facts, state.get("bad_numbers"), intent)
            return {"brief": text, "brief_source": f"llm ({llm.name})", "attempts": attempts,
                    "llm_calls": calls + 1, "tokens": tokens + used}
        except Exception as e:
            return {"brief": template_brief(intent, facts), "brief_source": f"template (LLM failed: {type(e).__name__})",
                    "attempts": 99, "llm_calls": calls + 1, "tokens": tokens}

    def check(state: State) -> dict:
        facts, intent = state["result"]["facts"], state["intent"]
        chk = check_numbers(state["brief"], facts)
        steps = list(state["steps"])
        brief, source = state["brief"], state["brief_source"]
        if not chk["ok"] and state["attempts"] >= 2:      # second failure: fixed template, no LLM
            brief, source = template_brief(intent, facts), "template (LLM brief failed the number check twice)"
            chk = {**check_numbers(brief, facts), "llm_unmatched": chk["unmatched"]}
        if chk["ok"] or state["attempts"] >= 2:
            steps.append({"agent": "writer", "tool": "write_finance_brief",
                          "check": "all numbers match tool outputs" if chk["ok"] else f"unmatched numbers {chk['unmatched']}",
                          "output": {"source": source, "numbers_checked": chk["numbers_checked"], "attempts": state["attempts"],
                                     **({"rejected_llm_numbers": chk["llm_unmatched"]} if "llm_unmatched" in chk else {})}})
        return {"check": chk, "brief": brief, "brief_source": source, "bad_numbers": chk["unmatched"], "steps": steps}

    def respond(state: State) -> dict:
        if state.get("exit"):
            body = {"answer_type": state["exit"], "summary": state["message"], "bridge": [], "recommendations": [],
                    "assumptions": [], "limitations": [], "period": state.get("new"), "compare_to": state.get("ref"), "data": None}
        else:
            r = state["result"]
            body = {"answer_type": r["answer_type"], "summary": state["brief"], "bridge": r["bridge"],
                    "recommendations": r["recommendations"], "assumptions": r["assumptions"], "limitations": r["limitations"],
                    "period": state["new"], "compare_to": state["ref"], "data": r["facts"]}
        return {"response": {"run_id": state["run_id"], **body}}

    g = StateGraph(State)
    for name, fn in [("parse", parse), ("periods", periods), ("analyse", analyse), ("write", write), ("check", check), ("respond", respond)]:
        g.add_node(name, timed(name, fn))
    g.set_entry_point("parse")
    g.add_edge("parse", "periods")
    g.add_conditional_edges("periods", lambda s: "respond" if s.get("exit") else "analyse", {"respond": "respond", "analyse": "analyse"})
    g.add_conditional_edges("analyse", lambda s: "respond" if s.get("exit") else "write", {"respond": "respond", "write": "write"})
    g.add_edge("write", "check")
    g.add_conditional_edges("check", lambda s: "respond" if s["check"]["ok"] or s["attempts"] >= 2 else "write",
                            {"respond": "respond", "write": "write"})
    g.add_edge("respond", END)
    return g.compile()


def run_question(graph, store, llm, question: str, period: str | None = None, compare_to: str | None = None) -> dict:
    """Runs one request through the graph and stores its trace."""
    run_id = "en-" + uuid.uuid4().hex[:6]
    t = time.perf_counter()
    state: dict[str, Any] = graph.invoke({"run_id": run_id, "question": question, "period": period,
                                          "compare_to": compare_to, "steps": [], "llm_calls": 0, "tokens": 0})
    resp = state["response"]
    store.save_trace(run_id, {
        "run_id": run_id, "question": question, "period": resp.get("period"), "compare_to": resp.get("compare_to"),
        "answer_type": resp["answer_type"], "steps": state["steps"], "llm": llm.name,
        "llm_calls": state.get("llm_calls", 0), "tokens": state.get("tokens", 0),
        "latency_ms": round((time.perf_counter() - t) * 1000, 1), "data_version": store.data_version(),
        "summary": resp["summary"]})
    return resp
