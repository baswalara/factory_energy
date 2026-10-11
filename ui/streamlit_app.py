"""UI: upload a new quarter, see which quarters are loaded, ask questions, see the bill bridge,
expected-vs-actual energy, actions and the trace.
Talks to the API over HTTP only, so the UI shows exactly what a curl call would get."""
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TABLES = ["meter_15min", "tariff", "production", "weather"]
LABEL = {"tariff": "Tariff (price)", "volume": "Volume", "product_mix": "Product mix", "weather": "Weather", "waste": "Unexplained / waste"}
EXAMPLES = ["Why did the bill go up?", "Is energy above the baseline?", "Which assets use the most energy?",
            "How much energy is used when nothing is produced?", "What should we do first?", "Which action pays back fastest?",
            "What if the tariff had not changed?", "What changed after the correction?"]

st.set_page_config(page_title="Factory energy bill", layout="wide")
st.title("Factory energy: why the bill changed, and what to fix")
st.caption("The current quarter is compared with the latest past quarter. Expected energy comes from a linear baseline "
           "learned from up to 4 quarters of history before the current one.")


def call(method: str, path: str, **kw):
    try:
        r = requests.request(method, API + path, timeout=300, **kw)
        return r.json()
    except Exception as e:
        st.error(f"API not reachable at {API}: {e}")
        return None


def table_of(name: str) -> str | None:
    stem = name.rsplit("/", 1)[-1].lower()
    return next((t for t in TABLES if stem.startswith(t)), None)


def bridge_chart(bridge: list, total: float, period: str, ref: str):
    rows = sorted(bridge, key=lambda r: (r["cause"] == "waste", -r["pct_points"]))
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["relative"] * len(rows) + ["total"],
        x=[LABEL[r["cause"]] for r in rows] + ["Total change"],
        y=[r["pct_points"] for r in rows] + [total],
        text=[f"{r['pct_points']:+.2f}<br>({r['range'][0]:.1f} to {r['range'][1]:.1f})" for r in rows] + [f"{total:+.2f}%"],
        textposition="outside", connector={"line": {"color": "#9aa0a6", "width": 1}},
        increasing={"marker": {"color": "#c2410c"}}, decreasing={"marker": {"color": "#15803d"}}, totals={"marker": {"color": "#334155"}}))
    fig.update_layout(title=f"Bill change {ref} to {period}: {total:+.2f}% (percentage points by cause, range in brackets)",
                      yaxis_title="Percentage points of the reference bill", showlegend=False, height=430, margin=dict(t=60, b=20))
    return fig


def expected_chart(labels: list, actual: list, expected: list, title: str):
    fig = go.Figure([go.Bar(name="Expected (baseline)", x=labels, y=expected, marker_color="#94a3b8"),
                     go.Bar(name="Actual (main meter)", x=labels, y=actual, marker_color="#2563eb")])
    fig.update_layout(barmode="group", title=title, yaxis_title="kWh", height=360, margin=dict(t=60, b=20))
    fig.update_xaxes(type="category")                        # '2026-10' and '2026-Q3' are labels, not dates
    return fig


periods = call("GET", "/periods") or {}
with st.sidebar:
    st.subheader("System")
    h = call("GET", "/health")
    if h:
        st.write(f"API **{h['api']}** · data **{h['data_store'].get('status')}**")
        st.caption(f"LLM: {h['llm']}")
    st.subheader("Quarters")
    if periods:
        st.write(periods.get("summary", ""))
        cur = periods.get("current")
        if cur:
            q = periods["quarters"][cur]
            st.caption(f"Current quarter **{cur}**, compared with **{q.get('compared_with') or 'none'}**. "
                       f"Baseline history: {', '.join(q.get('baseline_history') or []) or 'none'}.")
    st.subheader("Upload a new quarter")
    st.caption("Drop the quarter's CSV files (meter_15min*, tariff*, production*, weather*). The table is read from the "
               "file name. All four are needed before the quarter can be compared.")
    ups = st.file_uploader("Quarter files", type="csv", accept_multiple_files=True, key="quarter")
    if ups and st.button("Upload quarter", type="primary"):
        order = sorted(ups, key=lambda f: TABLES.index(table_of(f.name)) if table_of(f.name) else 9)
        for f in order:
            t = table_of(f.name)
            if not t:
                st.error(f"{f.name}: name does not start with one of {TABLES}; use the single-file form below")
                continue
            res = call("POST", "/ingest", files={"file": (f.name, f.getvalue())}, data={"table": t})
            if res:
                st.write(f"{f.name} → {t}: {res['status']}, {res['rows_loaded']} loaded, {res['rows_rejected']} rejected")
        st.session_state.pop("answer", None)
        st.rerun()
    with st.expander("Correct one table"):
        table = st.selectbox("Table", TABLES)
        up = st.file_uploader("CSV in the Annex B schema", type="csv", key="single")
        if up and st.button("Ingest"):
            res = call("POST", "/ingest", files={"file": (up.name, up.getvalue())}, data={"table": table})
            if res:
                st.write(f"{res['status']}: {res['rows_loaded']} loaded, {res['rows_rejected']} rejected")
                if res.get("rerun"):
                    st.caption("Bill split before and after this file")
                    st.dataframe(pd.DataFrame(res["rerun"]["bridge"]), hide_index=True)
    st.subheader("Data validation")
    v = call("GET", "/validate")
    if v:
        st.write(f"Overall: **{v['status']}**")
        for c in v["checks"]:
            with st.expander(f"{c['status']} · {c['name']}"):
                st.write(c["detail"])

complete, partial = periods.get("complete", []), periods.get("partial", [])
c1, c2 = st.columns(2)
period = c1.selectbox("Current quarter", ["auto (latest)"] + sorted(complete + partial, reverse=True))
compare = c2.selectbox("Compare with", ["auto (quarter before)"] + sorted(complete, reverse=True))
question = st.text_input("Ask a question", value=st.session_state.get("q", EXAMPLES[0]))
cols = st.columns(4)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 4].button(ex, key=f"ex{i}", width="stretch"):
        question = ex
        st.session_state["q"] = ex
        st.session_state["go"] = True

if st.button("Ask", type="primary") or st.session_state.pop("go", False):
    body = {"question": question}
    if not period.startswith("auto"):
        body["period"] = period
    if not compare.startswith("auto"):
        body["compare_to"] = compare
    st.session_state["answer"] = call("POST", "/analyze", json=body)

d = st.session_state.get("answer")
if d:
    st.caption(f"answer_type: {d['answer_type']} · run_id: {d['run_id']} · {d.get('compare_to') or ''} to {d.get('period') or ''}")
    (st.warning if d["answer_type"] in ("not_in_data", "clarification_needed", "refused") else st.info)(d["summary"])
    data = d.get("data") or {}
    if d["bridge"]:
        total = round(sum(r["pct_points"] for r in d["bridge"]), 2)
        st.plotly_chart(bridge_chart(d["bridge"], total, d["period"], d["compare_to"]), width="stretch")
    ev = data.get("expected_vs_actual")
    if ev:
        st.plotly_chart(expected_chart([d["compare_to"], d["period"]], [ev["reference"]["actual_kwh"], ev["new"]["actual_kwh"]],
                                       [ev["reference"]["expected_kwh"], ev["new"]["expected_kwh"]],
                                       f"Expected vs actual energy (baseline learned from {', '.join(ev['baseline_quarters'])})"),
                        width="stretch")
    if data.get("months"):
        m = data["months"]
        st.plotly_chart(expected_chart([x["month"] for x in m], [x["actual_kwh"] for x in m], [x["expected_kwh"] for x in m],
                                       f"{data['period']} by month: expected vs actual"), width="stretch")
        st.dataframe(pd.DataFrame(m), hide_index=True)
    if d["recommendations"]:
        st.subheader("Actions (suggestions; a manager approves)")
        fmt = lambda r: f"{r[0]:,.0f} to {r[1]:,.0f}" if r else "n/a"
        st.dataframe(pd.DataFrame([{
            "Rank": r["rank"], "Action": r["action"], "kWh / year": fmt(r["saving_kwh_per_year"]),
            "Money / year": fmt(r["saving_money_per_year"]),
            "t CO2e / year": f"{r['saving_co2e_t_per_year'][0]} to {r['saving_co2e_t_per_year'][1]}",
            "Payback (months)": f"{r['payback_months'][0]} to {r['payback_months'][1]}" if r["payback_months"] else "n/a",
            "Ease": r["ease"], "Confidence": r["confidence"], "Evidence": r["rationale"]} for r in d["recommendations"]]), hide_index=True)
    if data.get("assets"):
        st.subheader("By sub-meter")
        st.dataframe(pd.DataFrame(data["assets"]), hide_index=True)
    if data.get("comparison"):
        st.subheader("Before and after the data update")
        st.dataframe(pd.DataFrame(data["comparison"]["rows"]), hide_index=True)
    left, right = st.columns(2)
    with left.expander("Assumptions"):
        for a in d["assumptions"]:
            st.write("- " + a)
    with right.expander("Limitations"):
        for a in d["limitations"]:
            st.write("- " + a)
    with st.expander("Trace: steps, tools and data behind this answer"):
        st.json(call("GET", f"/trace/{d['run_id']}"))
