"""Builds docs/architecture_and_flow.pdf: pie charts, expected-vs-actual chart, flowcharts and the architecture.

Every number in the charts is computed by the application itself from data/history (loaded into a
temporary store), so the PDF always matches the code.

    python scripts/make_docs_pdf.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp()) / "docs.duckdb")
os.environ["MOCK_LLM"] = "true"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from app.cleaning import available_periods, build_dataset  # noqa: E402
from app.config import load_config  # noqa: E402
from app.loader import load_dir  # noqa: E402
from app.store import Store  # noqa: E402
from app.tools.assets import asset_breakdown  # noqa: E402
from app.tools.bridge import decompose  # noqa: E402
from app.tools.expected import baseline_check  # noqa: E402

OUT = ROOT / "docs" / "architecture_and_flow.pdf"
A4 = (11.69, 8.27)                       # landscape
INK, MUTED = "#1f2937", "#6b7280"
FILL = {"ui": "#dbeafe", "api": "#e0e7ff", "graph": "#ede9fe", "llm": "#fce7f3", "tool": "#dcfce7",
        "data": "#fef3c7", "store": "#ffedd5", "exit": "#fee2e2", "check": "#f1f5f9"}
CAUSE_COLOR = {"tariff": "#6366f1", "volume": "#0ea5e9", "product_mix": "#14b8a6", "weather": "#f59e0b", "waste": "#ef4444"}
LABEL = {"tariff": "Tariff (price)", "volume": "Production volume", "product_mix": "Product mix", "weather": "Weather",
         "waste": "Unexplained / waste"}


# ---- drawing helpers ------------------------------------------------------------------------------
def canvas(title: str, subtitle: str = ""):
    fig = plt.figure(figsize=A4)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")
    ax.text(3, 66.5, title, fontsize=19, weight="bold", color=INK, va="top")
    if subtitle:
        ax.text(3, 62.6, textwrap.fill(subtitle, 150), fontsize=10, color=MUTED, va="top", linespacing=1.3)
    return fig, ax


def box(ax, x, y, w, h, text, kind="tool", size=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=FILL[kind], ec="#475569", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            weight="bold" if bold else "normal", wrap=True, linespacing=1.25)
    return (x, y, w, h)


def mid(b, side):
    x, y, w, h = b
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]


def arrow(ax, p, q, label="", color="#334155", style="-|>", rad=0.0, dashed=False, lpos=0.5, loff=(0, 0.9)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, color=color, lw=1.2,
                                 connectionstyle=f"arc3,rad={rad}", linestyle="--" if dashed else "-"))
    if label:
        lx, ly = p[0] + (q[0] - p[0]) * lpos + loff[0], p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(lx, ly, label, fontsize=8, color=color, ha="center", va="center",
                bbox={"fc": "white", "ec": "none", "pad": 0.6})


def footer(ax, text):
    ax.text(3, 1.5, text, fontsize=8, color=MUTED)


# ---- data from the application --------------------------------------------------------------------
def numbers():
    store, cfg = Store(), load_config()
    load_dir(store, str(ROOT / "data" / "history"))
    ds = build_dataset(store, cfg)
    have = available_periods(ds)
    new, ref = have[-1], have[-2]
    return {"new": new, "ref": ref, "have": have, "bridge": decompose(ds, cfg, new, ref),
            "assets": asset_breakdown(ds, new), "check": baseline_check(ds, cfg, new, ref),
            "checks_by_q": {q: baseline_check(ds, cfg, q, None) for q in have[1:]}}


# ---- pages ---------------------------------------------------------------------------------------
def page_pies(pdf, n):
    b, a = n["bridge"], n["assets"]
    fig, ax = canvas(f"Where the money and the energy went: {n['ref']} to {n['new']}",
                     f"Computed by the app from data/history. Bill {b['bill']['reference']['bill']:,.0f} -> "
                     f"{b['bill']['new']['bill']:,.0f} ({b['bill']['change_pct']:+.2f}%). Baseline learned from "
                     f"{b['baseline']['fitted_on']}.")
    # pie 1: the bill change by cause (only possible as a pie when every slice has the same sign)
    rows = b["bridge"]
    left = fig.add_axes([0.04, 0.12, 0.44, 0.7])
    if all(r["pct_points"] >= 0 for r in rows) or all(r["pct_points"] <= 0 for r in rows):
        vals = [abs(r["pct_points"]) for r in rows]
        left.pie(vals, labels=[f"{LABEL[r['cause']]}\n{r['pct_points']:+.2f} pp" for r in rows],
                 colors=[CAUSE_COLOR[r["cause"]] for r in rows], startangle=90, counterclock=False,
                 autopct=lambda p: f"{p:.0f}%", pctdistance=0.72, wedgeprops={"ec": "white", "lw": 2},
                 textprops={"fontsize": 9, "color": INK})
        left.set_title(f"Bill increase of {b['bill']['change_pct']:.2f}% split into 5 causes\n(share of the increase)",
                       fontsize=11, color=INK)
    else:
        left.barh([LABEL[r["cause"]] for r in rows], [r["pct_points"] for r in rows],
                  color=[CAUSE_COLOR[r["cause"]] for r in rows])
        left.set_title("Bill change by cause (pp; mixed signs, so bars instead of a pie)", fontsize=11)
    # pie 2: energy by meter in the current quarter
    right = fig.add_axes([0.52, 0.12, 0.44, 0.7])
    items = [(x["meter"], x["kwh"]) for x in a["assets"]] + [("unmetered", a["unmetered_kwh"])]
    palette = ["#2563eb", "#f97316", "#16a34a", "#22c55e", "#4ade80", "#a855f7", "#06b6d4", "#eab308", "#94a3b8"]
    right.pie([v for _, v in items], labels=[f"{m}\n{v / 1000:,.0f} MWh" for m, v in items], colors=palette[:len(items)],
              startangle=90, counterclock=False, autopct=lambda p: f"{p:.1f}%", pctdistance=0.78,
              wedgeprops={"ec": "white", "lw": 2}, textprops={"fontsize": 8.5, "color": INK})
    right.set_title(f"{n['new']} energy by sub-meter\n(main meter {a['main_kwh'] / 1000:,.0f} MWh)", fontsize=11, color=INK)
    footer(ax, "pp = percentage points of the previous quarter's bill. The five slices add up exactly to the total change. "
               "Unmetered = main meter minus the sum of sub-meters.")
    pdf.savefig(fig)
    plt.close(fig)


def page_expected(pdf, n):
    c = n["check"]
    fig, ax = canvas("The baseline: expected energy vs actual energy",
                     f"Linear regression of daily kWh on working days, units per product, cooling and heating degree-days. "
                     f"For {n['new']} it learned from {c['baseline']['fitted_on']} ({c['baseline']['days_used']} days, "
                     f"R² {c['baseline']['r2']}, out-of-sample CV(RMSE) {c['baseline']['cv_rmse_pct']}%).")
    left = fig.add_axes([0.06, 0.13, 0.42, 0.62])
    qs = list(n["checks_by_q"])
    act = [n["checks_by_q"][q]["new"]["actual_kwh"] / 1000 for q in qs]
    exp = [n["checks_by_q"][q]["new"]["expected_kwh"] / 1000 for q in qs]
    x = range(len(qs))
    left.bar([i - 0.2 for i in x], exp, 0.4, color="#94a3b8", label="Expected (baseline from the quarters before)")
    left.bar([i + 0.2 for i in x], act, 0.4, color="#2563eb", label="Actual (main meter)")
    for i, q in enumerate(qs):
        g = n["checks_by_q"][q]["new"]["gap_pct"]
        left.text(i, max(act[i], exp[i]) + 25, f"{g:+.1f}%", ha="center", fontsize=9, color=INK)
    left.set_xticks(list(x), qs)
    left.set_ylabel("MWh per quarter")
    left.set_title("Each quarter against its own baseline\n(learned only from earlier quarters)", fontsize=11)
    left.set_ylim(0, max(act + exp) * 1.28)
    left.legend(fontsize=8, loc="upper left", frameon=False)
    left.spines[["top", "right"]].set_visible(False)

    right = fig.add_axes([0.56, 0.13, 0.40, 0.62])
    m = c["months"]
    xs = range(len(m))
    right.bar([i - 0.2 for i in xs], [r["expected_kwh"] / 1000 for r in m], 0.4, color="#94a3b8", label="Expected")
    right.bar([i + 0.2 for i in xs], [r["actual_kwh"] / 1000 for r in m], 0.4, color="#2563eb", label="Actual")
    for i, r in enumerate(m):
        right.text(i, max(r["expected_kwh"], r["actual_kwh"]) / 1000 + 8, f"{r['gap_pct']:+.1f}%", ha="center", fontsize=9)
    right.set_xticks(list(xs), [r["month"] for r in m])
    right.set_ylabel("MWh per month")
    right.set_title(f"{n['new']} by month (verdict: {c['verdict']} the expected range)", fontsize=11)
    right.set_ylim(0, max(max(r["expected_kwh"], r["actual_kwh"]) for r in m) / 1000 * 1.25)
    right.legend(fontsize=8, loc="upper left", frameon=False)
    right.spines[["top", "right"]].set_visible(False)
    footer(ax, "Waste slice = (actual - expected) in the current quarter minus (actual - expected) in the previous quarter. "
               "2025-Q3 is not shown: nothing earlier exists to learn its baseline from.")
    pdf.savefig(fig)
    plt.close(fig)


def page_question_flow(pdf):
    fig, ax = canvas("Flowchart 1: what happens when a user asks a question",
                     "POST /analyze runs one LangGraph flow with 6 nodes. The LLM is used at most for labelling an unclear "
                     "question and for wording the brief; code makes every number.")
    q = box(ax, 3, 47, 13, 8, "User question\n(UI or POST /analyze)", "ui", bold=True)
    parse = box(ax, 21, 47, 15, 8, "1. parse\nguard rules -> label\n(LLM only if rules unsure)", "graph")
    per = box(ax, 41, 47, 15, 8, "2. periods\ncurrent quarter vs\nlatest past quarter", "graph")
    ana = box(ax, 61, 47, 15, 8, "3. analyse\nrun the code tools\nfor this label", "graph")
    write = box(ax, 81, 47, 15, 8, "4. write\nbrief: LLM rewrites\nthe template draft", "llm")
    chk = box(ax, 81, 30, 15, 9, "5. check\nevery number in the\nbrief must match a fact", "check")
    resp = box(ax, 61, 30, 15, 9, "6. respond\nJSON answer + trace\n(run_id)", "api", bold=True)
    tmpl = box(ax, 81, 14, 15, 8, "fixed template\n(numbers always right)", "check")
    exits = box(ax, 33, 28, 22, 11, "Early exits, no tools, no LLM:\nrefused (guard rule)\nnot_in_data (forecast, gas, missing\n"
                "or incomplete quarter)\nclarification_needed (unknown meter)", "exit", size=8.5)
    arrow(ax, mid(q, "r"), mid(parse, "l"))
    arrow(ax, mid(parse, "r"), mid(per, "l"))
    arrow(ax, mid(per, "r"), mid(ana, "l"))
    arrow(ax, mid(ana, "r"), mid(write, "l"))
    arrow(ax, mid(write, "b"), mid(chk, "t"))
    arrow(ax, mid(chk, "l"), mid(resp, "r"), "pass")
    arrow(ax, (96.5, 34.5), (96.5, 51), "fail once:\nrewrite", rad=0.0, color="#b91c1c", lpos=0.5, loff=(-0.2, 0))
    arrow(ax, mid(chk, "b"), mid(tmpl, "t"), "fail twice", color="#b91c1c")
    arrow(ax, mid(tmpl, "l"), (68.5, 30), color="#475569", rad=-0.2)
    arrow(ax, mid(parse, "b"), (40, 39), "refuse /\nout of data", color="#b91c1c", dashed=True)
    arrow(ax, mid(per, "b"), (48, 39), "quarter missing", color="#b91c1c", dashed=True)
    arrow(ax, mid(exits, "r"), mid(resp, "l"), color="#b91c1c", dashed=True)

    tools = [("bill_bridge", "bill split into 5 causes + expected vs actual + meter evidence + actions"),
             ("baseline_check", "expected vs actual energy, by month (also for a partial quarter)"),
             ("top_users / idle_waste", "energy by sub-meter / energy while lines are stopped"),
             ("savings / ranking", "actions with kWh, money, CO2e and payback ranges"),
             ("what_if", "bill repriced at another tariff, energy kept as measured"),
             ("bill_total / data_change", "bill of a quarter / split before and after a correction")]
    ax.text(3, 21.5, "Labels and the tools they run (node 3)", fontsize=11, weight="bold", color=INK)
    for i, (k, v) in enumerate(tools):
        ax.text(3, 18.4 - i * 2.6, f"{k}", fontsize=9, weight="bold", color="#166534")
        ax.text(19, 18.4 - i * 2.6, v, fontsize=9, color=INK)
    footer(ax, "Every step (tool, inputs, outputs, timings, LLM calls, tokens) is saved in the trace: GET /trace/{run_id}.")
    pdf.savefig(fig)
    plt.close(fig)


def page_data_flow(pdf):
    fig, ax = canvas("Flowchart 2: from CSV files to answers (history + new quarter)",
                     "The history is loaded once; each new quarter is uploaded on top. A quarter is compared only when "
                     "meter, production, weather and tariff cover at least 80% of its days.")
    hist = box(ax, 3, 46, 16, 9, "data/history\n5 quarters\n2025-Q3 .. 2026-Q3", "data", bold=True)
    newq = box(ax, 3, 31, 16, 9, "New quarter CSVs\n(UI upload, POST /ingest,\nload_data.py)", "ui", bold=True)
    load = box(ax, 24, 38, 15, 10, "Loader\nschema check, bad rows\nrejected, duplicates\ncollapsed, upsert", "tool")
    db = box(ax, 44, 38, 13, 10, "DuckDB\n4 tables + traces\n+ saved results", "store", bold=True)
    clean = box(ax, 62, 38, 16, 10, "Cleaning\n15-min grid, resets, gaps,\ntariff blocks -> prices,\ndaily table", "tool")
    status = box(ax, 83, 46, 14, 9, "Quarter status\ncomplete / partial /\nmissing what", "check")
    valid = box(ax, 83, 31, 14, 9, "Validation\n8 checks\n(GET /validate)", "check")
    window = box(ax, 52, 16, 20, 11, "History window\nup to 4 complete quarters\nbefore the current one\n(never the current quarter)", "graph")
    base = box(ax, 77, 16, 20, 11, "Baseline (linear regression)\ndaily kWh = base + working day\n+ rate x units per product\n"
               "+ k x CDD + k x HDD", "tool")
    outs = box(ax, 24, 14, 24, 13, "Used by the tools\nexpected vs actual energy\nbill split (volume, mix, weather, waste)\n"
               "waste check per meter vs its\nweather-adjusted history", "tool")
    arrow(ax, mid(hist, "r"), (24, 45))
    arrow(ax, mid(newq, "r"), (24, 41))
    arrow(ax, mid(load, "r"), mid(db, "l"))
    arrow(ax, mid(db, "r"), mid(clean, "l"))
    arrow(ax, (78, 46), mid(status, "l"))
    arrow(ax, (78, 40), mid(valid, "l"))
    arrow(ax, mid(status, "b"), (62, 27), "complete quarters", rad=0.15)
    arrow(ax, mid(window, "r"), mid(base, "l"))
    arrow(ax, mid(base, "t"), mid(outs, "t"), "fitted rates", rad=0.3, lpos=0.5, loff=(0, 0))
    rows = [("Loaded", "Current", "Compared with", "Baseline learns from"),
            ("history", "2026-Q3", "2026-Q2", "2025-Q3, 2025-Q4, 2026-Q1, 2026-Q2"),
            ("history + demo 2026-Q4", "2026-Q4", "2026-Q3", "2025-Q4, 2026-Q1, 2026-Q2, 2026-Q3"),
            ("history + 20 days of Q4", "2026-Q3 (Q4 partial)", "2026-Q2", "same as row 1; Q4 gets expected vs actual only")]
    for i, r in enumerate(rows):
        for j, (cell, x) in enumerate(zip(r, (3, 23, 38, 52))):
            ax.text(x, 11.2 - i * 2.2, cell, fontsize=8.5, color=INK, weight="bold" if i == 0 else "normal")
    footer(ax, "Carried-forward prices never count as tariff coverage, so a forgotten tariff file keeps the quarter incomplete.")
    pdf.savefig(fig)
    plt.close(fig)


def page_architecture(pdf):
    fig, ax = canvas("System architecture",
                     "One FastAPI service, one LangGraph flow, deterministic analytics in app/tools/, one DuckDB file. "
                     "Docker Compose runs the API and the UI; start.bat does the same on Windows without Docker.")
    # layers
    ax.add_patch(FancyBboxPatch((2, 50), 96, 9, boxstyle="round,pad=0.2", fc="#f8fafc", ec="#cbd5e1"))
    ax.text(3, 57.6, "Presentation", fontsize=9, color=MUTED)
    ui = box(ax, 18, 51.5, 26, 6, "Streamlit UI (ui/streamlit_app.py)\nupload quarter, ask, charts, trace", "ui")
    cli = box(ax, 50, 51.5, 22, 6, "CLI scripts\nload_data, validate_data,\nmake_demo_quarter, run_eval", "ui", size=8.5)
    curl = box(ax, 77, 51.5, 18, 6, "Any HTTP client\n(curl, other apps)", "ui")

    ax.add_patch(FancyBboxPatch((2, 38.5), 96, 9.5, boxstyle="round,pad=0.2", fc="#f8fafc", ec="#cbd5e1"))
    ax.text(3, 46.6, "API (app/api.py)", fontsize=9, color=MUTED)
    eps = ["POST /analyze", "POST /ingest", "GET /periods", "GET /validate", "GET /trace/{id}", "GET /health"]
    epb = [box(ax, 4 + i * 15.6, 40, 14, 5, e, "api", size=9) for i, e in enumerate(eps)]

    ax.add_patch(FancyBboxPatch((2, 22), 63, 14, boxstyle="round,pad=0.2", fc="#f8fafc", ec="#cbd5e1"))
    ax.text(3, 34.6, "Agent layer (app/graph.py, intents.py, writer.py, numcheck.py)", fontsize=9, color=MUTED)
    nodes = ["parse", "periods", "analyse", "write", "check", "respond"]
    nb = [box(ax, 4 + i * 10.2, 27, 8.6, 5, n, "graph" if n != "write" else "llm", size=9) for i, n in enumerate(nodes)]
    for a_, b_ in zip(nb, nb[1:]):
        arrow(ax, mid(a_, "r"), mid(b_, "l"))
    ax.text(4, 23.3, "guard rules first  |  facts only from tools  |  number check, 1 retry, then template", fontsize=8.5, color=INK)

    llm = box(ax, 70, 24, 27, 11, "LLM adapter (app/llm.py)\nGemini gemini-3.5-flash-lite (current .env)\n"
              "also: Ollama, OpenAI, Anthropic\nMOCK_LLM=true -> rules + template", "llm", size=8.5)
    arrow(ax, mid(nb[3], "t"), (70, 33), "", color="#9d174d", rad=-0.25)
    ax.text(83.5, 22.3, "words the brief; labels a question\nonly when the rules are unsure", fontsize=7.5,
            color="#9d174d", ha="center", va="top")

    ax.add_patch(FancyBboxPatch((2, 6.5), 63, 13.5, boxstyle="round,pad=0.2", fc="#f8fafc", ec="#cbd5e1"))
    ax.text(3, 18.6, "Analytics (app/tools/ and app/*.py) - no LLM here", fontsize=9, color=MUTED)
    tl = ["loader", "cleaning", "validation", "billing", "baseline", "expected", "bridge", "waste", "assets", "savings"]
    for i, t in enumerate(tl):
        box(ax, 4 + (i % 5) * 12.2, 13 - (i // 5) * 5.3, 10.6, 4, t, "tool", size=9)

    store = box(ax, 70, 7, 27, 12, "DuckDB file (app/store.py)\nmeter_15min, tariff, production,\nweather, traces, saved bridges,\n"
                "metadata (data version, load stats)", "store", size=8.5)
    for e in epb[:2]:
        arrow(ax, mid(e, "b"), (mid(e, "b")[0], 36), color="#64748b")
    arrow(ax, mid(nb[2], "b"), (30, 20), "runs tools", color="#166534", lpos=0.85, loff=(4, 0))
    arrow(ax, (65, 13), mid(store, "l"), "read / write", color="#9a3412", style="<|-|>")
    arrow(ax, mid(ui, "b"), (25, 45), color="#64748b")
    arrow(ax, mid(curl, "b"), (86, 45), color="#64748b")
    footer(ax, "config.yaml holds every assumption (history_quarters 4, tolerance 6%, emission factor 0.4, action costs). "
               ".env holds the LLM provider, model and key (never committed).")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> int:
    n = numbers()
    OUT.parent.mkdir(exist_ok=True)
    with PdfPages(OUT) as pdf:
        page_pies(pdf, n)
        page_expected(pdf, n)
        page_question_flow(pdf)
        page_data_flow(pdf)
        page_architecture(pdf)
        d = pdf.infodict()
        d["Title"] = "Factory Energy Agent: charts, flowcharts and architecture"
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
