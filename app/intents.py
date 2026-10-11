"""Question -> one label from a fixed list. Rules always run first as a guard, so a request
to invent numbers or go beyond the data is stopped in code before any LLM sees it."""
from __future__ import annotations

import json
import re

INTENTS = ["bill_bridge", "bill_total", "baseline_check", "top_users", "idle_waste", "savings", "ranking",
           "what_if", "data_change", "out_of_data", "refuse", "unclear"]

REFUSE = re.compile(
    r"ignore (all |the |your |any |previous |prior |those )*(instructions|rules|tools|data|numbers|checks)|disregard"
    r"|make up|made[- ]up|invent|fabricat|just guess|guess (a|the|any) |pretend|round it up to"
    r"|without (using |running |calling |checking )?(the |any )?(tools|data|calculations?)"
    r"|do ?n[o']t (use|run|call) (the |any )?(tools|data)|system prompt|jailbreak"
    r"|approve (it|this|them|the action)|auto-?approve|mark (it|this) as approved"
    r"|(say|tell|report|write)( to)?( me| them| us| finance| the manager)?( that)? the (bill|saving|savings|waste|increase) "
    r"(is|was|are|went|rose|will be)\b.*\d"
    r"|assume (the|that) .*\b(is|was|are|costs?|saves?) [\d$]|put (that|this|it) in the brief"
    r"|\b(say|write|state|claim|report|put down|tell (me|them|us|finance|the manager))( to \w+)?( in the brief)? that\b.*\d"
    r"|\bskip (the )?(weather|tariff|price|volume|production|mix|baseline)|ignore (the )?(weather|tariff|price|volume|mix|baseline)"
    r"|\bjust blame\b|blame (it on )?the \w+ (anyway|regardless|instead)"
    r"|\b(poem|joke|password|write (me )?(some )?code)\b", re.I)
OUT_OF_DATA = re.compile(
    r"next (year|quarter|month|week)|forecast|predict(?!ed (energy|consumption|kwh|use|usage|load))|projection|in the future|will (the|it|we|our|electricity|energy|prices?)\b"
    r"|\b(gas|water|steam|diesel|solar|headcount|employees?|salary|salaries|supplier|vendor|competitor|revenue|profit|"
    r"carbon tax|subsidy|maintenance (log|record)s?|machine age|how old)\b", re.I)
VAGUE_ASSET = re.compile(r"\b(this|that|the) (asset|machine|meter|equipment)\b", re.I)

RULES = [
    ("data_change", r"what (has |had )?changed|before and after|after the (update|correction|late|new data|data drop|fix)"
                    r"|corrected (tariff|data|file)|calibration|since the (update|correction)"),
    ("what_if", r"what if|what would|suppose|had not changed|hadn'?t changed|had stayed|if the (tariff|price)|if (tariff|price)s?"),
    ("ranking", r"pay(s)? ?back|fastest|quickest|rank|easiest|priorit"),
    ("savings", r"what should we (do|fix)|should we do first|recommend|action|how (much )?(can|could) we save|saving|co2|carbon|"
                r"emission|fix first|reduce (the )?(bill|cost|energy)|cut (the )?(bill|cost)"),
    ("baseline_check", r"\bexpect(ed|ation)?\b|baseline|should (have )?(used|consumed|be using|use)|normal (level|use|usage|consumption)"
                       r"|(above|below|over|under|versus|vs\.?|against|compared (to|with)) (the )?(normal|expected|baseline|predicted|model)"
                       r"|actual (vs\.?|versus|against|compared)|predicted (energy|consumption|kwh)|over-?consum|more than (normal|usual)"),
    ("bill_total", r"(what|how much) (was|is|were) (the|our) (total )?(bill|cost|spend)|total (bill|cost|energy|kwh|consumption)"
                   r"|how much (energy|electricity|kwh) (did|was)|average price|how many kwh"),
    ("bill_bridge", r"\bwhy\b|bill|increase|went up|go up|gone up|rise|rose|higher|decompos|bridge|split|break ?down the (bill|increase)"
                    r"|cause|blame|reason|responsible|explain|driver|due to|because|15 ?%|percent|\bmix\b|product [a-z0-9]\b|heavier"
                    r"|\bbrief\b|summary|summari[sz]e|report for|write[- ]?up"),
    ("top_users", r"which (asset|meter|machine|area)s?|most energy|biggest|largest|top (user|consumer|asset)|sub-?meter|by asset"
                  r"|uses? the most|consum|share of|break ?down|how much (energy )?does"),
    ("idle_waste", r"idle|nothing is (being )?produced|no production|not producing|night|weekend|holiday|standby|wast|leak"
                   r"|when (the )?(lines?|factory|plant) (is|are) (stopped|off|down)"),
]


def find_meter(question: str, meters: list[str]) -> str | None:
    q = question.lower().replace("-", "_")
    for m in sorted(meters, key=len, reverse=True):
        names = {m, m.replace("_", " "), m.rstrip("s")} if len(m) > 4 else {m}
        if any(re.search(rf"\b{re.escape(n)}", q) or re.search(rf"\b{re.escape(n)}", q.replace("_", " ")) for n in names):
            return m
    return None


def tariff_scenario(question: str) -> dict:
    q = question.lower()
    if re.search(r"had ?n[o']t changed|had stayed|unchanged|no (tariff|price) (change|increase|rise)|without the (tariff|price) (change|increase|rise)"
                 r"|stayed the same|kept the old|at the old (tariff|price)", q):
        return {"tariff_unchanged": True}
    m = re.search(r"(\d+(?:\.\d+)?)\s*(%|percent|per cent)", q)
    if m and re.search(r"tariff|price|rate", q):
        pct = float(m.group(1))
        down = re.search(r"lower|less|cheaper|cut|reduc|drop|decreas|fall|fell|down|discount", q)
        return {"tariff_change_pct": -pct if down else pct}
    return {}


def parse_rules(question: str) -> dict:
    q = question.strip()
    if not q:
        return {"intent": "unclear"}
    if REFUSE.search(q):
        return {"intent": "refuse"}
    if OUT_OF_DATA.search(q):
        return {"intent": "out_of_data"}
    for intent, pattern in RULES:
        if re.search(pattern, q, re.I):
            if intent == "what_if":
                return {"intent": intent, **tariff_scenario(q)}
            return {"intent": intent}
    return {"intent": "unclear"}


SYSTEM = """You label questions about a factory's electricity bill. Reply with JSON only:
{"intent": "<label>", "tariff_change_pct": <number or null>, "tariff_unchanged": <true or false>}
Labels:
- bill_bridge: why the bill changed, what caused it, who or what is to blame; also a request to write a brief, summary or report for a manager
- bill_total: the bill, energy or average price of a period
- baseline_check: expected (baseline) energy versus actual energy; is use above or below normal
- top_users: which assets or meters use the most energy
- idle_waste: energy used when nothing is produced, idle load, leaks
- savings: what to do, how much can be saved, CO2
- ranking: which action pays back fastest or is easiest
- what_if: a what-if about the tariff or price (set tariff_change_pct, negative for lower; or tariff_unchanged)
- data_change: what changed after a data correction or update
- out_of_data: forecasts, future prices, or anything not in meter, tariff, production or weather data
- refuse: ONLY when it asks to invent, guess or assume a number, skip the tools, approve an action, or do an unrelated task
- unclear: too vague to tell
Never answer the question. Only label it."""


def parse_llm(question: str, llm) -> tuple[dict | None, int]:
    """LLM label, validated against the fixed list. None when the output is unusable."""
    try:
        text, tokens = llm.complete(SYSTEM, question, want_json=True)
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
        if obj.get("intent") not in INTENTS:
            return None, tokens
        out = {"intent": obj["intent"]}
        if obj["intent"] == "what_if":
            if obj.get("tariff_unchanged") is True:
                out["tariff_unchanged"] = True
            elif isinstance(obj.get("tariff_change_pct"), (int, float)) and abs(obj["tariff_change_pct"]) <= 90:
                out["tariff_change_pct"] = float(obj["tariff_change_pct"])
        return out, tokens
    except Exception:
        return None, 0
