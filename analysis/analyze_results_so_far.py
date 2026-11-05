#!/usr/bin/env python3
"""
Analysis of every COMPLETE cell (2/2 queries stored) under claude_code_book_strategies/.
Inputs : <cell>/<cell>.json records, the raw *_events.jsonl of each call, df_books_modified.csv (ground truth).
Outputs: analysis/results_so_far/{per_call.csv, per_cell.csv, *.png}
Nothing is estimated or imputed: a value missing in the stored files is left blank (NaN).
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "claude_code_book_strategies"
GT_FILE = REPO / "df_books_modified.csv"
OUT = Path(__file__).resolve().parent / "results_so_far"
OUT.mkdir(parents=True, exist_ok=True)

TIER_ORDER = ["no_evidence", "year_list_only", "titles_authors", "titles_authors_prize",
              "full_with_birthplace", "full_with_population", "full_with_distractors"]
TIER_LABEL = {"no_evidence": "RAG0 none", "year_list_only": "RAG1 titles", "titles_authors": "RAG2 +authors",
              "titles_authors_prize": "RAG3 +prize", "full_with_birthplace": "RAG4 +birthplace",
              "full_with_population": "RAG5 +population", "full_with_distractors": "RAG6 +distractors"}
# categorical slots 1-4 of the reference palette (light mode), fixed order, one per configuration
CONFIGS = [("off", "none", 1, "Thinking off · no tools", "#2a78d6"),
           ("low", "none", 1, "Low effort · no tools", "#eb6834"),
           ("off", "web", 10, "Thinking off · web, 10 turns", "#1baf7a"),
           ("off", "web", 30, "Thinking off · web, 30 turns", "#eda100"),
           ("low", "web", 30, "Low effort · web, 30 turns", "#e87ba4")]
CFG_LABEL = {(t, m, n): lab for t, m, n, lab, _ in CONFIGS}
CFG_COLOR = {lab: c for *_, lab, c in CONFIGS}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def norm(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s).split()


def name_key(s):
    """Tolerant author key: first + last name token, initials / middle names / suffixes dropped
    ('James A. Michener' == 'James Michener', 'A. B. Guthrie Jr.' -> 'guthrie')."""
    t = [w for w in norm(s) if len(w) > 1 and w not in ("jr", "sr", "ii", "iii")]
    return " ".join([t[0], t[-1]]) if len(t) >= 2 else " ".join(t)


def is_sentinel(s):
    k = " ".join(norm(s))
    return k in ("", "unknown", "none", "n a") or k.startswith(("no result", "no matching", "error", "none "))


def place_match(pred, gt):
    """True when the city token sequence of the GT birthplace (text before the first comma) is in the prediction."""
    if not gt or not pred:
        return np.nan
    city = " ".join(norm(str(gt).split(",")[0]))
    return float(city != "" and city in " ".join(norm(pred)))


def pop_rel_err(pred, gt):
    try:
        p, g = float(str(pred).replace(",", "")), float(gt)
    except (TypeError, ValueError):
        return np.nan
    return abs(p - g) / g if g else np.nan


def last_result_event(raw_files):
    """num_turns / subtype / duration / per-model usage from the events file of the call that produced the record."""
    out = {"num_turns": np.nan, "result_subtype": None, "stop_reason": None, "subagent_web_searches": np.nan,
           "models_used": None}
    ev = next((f for f in (raw_files or []) if f.endswith("_events.jsonl")), None)
    if not ev or not (REPO / ev).exists():
        return out
    res = None
    for line in open(REPO / ev, encoding="utf-8", errors="replace"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") == "result":
            res = e
    if res:
        mu = res.get("modelUsage") or res.get("model_usage_since_last_result") or {}
        out.update(num_turns=res.get("num_turns"), result_subtype=res.get("subtype"), stop_reason=res.get("stop_reason"),
                   subagent_web_searches=sum(int(v.get("webSearchRequests") or 0) for v in mu.values()) if mu else np.nan,
                   models_used=";".join(sorted(mu)) if mu else None)
    return out


def count_attempts(cell_dir, qid):
    return len(list((cell_dir / "raw_calls").glob(f"*_q{qid}_try*_events.jsonl")))


# ---------------------------------------------------------------- ground truth
gt = pd.read_csv(GT_FILE)
gt["S.No"] = gt["S.No"].astype(str)
GT = {}
for sno, g in gt.groupby("S.No"):
    rows = g[g["Author"].astype(str).str.upper() != "NO RESULT"]
    GT[sno] = {name_key(r.Author): (r.Author, r.Birthplace if pd.notna(r.Birthplace) else None,
                                    r.PopCount if pd.notna(r.PopCount) else None) for r in rows.itertuples()}

# ---------------------------------------------------------------- records
calls, rows_out = [], []
for jf in sorted(ROOT.glob("*/*.json")):
    cell = jf.parent.name
    try:
        data = json.loads(jf.read_text(encoding="utf-8"))
    except ValueError:
        continue
    recs = data.get("results", [])
    if len(recs) < 2:                       # incomplete / in-progress cell -> excluded
        continue
    for r in recs:
        # configuration from the cell folder name (an error record carries no config fields)
        th = "off" if "_THOFF" in cell else ("low" if "_EFFLOW" in cell else "default")
        tm = "web" if "_WEB" in cell else "none"
        mt = 30 if "_WEB_T30" in cell else (10 if tm == "web" else 1)
        cfg = CFG_LABEL.get((th, tm, mt), f"{th}/{tm}/{mt}")
        qid = str(r["Query_ID"])
        gtn = GT.get(qid, {})
        is_err = str(r.get("Generated_Response", "")).startswith("ERROR")
        parsed = [p for p in (r.get("Parsed_Rows") or [])
                  if p.get("Author") and not is_sentinel(p["Author"])]
        pnames = {name_key(p["Author"]) for p in parsed}
        correct = pnames & set(gtn)
        bp_hits, pop_errs = [], []
        for p in parsed:
            k = name_key(p["Author"])
            if k in gtn:
                bp_hits.append(place_match(p.get("Birthplace"), gtn[k][1]))
                pop_errs.append(pop_rel_err(p.get("PopCount"), gtn[k][2]))
            rows_out.append({"cell": cell, "config": cfg, "rag_tier": r.get("rag_tier"), "Query_ID": qid,
                             "Author": p["Author"], "Birthplace": p.get("Birthplace"), "PopCount": p.get("PopCount"),
                             "in_GT": k in gtn, "GT_Birthplace": gtn.get(k, (None, None, None))[1],
                             "GT_PopCount": gtn.get(k, (None, None, None))[2]})
        ev = last_result_event(r.get("raw_files"))
        usage = r.get("usage") or {}
        calls.append({
            "cell": cell, "config": cfg, "thinking": th, "tools_mode": tm, "max_turns": mt,
            "rag_tier": r.get("rag_tier"), "Query_ID": qid, "Book_Title": r.get("Book_Title"),
            "status": "error" if is_err else r.get("generation_status"), "is_error": is_err,
            "attempts_on_disk": count_attempts(jf.parent, qid),
            "year_ok": r.get("year_agrees_with_source"),
            "gt_author_count": len(gtn), "gt_expects_no_result": len(gtn) == 0,
            "pred_author_count": np.nan if is_err else len(pnames),
            "correct": np.nan if is_err else len(correct), "incorrect": np.nan if is_err else len(pnames - set(gtn)),
            "recall": (len(correct) / len(gtn)) if (gtn and not is_err) else np.nan,
            "precision": (len(correct) / len(pnames)) if (pnames and not is_err) else np.nan,
            "no_result_correct": (len(pnames) == 0) if (not gtn and not is_err) else np.nan,
            "birthplace_match_rate": np.nanmean(bp_hits) if bp_hits and not all(np.isnan(bp_hits)) else np.nan,
            "pop_median_rel_err": np.nanmedian(pop_errs) if pop_errs and not all(np.isnan(pop_errs)) else np.nan,
            "cost_usd": r.get("cost_usd_reported"), "latency_s": r.get("generation_time_seconds"),
            "output_tokens": usage.get("output_tokens"), "tool_calls": r.get("n_tool_calls"),
            "num_turns": ev["num_turns"], "result_subtype": ev["result_subtype"],
            "subagent_web_searches": ev["subagent_web_searches"], "models_used": ev["models_used"],
        })

df = pd.DataFrame(calls)
if df.empty:
    sys.exit("no complete cells found")
df["tier_label"] = df["rag_tier"].map(TIER_LABEL)
df.to_csv(OUT / "per_call.csv", index=False)
pd.DataFrame(rows_out).to_csv(OUT / "per_author_row.csv", index=False)

cfg_order = [c[3] for c in CONFIGS if c[3] in set(df["config"])]

# total cost of a cell = every attempt on disk (retries included), read from the events files
def cell_cost_all_attempts(cell):
    tot = 0.0
    for ev in (ROOT / cell / "raw_calls").glob("*_events.jsonl"):
        if "_validate_" in ev.name:
            continue
        for line in open(ev, encoding="utf-8", errors="replace"):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("type") == "result" and e.get("total_cost_usd") is not None:
                tot += float(e["total_cost_usd"])
    return tot

cell = (df.groupby(["config", "rag_tier", "cell"], as_index=False)
          .agg(calls=("Query_ID", "size"), errors=("is_error", "sum"),
               q1_no_result_correct=("no_result_correct", "max"),
               cost_recorded_usd=("cost_usd", "sum"), latency_mean_s=("latency_s", "mean"),
               tool_calls_mean=("tool_calls", "mean"), turns_mean=("num_turns", "mean")))
pv = df.pivot_table(index="cell", columns="Query_ID", values="pred_author_count", aggfunc="first")
cell["pred_authors_q1"] = cell["cell"].map(pv.get("1", pd.Series(dtype=float)))
cell["pred_authors_q2"] = cell["cell"].map(pv.get("2", pd.Series(dtype=float)))
q2 = df[df["Query_ID"] == "2"].set_index("cell")
cell["q2_recall"] = cell["cell"].map(q2["recall"]); cell["q2_precision"] = cell["cell"].map(q2["precision"])
cell["cost_all_attempts_usd"] = cell["cell"].map(cell_cost_all_attempts)
cell["tier_label"] = cell["rag_tier"].map(TIER_LABEL)
cell.to_csv(OUT / "per_cell.csv", index=False)


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def grouped_bars(metric_by, title, ylabel, fname, fmt="{:.2f}", log=False):
    tiers = [t for t in TIER_ORDER if t in set(df["rag_tier"])]
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
    present = [c for c in cfg_order if any(k[0] == c for k in metric_by)]
    n = len(present); w = 0.8 / n; x = np.arange(len(tiers))
    for i, c in enumerate(present):
        vals = [metric_by.get((c, t), np.nan) for t in tiers]
        b = ax.bar(x + (i - (n - 1) / 2) * w, vals, w * 0.92, color=CFG_COLOR[c], label=c, edgecolor="white", lw=1)
        for xi, v in zip(x + (i - (n - 1) / 2) * w, vals):
            if v == v:
                ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([TIER_LABEL[t] for t in tiers])
    if log: ax.set_yscale("log")
    style(ax, title, "RAG evidence tier", ylabel)
    ax.legend(frameon=False, fontsize=8, ncol=min(n, 3), loc="upper left", bbox_to_anchor=(0, -0.14))
    fig.tight_layout(); fig.savefig(OUT / fname, bbox_inches="tight"); plt.close(fig)


# 1. accuracy heatmap: rows = config, cols = tier; two panels (Q1 abstention, Q2 recall)
tiers = [t for t in TIER_ORDER if t in set(df["rag_tier"])]
fig, axes = plt.subplots(1, 2, figsize=(13, 0.75 * len(cfg_order) + 2.2), dpi=150)
for ax, (qid, metric, title) in zip(axes, [("1", "pred_author_count", "Q1 'Dianetics' (GT: no qualifying author) — authors listed (0 = correct)"),
                                           ("2", "correct", "Q2 'Speak, Memory' (GT: Wouk, Michener, Cheever) — GT authors found of 3")]):
    sub = df[df["Query_ID"] == qid]
    M = np.full((len(cfg_order), len(tiers)), np.nan)
    for i, c in enumerate(cfg_order):
        for j, t in enumerate(tiers):
            v = sub[(sub["config"] == c) & (sub["rag_tier"] == t)][metric]
            if len(v) and v.notna().any():
                M[i, j] = v.iloc[0]
    cmap = plt.get_cmap("Blues") if qid == "2" else plt.get_cmap("Blues_r")
    vmax = 3 if qid == "2" else max(6, np.nanmax(M) if np.isfinite(np.nanmax(M)) else 6)
    cmap = cmap.copy(); cmap.set_bad("#f2f1ee")              # not run / not complete -> neutral, not a score colour
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = "not run" if np.isnan(v) else f"{int(v)}"
            if not np.isnan(v):
                frac = v / vmax
                dark = frac > 0.55 if qid == "2" else frac < 0.45
            else:
                dark = False
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="white" if dark else INK)
    ax.set_xticks(range(len(tiers))); ax.set_xticklabels([TIER_LABEL[t] for t in tiers], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(cfg_order))); ax.set_yticklabels(cfg_order, fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "01_accuracy_heatmap.png", bbox_inches="tight"); plt.close(fig)

# 2. cost per cell (all attempts, retries included)
grouped_bars({(r.config, r.rag_tier): r.cost_all_attempts_usd for r in cell.itertuples()},
             "Cost per cell (2 queries, every attempt on disk incl. retries), USD", "USD (log scale)",
             "02_cost_per_cell.png", fmt="${:.2f}", log=True)
# 3. mean latency per call (record-producing attempt)
grouped_bars({(r.config, r.rag_tier): r.latency_mean_s for r in cell.itertuples()},
             "Mean latency per stored call, seconds", "seconds (log scale)", "03_latency_per_call.png", fmt="{:.0f}", log=True)
# 4. web tool calls per call (web configs only)
web = cell[cell["config"].str.contains("web")]
if not web.empty:
    grouped_bars({(r.config, r.rag_tier): r.tool_calls_mean for r in web.itertuples()},
                 "Mean tool calls (WebSearch + WebFetch) per stored call", "tool calls", "04_tool_calls.png", fmt="{:.0f}")

# 5. Q2 precision vs recall scatter per call
sub = df[(df["Query_ID"] == "2") & df["recall"].notna()]
fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
rng = np.random.default_rng(0)
for c in cfg_order:
    s = sub[sub["config"] == c]
    ax.scatter(s["recall"] + rng.uniform(-0.025, 0.025, len(s)), s["precision"].fillna(0) + rng.uniform(-0.025, 0.025, len(s)),
               s=60, color=CFG_COLOR[c], edgecolor="white", lw=1.5, label=c, zorder=3)
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
style(ax, "Q2 'Speak, Memory': author precision vs recall per cell", "recall (GT authors found / 3)", "precision (GT authors / listed)")
ax.xaxis.grid(True, color=GRID, lw=0.8)
ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout(); fig.savefig(OUT / "05_q2_precision_recall.png", bbox_inches="tight"); plt.close(fig)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 40)
print(cell[["config", "tier_label", "calls", "errors", "pred_authors_q1", "q2_recall", "q2_precision", "pred_authors_q2",
            "cost_recorded_usd", "cost_all_attempts_usd", "latency_mean_s", "tool_calls_mean", "turns_mean"]]
      .sort_values(["config", "tier_label"]).to_string(index=False))
print()
print(df[["config", "tier_label", "Query_ID", "status", "attempts_on_disk", "result_subtype", "num_turns", "tool_calls",
          "subagent_web_searches", "models_used", "year_ok", "birthplace_match_rate", "pop_median_rel_err"]]
      .sort_values(["config", "tier_label", "Query_ID"]).to_string(index=False))
print()
print("TOTAL cost, all attempts, complete cells:", round(cell["cost_all_attempts_usd"].sum(), 3))
print("by config:"); print(cell.groupby("config")[["cost_all_attempts_usd"]].sum().round(3).to_string())
print("outputs ->", OUT)
