#!/usr/bin/env python3
"""
Book Author Analyser - Claude Code (cloud session) Edition
==========================================================
SCRIPT_VERSION 2026-11-05.1-books-claude-code   (port of the books HuggingFace edition v10.9.4)

  NLQ_GENERAL  x  15 strategies (S1-S9, S11-S15) + all  x  7 RAG tiers (RAG applies to S1 only, as in the HF script)

Question under test (per target book):
  Which authors have EVER won the Booker Prize or the Pulitzer Prize, AND authored at least one book published in the
  same calendar year as <target book>? For each: birthplace and the population of that birthplace.

SCOPE
  Claude models ONLY (8 aliases). Every call is a streamed headless Claude Code invocation (`claude -p`) made from
  inside a Claude Code cloud session. No HuggingFace, Bedrock, API key or GPU. Prompts, strategies, the
  PRIZE WINNERS / CO-YEAR CHECK scaffold (FIX_54..57), the few-shot exemplar, RAG tiers, evidence stores and the parser
  with its scaffold diagnostics are those of the HF script.
  Not portable: the assistant-turn prefill ("ANSWER:\\n" / "PRIZE WINNERS:\\n") - `claude -p` cannot prefill. The cue
  "ANSWER:" comes from the prompt's output-format text instead.

CALL CONFIGURATION
  claude -p --model <id> --output-format stream-json --verbose --include-partial-messages --tools "" --max-turns 1
         [--system-prompt <neutral text>]        (--system_mode neutral, default)
  prompt via stdin; cwd = empty temp dir; output cap via CLAUDE_CODE_MAX_OUTPUT_TOKENS; optional MAX_THINKING_TOKENS.
  temperature / top_p are NOT controllable. --system_mode default: no flag, Claude Code's own system prompt applies.

EVIDENCE ON DISK (nothing is discarded)
  <cell dir>/raw_calls/<label>_try<n>_prompt.txt | _events.jsonl | _text.txt | _stderr.txt
  <output root>/run.log
  A call that hits the output-token cap is NOT retried: it is stored as a truncated record with its partial text.
  Generated_Response is the model text exactly as returned (the HF script replaced / truncated it on 'degenerate'
  output; here that is only FLAGGED). Self-Ask trimming only affects parsing. --reparse is offline.

Output root: ./claude_code_book_strategies   (RAG cells: ..._S1_FLAT_RAG<n>_<TIER>)
"""
import os
import re
import sys
import json
import math
import time
import atexit
import shutil
import hashlib
import logging
import tempfile
import argparse
import threading
import traceback
import subprocess
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

import numpy as np
import pandas as pd

SCRIPT_VERSION = "2026-11-05.1-books-claude-code"

# ---------------------------------------------------------------------------
# CONSTANTS - Claude models only
# ---------------------------------------------------------------------------
FAMILY = "claude"
WEB_DEFAULT_MAX_TURNS = 10
MODEL_REGISTRY: Dict[str, str] = {
    "claude-opus-4-8":   "claude-opus-4-8",
    "claude-opus-4-7":   "claude-opus-4-7",
    "claude-opus-4-6":   "claude-opus-4-6",
    "claude-opus-4-5":   "claude-opus-4-5-20251101",
    "claude-opus-4-1":   "claude-opus-4-1-20250805",
    "claude-haiku-4-5":  "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
}
MODEL_SIZE_LABEL: Dict[str, str] = {
    "claude-opus-4-8": "opus-4-8", "claude-opus-4-7": "opus-4-7", "claude-opus-4-6": "opus-4-6",
    "claude-opus-4-5": "opus-4-5", "claude-opus-4-1": "opus-4-1", "claude-haiku-4-5": "haiku-4-5",
    "claude-sonnet-4-6": "sonnet-4-6", "claude-sonnet-4-5": "sonnet-4-5",
}
DEFAULT_MODEL_ALIAS = "claude-sonnet-4-6"

SOURCE_FILE       = "source_books_writers.csv"
REFERENCE_FILE    = "writers_books_by_year.csv"
GROUND_TRUTH_FILE = "df_books_modified.csv"

OUTPUT_ROOT = Path("./claude_code_book_strategies")
ANALYSIS_DIR_NAME = "analysis"

GENERATION_MAX_NEW_TOKENS = 3000      # output-token cap per call (raise with --max_new_tokens)
MAX_RETRIES    = 3
RETRY_BASE_SEC = 5.0
REQUEST_TIMEOUT = 1800
MAX_CONSECUTIVE_BAD = 3               # a cell stops after this many consecutive errors / token-cap hits
PROGRESS_LOG_EVERY_SEC = 30
NEUTRAL_SYSTEM = "You are a helpful assistant. Answer the user's message directly in plain text."

QUERY_MODE_NLQ_GENERAL = "nlq_general"
SUPPORTED_MODES: List[str] = [QUERY_MODE_NLQ_GENERAL]
MODE_SUFFIX: Dict[str, str] = {QUERY_MODE_NLQ_GENERAL: "NLQ_GENERAL"}

RECALL_SCAFFOLD_ENABLED  = True       # --no_recall_scaffold switches it off
FEWSHOT_EXEMPLAR_ENABLED = True       # --no_fewshot switches it off
FEWSHOT_MAX_ROWS         = 3
RECALL_MIN_CANDIDATES    = 8
RAG_EVIDENCE_MAX_ROWS    = 40         # HF default; --rag_max_rows 0 = inject every record
RAG_DISTRACTOR_MAX_ROWS  = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
STRATEGY_FLAT_CONJUNCTIVE   = "flat_conjunctive"
STRATEGY_SEQUENTIAL_CHAIN   = "sequential_chain"
STRATEGY_VARIABLE_BINDING   = "variable_binding"
STRATEGY_ENTITY_FIRST       = "entity_first"
STRATEGY_CONSTRAINT_FIRST   = "constraint_first"
STRATEGY_PAIRWISE_EDGES     = "pairwise_edges"
STRATEGY_CONTRASTIVE        = "contrastive_constraint"
STRATEGY_SELF_CHECK         = "self_check"
STRATEGY_MATERIALIZATION    = "materialization_checkpoint"
STRATEGY_TEMPORAL_SCOPE     = "temporal_scope_disambiguation"
STRATEGY_SELF_ASK           = "self_ask"
STRATEGY_FACTORIZATION      = "factorization"
STRATEGY_PROB_CHAINING      = "probabilistic_chaining"
STRATEGY_BACKWARD_GOAL      = "backward_goal_decomposition"
STRATEGY_DUAL_PATH          = "dual_path_consistency"
STRATEGY_ALL                = "all"
SUPPORTED_STRATEGIES: List[str] = [
    STRATEGY_FLAT_CONJUNCTIVE, STRATEGY_SEQUENTIAL_CHAIN, STRATEGY_VARIABLE_BINDING,
    STRATEGY_ENTITY_FIRST, STRATEGY_CONSTRAINT_FIRST, STRATEGY_PAIRWISE_EDGES,
    STRATEGY_CONTRASTIVE, STRATEGY_SELF_CHECK, STRATEGY_MATERIALIZATION,
    STRATEGY_TEMPORAL_SCOPE, STRATEGY_SELF_ASK, STRATEGY_FACTORIZATION,
    STRATEGY_PROB_CHAINING, STRATEGY_BACKWARD_GOAL, STRATEGY_DUAL_PATH, STRATEGY_ALL,
]
INDIVIDUAL_STRATEGIES: List[str] = [s for s in SUPPORTED_STRATEGIES if s != STRATEGY_ALL]
STRATEGY_SUFFIX: Dict[str, str] = {
    STRATEGY_FLAT_CONJUNCTIVE: "S1_FLAT", STRATEGY_SEQUENTIAL_CHAIN: "S2_SEQ", STRATEGY_VARIABLE_BINDING: "S3_VAR",
    STRATEGY_ENTITY_FIRST: "S4A_ENT", STRATEGY_CONSTRAINT_FIRST: "S4B_CON", STRATEGY_PAIRWISE_EDGES: "S5_PAIR",
    STRATEGY_CONTRASTIVE: "S6_CNEG", STRATEGY_SELF_CHECK: "S7_CHK", STRATEGY_MATERIALIZATION: "S8_CKPT",
    STRATEGY_TEMPORAL_SCOPE: "S9_TSCOPE", STRATEGY_SELF_ASK: "S11_SASK", STRATEGY_FACTORIZATION: "S12_FACT",
    STRATEGY_PROB_CHAINING: "S13_PCHAIN", STRATEGY_BACKWARD_GOAL: "S14_BGOAL", STRATEGY_DUAL_PATH: "S15_DPATH",
}
STRATEGY_CODE: Dict[str, str] = {s: STRATEGY_SUFFIX[s].split("_")[0] for s in INDIVIDUAL_STRATEGIES}
STRATEGY_HINT: Dict[str, str] = {
    STRATEGY_FLAT_CONJUNCTIVE: "Apply all conditions simultaneously in a single pass.",
    STRATEGY_SEQUENTIAL_CHAIN: "Work through publication year → co-year books → authors → prize filter → birthplace → population in order.",
    STRATEGY_VARIABLE_BINDING: "Bind: year=publication_year(book), co_authors=authors(books(year)), prize_set={a in co_authors | won Booker or Pulitzer Prize}, then report birthplace+population.",
    STRATEGY_ENTITY_FIRST:     "Start from the book, expand outward: book→year→co-year books→authors→prize filter→birthplace→population.",
    STRATEGY_CONSTRAINT_FIRST: "Start from Booker or Pulitzer Prize winners, filter to those with a co-year book, then report birthplace and population.",
    STRATEGY_PAIRWISE_EDGES:   "Establish each relation independently (author-book, book-year, author-prize, person-birthplace, place-population) then compose.",
    STRATEGY_CONTRASTIVE:      "Include authors satisfying BOTH prize AND co-year conditions; exclude any missing either.",
    STRATEGY_SELF_CHECK:       "After each reasoning step, verify the result before proceeding. Write 'unknown' for any unverified fact.",
    STRATEGY_MATERIALIZATION:  "Materialize intermediate results (especially publication year) as checkpoints; use each checkpoint to anchor downstream reasoning.",
    STRATEGY_TEMPORAL_SCOPE:   "Separate two temporal scopes: (A) year-specific (books published in target year) vs. (B) career-wide (prize in any year). Keep both in parallel.",
    STRATEGY_SELF_ASK:         "Use Self-Ask decomposition: explicitly ask and answer follow-up sub-questions before finalizing, then converge to a single self-contained answer.",
    STRATEGY_FACTORIZATION:    "Factorize the question into independent atomic factors (year, co-year books, prize winners, birthplace, population); solve each factor independently, then combine via explicit logical AND.",
    STRATEGY_PROB_CHAINING:    "Decompose into ordered units; assign each unit a confidence score in [0,1] and propagate a joint confidence via the probability chain rule instead of a hard AND filter; keep only authors above a 0.5 joint-confidence threshold.",
    STRATEGY_BACKWARD_GOAL:    "Work backward from the required output line to its sub-goals until reaching a base case with no further dependencies (publication year), then resolve forward from that base case.",
    STRATEGY_DUAL_PATH:        "Solve via two independent decomposition paths (year-first and prize-first) and cross-check that both yield the same author set before finalizing the answer.",
}

# ---------------------------------------------------------------------------
# RAG evidence tiers (applied to S1 flat_conjunctive only, as in the HF script)
# ---------------------------------------------------------------------------
RAG_TIER_NO_EVIDENCE            = "no_evidence"
RAG_TIER_YEAR_LIST_ONLY         = "year_list_only"
RAG_TIER_TITLES_AUTHORS         = "titles_authors"
RAG_TIER_TITLES_AUTHORS_PRIZE   = "titles_authors_prize"
RAG_TIER_FULL_WITH_BIRTHPLACE   = "full_with_birthplace"
RAG_TIER_FULL_WITH_POPULATION   = "full_with_population"
RAG_TIER_FULL_WITH_DISTRACTORS  = "full_with_distractors"
RAG_TIERS: List[str] = [
    RAG_TIER_NO_EVIDENCE, RAG_TIER_YEAR_LIST_ONLY, RAG_TIER_TITLES_AUTHORS, RAG_TIER_TITLES_AUTHORS_PRIZE,
    RAG_TIER_FULL_WITH_BIRTHPLACE, RAG_TIER_FULL_WITH_POPULATION, RAG_TIER_FULL_WITH_DISTRACTORS,
]
REF_TIERS = (RAG_TIER_YEAR_LIST_ONLY, RAG_TIER_TITLES_AUTHORS, RAG_TIER_TITLES_AUTHORS_PRIZE)
GT_TIERS = (RAG_TIER_FULL_WITH_BIRTHPLACE, RAG_TIER_FULL_WITH_POPULATION, RAG_TIER_FULL_WITH_DISTRACTORS)
RAG_TIER_SUFFIX: Dict[str, str] = {
    RAG_TIER_NO_EVIDENCE: "RAG0_NONE", RAG_TIER_YEAR_LIST_ONLY: "RAG1_TITLES", RAG_TIER_TITLES_AUTHORS: "RAG2_AUTHORS",
    RAG_TIER_TITLES_AUTHORS_PRIZE: "RAG3_PRIZE", RAG_TIER_FULL_WITH_BIRTHPLACE: "RAG4_BIRTHPLACE",
    RAG_TIER_FULL_WITH_POPULATION: "RAG5_POPULATION", RAG_TIER_FULL_WITH_DISTRACTORS: "RAG6_DISTRACTORS",
}
RAG_TIER_DESCRIPTION: Dict[str, str] = {
    RAG_TIER_NO_EVIDENCE:           "No evidence — knowledge retrieval / reasoning from parametric memory.",
    RAG_TIER_YEAR_LIST_ONLY:        "Co-year title list only (REFERENCE_FILE) — candidate identification + knowledge retrieval.",
    RAG_TIER_TITLES_AUTHORS:        "Titles + authors (REFERENCE_FILE) — entity linking + further knowledge retrieval.",
    RAG_TIER_TITLES_AUTHORS_PRIZE:  "Titles + authors + prize (REFERENCE_FILE) — filtering + further retrieval.",
    RAG_TIER_FULL_WITH_BIRTHPLACE:  "Candidate authors + birthplace, no population (GROUND_TRUTH_FILE) — multi-hop comprehension + population retrieval.",
    RAG_TIER_FULL_WITH_POPULATION:  "Complete evidence including population (GROUND_TRUTH_FILE) — reading comprehension + multi-hop synthesis.",
    RAG_TIER_FULL_WITH_DISTRACTORS: "Complete evidence + distractors (GROUND_TRUTH_FILE) — comprehension + relevance filtering + synthesis.",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def model_name_to_slug(model_name: str) -> str:
    return model_name.replace(":", "_").replace(".", "_")


def _resolve_path(filename: str) -> Path:
    for c in [Path(filename), Path(__file__).resolve().parent / filename, Path.cwd() / filename]:
        if c.exists():
            return c
    raise FileNotFoundError(f"'{filename}' not found in cwd or script directory.")


def clean_year(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "unknown", ""):
        return ""
    m = re.match(r"^(\d{3,4})(?:\.0+)?$", s)
    if m:
        return m.group(1).zfill(4)
    try:
        f = float(s)
        if math.isnan(f):
            return ""
        return f"{int(f):04d}"
    except (ValueError, TypeError):
        return ""


_SENTINELS = {"no result", "unknown", "na", "none", "error", "declined"}


def is_sentinel_author(s: str) -> bool:
    n = " ".join(re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split())
    return n in _SENTINELS or n.startswith("no result") or n.endswith(" error")


def is_error_row(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+_ERROR", str(s).strip()))


def _ascii_fold(s: Any) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def name_key(s: Any) -> str:
    t = re.sub(r"\([^)]*\)", " ", str(s))
    t = re.sub(r"[^a-z0-9 ]+", " ", _ascii_fold(t).lower())
    return " ".join(w for w in t.split() if w not in {"jr", "sr", "ii", "iii"})


def names_match(a: Any, b: Any) -> bool:
    ka, kb = name_key(a), name_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    ta, tb = set(ka.split()), set(kb.split())
    small, large = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return small <= large and (len(large) - len(small)) <= 2


def place_key(s: Any) -> str:
    first = str(s).split(",")[0]
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", _ascii_fold(first).lower()).split())


def places_match(a: Any, b: Any) -> bool:
    ka, kb = place_key(a), place_key(b)
    return bool(ka and kb and (ka == kb or ka in kb or kb in ka))


# ---- population parsing (float-first for machine numbers; unit-aware for model text) ----
_UNIT = {"billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "millions": 1e6, "mn": 1e6, "mm": 1e6, "m": 1e6,
         "thousand": 1e3, "k": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "crore": 1e7, "cr": 1e7, "hundred": 100.0}
_UNIT_RE = r"billion|bn|millions?|mn|mm|thousand|lakhs?|lac|crore|cr|hundred|[bmk]"
_NUM_RE = re.compile(
    r"(?<![\w.])("
    r"\d{1,3}(?:[,\u00a0\u202f ]\d{3})+(?:\.\d+)?"
    r"|\d{1,3}(?:\.\d{3})+(?!\d)"
    r"|\d+(?:[.,]\d+)?"
    r")\s*(" + _UNIT_RE + r")?(?![a-z])", re.I)
_RANGE_RE = re.compile(r"(\d[\d.,]*)\s*(?:-|\u2013|\u2014|to)\s*(\d[\d.,]*)\s*(" + _UNIT_RE + r")(?![a-z])", re.I)
_SCI_RE = re.compile(r"[+-]?\d+(?:\.\d+)?[eE][+-]?\d+")
_WORD_NUM_RE = re.compile(r"\b(?:one|a)\s+(billion|million|thousand)\b", re.I)


def _to_float(tok: str, has_unit: bool) -> float:
    t = tok.replace("\u00a0", " ").replace("\u202f", " ").strip()
    if re.fullmatch(r"\d{1,3}(?:[, ]\d{3})+(?:\.\d+)?", t):
        return float(re.sub(r"[, ]", "", t))
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", t):
        return float(t) if (has_unit and t.count(".") == 1) else float(t.replace(".", ""))
    if re.fullmatch(r"\d+,\d{1,2}", t):
        return float(t.replace(",", "."))
    return float(t.replace(",", ""))


def parse_population_value(text: Any) -> Optional[float]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    if _SCI_RE.fullmatch(s):
        return float(s)
    if not re.search(r"\d", s):
        m = _WORD_NUM_RE.search(s)
        return _UNIT[m.group(1).lower()] if m else None
    rng = _RANGE_RE.search(s)
    if rng:
        try:
            a, b = _to_float(rng.group(1), True), _to_float(rng.group(2), True)
            return (a + b) / 2.0 * _UNIT[rng.group(3).lower()]
        except ValueError:
            pass
    for m in _NUM_RE.finditer(s):
        tok, unit = m.group(1), (m.group(2) or "").lower()
        if not unit and re.fullmatch(r"(?:1[5-9]|20)\d{2}", tok) and \
                re.search(r"(?:as of|in|census|est\.?|\()\s*$", s[:m.start()].lower()):
            continue
        try:
            return _to_float(tok, bool(unit)) * _UNIT.get(unit, 1.0)
        except ValueError:
            continue
    return None


def pop_to_str(raw: Any) -> str:
    v = parse_population_value(raw)
    return str(int(round(v))) if v is not None else ""


def gt_pop_value(raw: Any) -> Optional[int]:
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    try:
        f = float(s.replace(",", ""))
        if math.isfinite(f):
            return int(round(f))
    except ValueError:
        pass
    v = parse_population_value(s)
    return int(round(v)) if v is not None else None


def gt_pop_to_str(raw: Any) -> str:
    v = gt_pop_value(raw)
    return str(v) if v is not None else ""


ST_OK, ST_TRUNC_LENGTH = "ok", "truncated_length"
ST_TRUNC_REASONING, ST_EMPTY = "truncated_in_reasoning", "empty"


# ---------------------------------------------------------------------------
# RAG evidence stores
# ---------------------------------------------------------------------------
@dataclass
class ReferenceRecord:
    year: str
    title: str
    author: str
    prize: str
    title_uri: str = ""
    author_uri: str = ""


@dataclass
class GroundTruthRecord:
    sno: str
    book_title: str
    book_uri: str
    prize_type: str
    year: str
    author: str
    birthplace: str
    popcount: str


class ReferenceStore:
    REQUIRED_COLUMNS = {"Year", "Prize", "Title", "Author"}

    def __init__(self, path: str = REFERENCE_FILE):
        self.path = path
        self.by_year: Dict[str, List[ReferenceRecord]] = {}
        self._load()

    def _load(self):
        candidate = _resolve_path(self.path)
        df = pd.read_csv(candidate)
        df.columns = [c.strip() for c in df.columns]
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"'{candidate}' missing columns {sorted(missing)}. Found: {list(df.columns)}")
        df = df.fillna("")
        count = 0
        for _, row in df.iterrows():
            year = clean_year(row.get("Year", ""))
            if not year:
                continue
            self.by_year.setdefault(year, []).append(ReferenceRecord(
                year=year, title=str(row.get("Title", "")).strip(), author=str(row.get("Author", "")).strip(),
                prize=str(row.get("Prize", "")).strip(), title_uri=str(row.get("TitleURI", "")).strip(),
                author_uri=str(row.get("AuthorURI", "")).strip()))
            count += 1
        logger.info(f"ReferenceStore loaded {count} rows from '{candidate}' spanning {len(self.by_year)} years.")

    def co_year_records(self, year: str, exclude_title: str = "") -> List[ReferenceRecord]:
        recs = self.by_year.get(year, [])
        if exclude_title:
            recs = [r for r in recs if r.title.lower() != exclude_title.strip().lower()]
        return recs


class GroundTruthStore:
    REQUIRED_COLUMNS = {"S.No", "BookTitle", "Prize_Type", "PublicationYear", "Author", "Birthplace", "PopCount"}

    def __init__(self, path: str = GROUND_TRUTH_FILE):
        self.path = path
        self.by_sno: Dict[str, List[GroundTruthRecord]] = {}
        self.all_records: List[GroundTruthRecord] = []
        self._load()

    def _load(self):
        candidate = _resolve_path(self.path)
        df = pd.read_csv(candidate)
        df.columns = [c.strip() for c in df.columns]
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"'{candidate}' missing columns {sorted(missing)}. Found: {list(df.columns)}")
        df = df.fillna("")
        for _, row in df.iterrows():
            sno = str(row.get("S.No", "")).strip()
            year = clean_year(row.get("PublicationYear", ""))
            if not sno or not year:
                continue
            rec = GroundTruthRecord(
                sno=sno, book_title=str(row.get("BookTitle", "")).strip(), book_uri=str(row.get("Book_URI", "")).strip(),
                prize_type=str(row.get("Prize_Type", "")).strip(), year=year, author=str(row.get("Author", "")).strip(),
                birthplace=str(row.get("Birthplace", "")).strip(), popcount=gt_pop_to_str(row.get("PopCount", "")))
            self.by_sno.setdefault(sno, []).append(rec)
            self.all_records.append(rec)
        n_complete = sum(1 for r in self.all_records
                         if r.author and r.author.upper() != "NO RESULT" and r.birthplace and r.popcount)
        n_no_result = sum(1 for r in self.all_records if r.author.upper() == "NO RESULT")
        logger.info(f"GroundTruthStore loaded {len(self.all_records)} rows from '{candidate}' spanning "
                    f"{len(self.by_sno)} distinct S.No entries ({n_complete} fully-populated, {n_no_result} marked NO RESULT).")

    def records_for(self, sno: str, expected_year: str) -> List[GroundTruthRecord]:
        verified = []
        for r in self.by_sno.get(str(sno), []):
            if expected_year and r.year != expected_year:
                logger.warning(f"GroundTruthStore: S.No={sno} year mismatch (ground_truth={r.year} vs source={expected_year}) — skipping record.")
                continue
            verified.append(r)
        return verified

    def sample_distractors(self, exclude_sno: str, n: int = 5) -> List[GroundTruthRecord]:
        pool = [r for r in self.all_records if r.sno != str(exclude_sno)]
        if not pool:
            return []
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        return [pool[i] for i in idx]


class FewShotExemplarStore:
    """One worked example (a DIFFERENT book) from fully-populated ground-truth rows. Caution: it uses GT rows, so an
    exemplar book from the same publication year would leak candidates; disable with --no_fewshot for a clean run."""

    def __init__(self, gt_store: Optional[GroundTruthStore], max_rows: int = FEWSHOT_MAX_ROWS):
        self.max_rows = max_rows
        self._ranked_snos: List[str] = []
        self._by_sno: Dict[str, List[GroundTruthRecord]] = {}
        self._cache: Dict[str, Optional[str]] = {}
        self.available = False
        if gt_store is not None:
            self._build_index(gt_store)

    def _build_index(self, gt_store: GroundTruthStore):
        for r in gt_store.all_records:
            if not r.author or r.author.upper() == "NO RESULT" or not r.birthplace or not r.popcount:
                continue
            self._by_sno.setdefault(r.sno, []).append(r)
        self._ranked_snos = [sno for sno, _ in sorted(self._by_sno.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))]
        self.available = bool(self._ranked_snos)
        if self.available:
            head = self._ranked_snos[0]
            logger.info(f"FewShotExemplarStore: {len(self._ranked_snos)} usable S.No entries; primary exemplar S.No={head} "
                        f"('{self._by_sno[head][0].book_title}', {len(self._by_sno[head])} rows).")
        else:
            logger.warning("FewShotExemplarStore: no fully-populated ground-truth rows found.")

    def render(self, exclude_sno: str) -> Optional[str]:
        if not self.available:
            return None
        key = str(exclude_sno)
        if key in self._cache:
            return self._cache[key]
        chosen = next((s for s in self._ranked_snos if s != key), None)
        if chosen is None:
            self._cache[key] = None
            return None
        recs = self._by_sno[chosen][:self.max_rows]
        lines = ["Worked example — a DIFFERENT book, showing exactly what a correct and complete answer looks like:", "",
                 f'Target book: "{recs[0].book_title}"', "ANSWER:", f"Publication year: {recs[0].year}"]
        for r in recs:
            lines.append(f"{r.author} | {r.birthplace} | {r.popcount}")
        lines += ["", "Notice that the answer contains actual author rows. Produce an answer of the same shape for the "
                      "target book below."]
        self._cache[key] = "\n".join(lines)
        return self._cache[key]


def build_rag_evidence_block(tier: str, sno: str, year: str, book_title: str,
                             ref_store: Optional[ReferenceStore], gt_store: Optional[GroundTruthStore],
                             max_rows: int = RAG_EVIDENCE_MAX_ROWS) -> Tuple[Optional[str], Dict[str, Any]]:
    stats: Dict[str, Any] = {"rows_available": 0, "rows_injected": 0, "truncated": False}
    if tier == RAG_TIER_NO_EVIDENCE or not year:
        return None, stats
    lines = [f"RETRIEVED EVIDENCE (S.No={sno}, publication year = {year})", "-" * 60]
    omitted = 0
    if tier in REF_TIERS:
        if ref_store is None:
            return None, stats
        co_year = ref_store.co_year_records(year, exclude_title=book_title)
        if not co_year:
            return None, stats
        stats["rows_available"] = len(co_year)
        if max_rows and len(co_year) > max_rows:
            omitted, co_year = len(co_year) - max_rows, co_year[:max_rows]
        stats["rows_injected"] = len(co_year)
        if tier == RAG_TIER_YEAR_LIST_ONLY:
            lines.append("Titles published in this year (source: reference index):")
            lines += [f"  - {r.title}" for r in co_year]
        elif tier == RAG_TIER_TITLES_AUTHORS:
            lines.append("Titles and authors published in this year:")
            lines += [f"  - {r.title} | {r.author}" for r in co_year]
        else:
            lines.append("Titles, authors, and prize provenance for this year:")
            lines += [f"  - {r.title} | {r.author} | prize_source={r.prize or 'unknown'}" for r in co_year]
    elif tier in GT_TIERS:
        if gt_store is None:
            return None, stats
        gt_recs = gt_store.records_for(sno, expected_year=year)
        if not gt_recs:
            return None, stats
        stats["rows_available"] = len(gt_recs)
        if max_rows and len(gt_recs) > max_rows:
            omitted, gt_recs = len(gt_recs) - max_rows, gt_recs[:max_rows]
        stats["rows_injected"] = len(gt_recs)
        if tier == RAG_TIER_FULL_WITH_BIRTHPLACE:
            lines.append("Candidate authors with prize provenance and birthplace (population withheld):")
            for r in gt_recs:
                if r.author.upper() == "NO RESULT":
                    lines.append(f"  - [{r.prize_type}] No qualifying author found in ground-truth evidence.")
                    continue
                lines.append(f"  - [{r.prize_type}] {r.author} | birthplace={r.birthplace or 'unknown'}")
            lines.append("  NOTE: population is not provided at this tier; resolve it yourself if needed.")
        else:
            lines.append("Complete evidence including population:")
            for r in gt_recs:
                if r.author.upper() == "NO RESULT":
                    lines.append(f"  - [{r.prize_type}] No qualifying author found in ground-truth evidence.")
                    continue
                lines.append(f"  - [{r.prize_type}] {r.author} | birthplace={r.birthplace or 'unknown'} | "
                             f"population={r.popcount or 'unknown'}")
            if tier == RAG_TIER_FULL_WITH_DISTRACTORS:
                lines.append("")
                lines.append("Additional records from OTHER books/years (DISTRACTORS — these belong to a DIFFERENT "
                             "S.No/book and must NOT be used unless independently verified as relevant to THIS book):")
                for r in gt_store.sample_distractors(exclude_sno=sno, n=min(max(len(gt_recs), 5), RAG_DISTRACTOR_MAX_ROWS)):
                    lines.append(f"  - [S.No={r.sno}, year={r.year}, {r.prize_type}] {r.book_title} | {r.author}")
    else:
        return None, stats
    if omitted:
        stats["truncated"] = True
        lines.append(f"  ... ({omitted} further evidence rows omitted for length; the list above is NOT exhaustive)")
    lines.append("-" * 60)
    lines.append("Use the evidence above only where it is directly relevant to THIS book's S.No and year. Do not use "
                 "distractor or mismatched-year rows as qualifying. Where evidence is silent or marked unknown/NO RESULT, "
                 "rely on your own recall rather than inventing a fact.")
    return "\n".join(lines), stats


def evidence_stats(evidence_block: Optional[str]) -> Dict[str, Any]:
    if not evidence_block:
        return {"evidence_row_count": 0, "evidence_no_result_rows": 0, "evidence_char_len": 0, "evidence_all_no_result": False}
    rows = evidence_block.count("\n  - ")
    no_res = evidence_block.count("No qualifying author found")
    return {"evidence_row_count": rows, "evidence_no_result_rows": no_res, "evidence_char_len": len(evidence_block),
            "evidence_all_no_result": bool(rows) and rows == no_res}


def score_authors_against_ground_truth(parsed_rows: List[Dict[str, str]], gt_records: List[GroundTruthRecord]) -> Dict[str, Any]:
    """Per-record exact-name score (HF behaviour). The analysis stage re-scores with tolerant name matching."""
    out = {"gt_available": bool(gt_records), "gt_author_count": 0, "gt_expects_no_result": False,
           "authors_correct_count": 0, "authors_correct_names": [], "authors_incorrect_count": 0,
           "author_recall": None, "author_precision": None}
    if not gt_records:
        return out
    gt_names = {name_key(r.author) for r in gt_records if r.author and r.author.upper() != "NO RESULT"}
    out["gt_expects_no_result"] = bool(gt_records) and not gt_names
    out["gt_author_count"] = len(gt_names)
    parsed_names = {name_key(r["Author"]) for r in parsed_rows
                    if r.get("Author") and not is_sentinel_author(r["Author"])}
    correct = gt_names & parsed_names
    out["authors_correct_count"] = len(correct)
    out["authors_correct_names"] = sorted(correct)
    out["authors_incorrect_count"] = len(parsed_names - gt_names)
    if gt_names:
        out["author_recall"] = round(len(correct) / len(gt_names), 4)
    if parsed_names:
        out["author_precision"] = round(len(correct) / len(parsed_names), 4)
    return out


# ---------------------------------------------------------------------------
# Response-shape helpers (used for PARSING only; the stored response is never altered)
# ---------------------------------------------------------------------------
def _is_interactive_pattern(line: str) -> bool:
    markers = [
        r"^\s*\?", r"(?:would you|can you|please|could you)\s+(?:provide|tell|give|help)",
        r"(?:additional|more|further|next)\s+(?:information|details|question)",
        r"(?:clarif|confirm|verify)\s+(?:this|that|the)", r"(?:do you|did you|have you|are you)",
        r"(?:waiting|expecting|need|require[ds]?)\s+(?:for|your)", r"let me (?:ask|search|check|clarify|know)",
        r"^\s*>", r"(?:need|request|require)\s+(?:clarification|more info)",
    ]
    low = line.lower()
    return any(re.search(p, low) for p in markers)


def _strip_interactive_patterns(text: str) -> str:
    cutoffs = [r"\n\s*(?:What|Which|Who|How|When|Where|Why|Please|To|Do|Can|Would|Could)\s+",
               r"\n\s*>", r"\n\s*let me (?:ask|search|clarify|check)", r"\n\s*(?:do|can|would|could) you",
               r"\n\s*(?:request|asking|seeking)\s+(?:further|additional|more)"]
    for pattern in cutoffs:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            text = text[:m.start()]
    return text.rstrip()


def enforce_single_shot(text: str) -> str:
    am = re.search(r"ANSWER\s*:", text, re.IGNORECASE)
    if not am:
        return text
    section = text[am.end():]
    em = re.search(r"END\s+OF\s+RESPONSE", section, re.IGNORECASE)
    if em:
        section = section[:em.start()]
    lines = section.split("\n")
    last_valid = -1
    for idx, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "|" in line:
            last_valid = idx
        elif last_valid >= 0 and _is_interactive_pattern(line):
            break
    if last_valid >= 0:
        section = "\n".join(lines[:last_valid + 1])
    return text[:am.end()] + section


def detect_degenerate(text: str) -> Tuple[bool, str]:
    """FLAG ONLY (the HF script replaced / truncated the text; here the text is always kept)."""
    if not text:
        return False, ""
    for pattern, label in [(r"(\?[\s\n]*){3,}", "Multiple question marks"), (r"(please[\s\n]*){3,}", "Multiple 'please'"),
                           (r"(what[\s\n]*){5,}", "Repeated 'what'")]:
        if re.search(pattern, text, re.IGNORECASE):
            return True, label
    lines = text.split("\n")
    if len(lines) > 10:
        cnt = Counter(l.strip() for l in lines if l.strip())
        if cnt:
            top, n = cnt.most_common(1)[0]
            if n > 10:
                return True, f"'{top[:60]}' repeated {n}x"
    for tok, grp in groupby(text.split()):
        if sum(1 for _ in grp) > 10:
            return True, f"'{tok[:40]}' repeated consecutively"
    return False, ""


# ---------------------------------------------------------------------------
# SelfAskPromptBuilder (S11)
# ---------------------------------------------------------------------------
class SelfAskPromptBuilder:
    _EXEMPLAR_1 = """\
Question: Who lived longer, Gabriel García Márquez or Kazuo Ishiguro?
Are follow up questions needed here: Yes.
Follow up: When did Gabriel García Márquez die and how old was he?
Intermediate answer: Gabriel García Márquez died in 2014 at age 87.
Follow up: Is Kazuo Ishiguro still living, and how old is he currently?
Intermediate answer: Kazuo Ishiguro is still living.
So the final answer is: Gabriel García Márquez lived longer (so far).
END OF RESPONSE"""
    _EXEMPLAR_2 = """\
Question: What is the population of the birthplace of the author who won the Booker or Pulitzer Prize for "Life of Pi"?
Are follow up questions needed here: Yes.
Follow up: Who wrote "Life of Pi"?
Intermediate answer: Yann Martel wrote "Life of Pi".
Follow up: Did Yann Martel win the Booker or Pulitzer Prize for "Life of Pi"?
Intermediate answer: Yes, Yann Martel won the Booker Prize in 2002 for "Life of Pi".
Follow up: Where was Yann Martel born?
Intermediate answer: Yann Martel was born in Salamanca, Spain.
Follow up: What is the population of Salamanca?
Intermediate answer: Salamanca has a population of approximately 144,000.
So the final answer is: Yann Martel | Salamanca, Spain | 144000
END OF RESPONSE"""
    _OUTPUT_FORMAT = """\
Output format
-------------
After "So the final answer is:" write:
  ANSWER:
  Publication year: [YYYY]
  [Author Name] | [Birthplace] | [Population]
One line per matching author. If none match write: No matching authors found.
Write "unknown" for a field you cannot recall, but still include the author.
Then write: END OF RESPONSE
Do NOT ask follow-up questions. Do NOT wait for user input. Complete immediately."""
    _CRITICAL_INSTRUCTION = """\
CRITICAL SINGLE-SHOT INSTRUCTIONS
----------------------------------
1. You MUST complete your ENTIRE reasoning in ONE response.
2. Do NOT ask the user any questions or requests.
3. Do NOT request clarification, additional input, or feedback.
4. After writing "ANSWER:" with results, immediately write "END OF RESPONSE".
5. Your response must be SELF-CONTAINED and CONCLUSIVE.
6. Generate follow-ups only INTERNALLY, then consolidate into final answer."""
    _ROLE_GENERAL = (
        "You are a literary history research assistant with broad knowledge of "
        "Booker Prize and Pulitzer Prize records, author bibliographies, and "
        "biographical reference data.")
    _NLQ_QUESTION = (
        'Which authors have EVER won the Booker or Pulitzer Prize, AND authored at least '
        'one book published in the same calendar year as "{book_title}"? '
        'For each such author state their birthplace and the population of that birthplace.')
    _QUERY_TEMPLATE = "Question: {question}\nAre follow up questions needed here:"

    @classmethod
    def build(cls, query_text: str, mode: str, query_id: Any) -> str:
        question = f"[Query ID: {query_id}]  " + cls._NLQ_QUESTION.format(book_title=query_text)
        return "\n".join([cls._ROLE_GENERAL, "", cls._CRITICAL_INSTRUCTION, "", "# Few-shot examples of Self-Ask reasoning", "",
                          cls._EXEMPLAR_1, "", cls._EXEMPLAR_2, "", cls._OUTPUT_FORMAT, "", "# Your task", "",
                          cls._QUERY_TEMPLATE.format(question=question)])


# ---------------------------------------------------------------------------
# StrategyBuilder (S1-S9, S12-S15) — FIX_54..57 scaffold in S1 only
# ---------------------------------------------------------------------------
class StrategyBuilder:
    _UNCERTAINTY_TAIL = (
        "Uncertainty policy: do not drop an author because you are unsure of "
        "their birthplace or its population — include the author and write "
        "unknown in the field you cannot determine. Never invent a specific "
        "figure.")

    @classmethod
    def build(cls, book_title: str, strategy: str, evidence_block: Optional[str] = None,
              fewshot_block: Optional[str] = None, recall_scaffold: bool = True) -> str:
        if strategy == STRATEGY_FLAT_CONJUNCTIVE:
            return cls._flat_conjunctive(book_title, evidence_block, fewshot_block, recall_scaffold)
        builders = {
            STRATEGY_SEQUENTIAL_CHAIN: cls._sequential_chain, STRATEGY_VARIABLE_BINDING: cls._variable_binding,
            STRATEGY_ENTITY_FIRST: cls._entity_first, STRATEGY_CONSTRAINT_FIRST: cls._constraint_first,
            STRATEGY_PAIRWISE_EDGES: cls._pairwise_edges, STRATEGY_CONTRASTIVE: cls._contrastive,
            STRATEGY_SELF_CHECK: cls._self_check, STRATEGY_MATERIALIZATION: cls._materialization_checkpoint,
            STRATEGY_TEMPORAL_SCOPE: cls._temporal_scope_disambiguation, STRATEGY_FACTORIZATION: cls._factorization,
            STRATEGY_PROB_CHAINING: cls._probabilistic_chaining, STRATEGY_BACKWARD_GOAL: cls._backward_goal_decomposition,
            STRATEGY_DUAL_PATH: cls._dual_path_consistency,
        }
        if strategy not in builders:
            raise ValueError(f"Unknown strategy '{strategy}'.")
        body = builders[strategy](book_title)
        if fewshot_block:
            body = f"{fewshot_block}\n\n{body}"
        return body

    @classmethod
    def _flat_conjunctive(cls, book_title: str, evidence_block: Optional[str] = None,
                          fewshot_block: Optional[str] = None, recall_scaffold: bool = True) -> str:
        evidence_section = f"\n{evidence_block}\n" if evidence_block else ""
        fewshot_section = f"\n{fewshot_block}\n" if fewshot_block else ""
        if evidence_block:
            gate = (
                "\n\nGrounding requirement: if the RETRIEVED EVIDENCE block above "
                "names one or more authors, output one row for each named author, "
                "copying the name and birthplace exactly as given. Write "
                "\"No matching authors found.\" ONLY if the evidence block itself "
                "states that no qualifying author was found.")
            scaffold = ""
        else:
            gate = (
                "\n\nREFUSAL GATE: write \"No matching authors found.\" ONLY if "
                "your CO-YEAR CHECK marked EVERY candidate `no`. If ANY candidate "
                "is marked `yes` or `unsure`, that candidate MUST appear as a row "
                "in the ANSWER section.")
            scaffold = (
                "\n\nRequired working, BEFORE the answer:\n"
                "YEAR: state the four-digit publication year, then `confident` or "
                "`uncertain`.\n"
                f"PRIZE WINNERS: list at least {RECALL_MIN_CANDIDATES} authors who "
                "have EVER won the Booker or Pulitzer Prize, career-wide. "
                "One name per line, NO yes/no marker in this section.\n"
                "CO-YEAR CHECK: for EACH name in PRIZE WINNERS write exactly "
                "`Name - yes` / `Name - no` / `Name - unsure` for whether they "
                "published any book in the YEAR above.\n") if recall_scaffold else ""
        return (
            f"Task (Strategy 1 — flat conjunctive)\n--------------------------------------\n"
            f"Target book: \"{book_title}\"\n{fewshot_section}{evidence_section}\n"
            f"Find every author who (a) has EVER won the Booker Prize or the "
            f"Pulitzer Prize, AND (b) authored at least one book published in the "
            f"same calendar year as \"{book_title}\". For each matching author "
            f"give their birthplace and the population of that birthplace.\n\n"
            f"Note: The prize-winning book and the co-year book are typically "
            f"different books. Condition (b) does not require the prize-winning "
            f"book — any book published by that author in that year qualifies.\n"
            f"{cls._UNCERTAINTY_TAIL}{scaffold}{gate}")

    @classmethod
    def _sequential_chain(cls, book_title: str) -> str:
        return (
            f"Task (Strategy 2 — sequential chain decomposition)\n----------------------------------------------------\n"
            f"Target book: \"{book_title}\"\n\nWork through the following steps in order:\n\n"
            f"  Step 1 — Publication year\n    Identify the publication year of \"{book_title}\". State it "
            f"explicitly before continuing.\n\n"
            f"  Step 2 — Co-year books\n    List books published in that same year (any genre, any author).\n\n"
            f"  Step 3 — Co-year authors\n    From Step 2, identify the authors of those books.\n\n"
            f"  Step 4 — Prize filter\n    From Step 3, keep only authors who have EVER won the Booker or Pulitzer Prize.\n\n"
            f"  Step 5 — Birthplace\n    For each author retained in Step 4, identify their birthplace.\n\n"
            f"  Step 6 — Population\n    For each birthplace from Step 5, state the population of that place.\n\n"
            f"Label each step. Keep each step concise. {cls._UNCERTAINTY_TAIL}")

    @classmethod
    def _variable_binding(cls, book_title: str) -> str:
        return (
            f"Task (Strategy 3 — variable-centred decomposition)\n----------------------------------------------------\n"
            f"Target book: \"{book_title}\"\n\nEvaluate the following bindings in order:\n\n"
            f"  let year        = publication_year( \"{book_title}\" )\n"
            f"  let co_books    = books_published_in( year )\n"
            f"  let co_authors  = authors_of( co_books )\n"
            f"  let prize_set   = {{ a in co_authors |\n"
            f"                       a has EVER won the Booker or Pulitzer Prize }}\n"
            f"  let birthplaces = birthplace_of( prize_set )\n"
            f"  let populations = population_of( birthplaces )\n\n"
            f"State the value of each binding as you compute it, then produce the "
            f"answer from the final bindings. {cls._UNCERTAINTY_TAIL}")

    @classmethod
    def _entity_first(cls, book_title: str) -> str:
        return (
            f"Task (Strategy 4a — entity-first / outward expansion)\n-------------------------------------------------------\n"
            f"Start from the book entity and expand outward:\n\n  \"{book_title}\"\n"
            f"     -> publication year\n         -> other books published in that same year\n"
            f"             -> authors of those books\n"
            f"                 -> filter: keep only authors who EVER won\n"
            f"                           the Booker or Pulitzer Prize\n"
            f"                     -> birthplace of each retained author\n"
            f"                         -> population of each birthplace\n\n"
            f"{cls._UNCERTAINTY_TAIL}")

    @classmethod
    def _constraint_first(cls, book_title: str) -> str:
        return (
            f"Task (Strategy 4b — constraint-first / inward traversal)\n---------------------------------------------------------\n"
            f"Start from the prize constraint and traverse inward:\n\n"
            f"  {{ all authors who EVER won the Booker or Pulitzer Prize }}\n"
            f"     -> for each such author, list books they authored\n"
            f"         -> filter: keep only books published in the same year\n"
            f"                   as \"{book_title}\"\n"
            f"             -> from the remaining authors\n"
            f"                 -> birthplace of each author\n"
            f"                     -> population of each birthplace\n\n"
            f"Begin by naming at least {RECALL_MIN_CANDIDATES} prize winners "
            f"before applying any year filter. {cls._UNCERTAINTY_TAIL}")

    @classmethod
    def _pairwise_edges(cls, book_title: str) -> str:
        return (
            f"Task (Strategy 5 — pairwise edge factoring)\n---------------------------------------------\n"
            f"Target book: \"{book_title}\"\n\nFirst establish each relational edge independently:\n\n"
            f"  Edge A  author-book:       which authors wrote which books?\n"
            f"  Edge B  book-year:         what year was \"{book_title}\" published?\n"
            f"  Edge C  year-book:         which books were published in that same year?\n"
            f"  Edge D  author-prize:      which authors have EVER won the Booker or Pulitzer Prize?\n"
            f"  Edge E  person-birthplace: where was each qualifying author born?\n"
            f"  Edge F  place-population:  what is the population of each birthplace?\n\n"
            f"Then compose the join:\n  (Edge D intersect Edge A applied to Edge C results)\n  -> apply Edge E -> apply Edge F\n\n"
            f"State each edge result before composing. {cls._UNCERTAINTY_TAIL}")

    @classmethod
    def _contrastive(cls, book_title: str) -> str:
        return (
            f"Task (Strategy 6 — contrastive constraint prompting)\n------------------------------------------------------\n"
            f"Target book: \"{book_title}\"\n\nINCLUSION criteria — an author must satisfy ALL of the following:\n"
            f"  [INCLUDE A]  Has EVER won the Booker or Pulitzer Prize.\n"
            f"  [INCLUDE B]  Authored >=1 book published in the same year as \"{book_title}\".\n\n"
            f"EXCLUSION criteria — drop an author if ANY of the following:\n"
            f"  [EXCLUDE 1]  Has NOT won the Booker or Pulitzer Prize.\n"
            f"  [EXCLUDE 2]  Every book is in a different year from \"{book_title}\".\n"
            f"  [EXCLUDE 3]  Only co-year book IS \"{book_title}\" itself.\n\n"
            f"Note that uncertainty about birthplace or population is NOT an "
            f"exclusion criterion — such an author is still included, with "
            f"unknown in the missing field.\n"
            f"Report birthplace and population for each surviving author.\n{cls._UNCERTAINTY_TAIL}")

    @classmethod
    def _self_check(cls, book_title: str) -> str:
        return (
            f"Task (Strategy 7 — intermediate verification / self-check)\n-------------------------------------------------------------\n"
            f"Target book: \"{book_title}\"\n\n"
            f"  Step 1 — Publication year.  State it, then state whether you are confident or uncertain.\n"
            f"  Step 2 — Booker or Pulitzer Prize winners. List at least {RECALL_MIN_CANDIDATES} names.\n"
            f"  Step 3 — Co-year filter. Keep those with >=1 book in Step 1's year; mark each yes/no/unsure.\n"
            f"  Step 4 — Birthplace and population for everyone marked yes or unsure.\n\n"
            f"Write 'unknown' for any fact you cannot recall, but still include "
            f"the author. {cls._UNCERTAINTY_TAIL}")

    @classmethod
    def _materialization_checkpoint(cls, book_title: str) -> str:
        return f"""Task (Strategy 8 — Materialization as Checkpointing)
{"=" * 50}
Target book: "{book_title}"
CHECKPOINT 1 (Materialize year, verify before proceeding)
──────────────────────────────────────────────────────────
State the exact publication year of "{book_title}" as a single four-digit number.
Write: "Publication year: [YYYY]"
This year becomes your anchor for all downstream filters.

CHECKPOINT 2 (Filter co-year books)
────────────────────────────────────
Using ONLY the year from CHECKPOINT 1, list all other books published in
that same year (any genre, any author). Name each book.

CHECKPOINT 3 (Extract authors)
────────────────────────────────
From the books listed in CHECKPOINT 2, extract the full name of each author.
Keep only unique author names.

CHECKPOINT 4 (Apply prize filter)
───────────────────────────────────
From the authors in CHECKPOINT 3, keep ONLY those who have EVER won the
Booker or Pulitzer Prize.
Write each retained author's name. If none qualify, write "No matching authors."

CHECKPOINT 5 (Geography)
─────────────────────────
For each author from CHECKPOINT 4, state:
  (a) Birthplace city
  (b) Population of that city
Write one author per line in the format:
  [Author Name] | [Birthplace] | [Population]
─────────────────────────────────────────────
RULE: Each checkpoint must complete before the next begins.
{cls._UNCERTAINTY_TAIL}
Do not skip checkpoints. Do not proceed without materializing each result.
"""

    @classmethod
    def _temporal_scope_disambiguation(cls, book_title: str) -> str:
        return f"""Task (Strategy 9 — Temporal-Scope Disambiguation)
{"=" * 50}
Target book: "{book_title}"
SCOPE DECLARATION (Do this BEFORE reasoning)
═════════════════════════════════════════════
  CONDITION A (Year-Specific Scope)
  ─────────────────────────────────
    "authored at least one book published in the same calendar year as {book_title}"
    → ONLY books published in THAT YEAR qualify
    → The books in this scope do NOT have to be prize-winning

  CONDITION B (Career-Wide Scope)
  ─────────────────────────────────
    "has EVER won the Booker or Pulitzer Prize"
    → The prize can be from ANY year in the author's career
    → Prize year ≠ {book_title} year — they may differ completely

STEP 1: State the publication year of "{book_title}"
────────────────────────────────────────────────────
Write: "Publication year: [YYYY]", then `confident` or `uncertain`.

STEP 2: Find all books published in the year from STEP 1
─────────────────────────────────────────────────────────
In Condition A scope: List books published in that year.

STEP 3: Extract authors from STEP 2
──────────────────────────────────────
In Condition A scope: Identify authors of the books from STEP 2.

STEP 4: Apply Condition B filter (career-wide scope)
────────────────────────────────────────────────────
From the authors in STEP 3, keep ONLY those who have EVER won the Booker or
Pulitzer Prize.

STEP 5: Birthplace and population
──────────────────────────────────
For each author from STEP 4, state:
  (a) Birthplace city
  (b) Population of that city
Format: [Author Name] | [Birthplace] | [Population]
─────────────────────────────────────────────────────
VERIFICATION: Confirm you have kept Condition A and B separate throughout.
{cls._UNCERTAINTY_TAIL}
"""

    @classmethod
    def _factorization(cls, book_title: str) -> str:
        return f"""Task (Strategy 12 — Factorization into atomic units)
{"=" * 50}
Target book: "{book_title}"
  FACTOR F1  (book -> year)
    What is the publication year Y of "{book_title}"?
  FACTOR F2  (year -> co-year book set)
    Given Y, which books (any author) were published in year Y?
  FACTOR F3  (career-wide prize membership)
    Which authors have EVER won the Booker or Pulitzer Prize?
    [name at least {RECALL_MIN_CANDIDATES}]
  FACTOR F4  (author -> birthplace)
    For a given author A, what is A's birthplace?
  FACTOR F5  (place -> population)
    For a given place P, what is the population of P?

SOLVE EACH FACTOR INDEPENDENTLY FIRST
──────────────────────────────────────
  F1 = [YYYY]
  F2 = [list of co-year books and their authors]
  F3 = [list of Booker/Pulitzer-winning authors]

COMBINATION STEP (explicit logical conjunction)
──────────────────────────────────────────────
Final author set = authors(F2) ∩ F3
For each author in the final set, resolve F4 then F5 and report:
  [Author Name] | [Birthplace] | [Population]
If the final set is empty, write: No matching authors found.
{cls._UNCERTAINTY_TAIL}
"""

    @classmethod
    def _probabilistic_chaining(cls, book_title: str) -> str:
        return f"""Task (Strategy 13 — Probability-weighted chain-rule composition)
{"=" * 50}
Target book: "{book_title}"
UNITS
─────
  U1: publication_year("{book_title}") = Y
  U2: books_published_in(Y) -> co-year book set
  U3: author_of(book) for each co-year book -> candidate author set
  U4: has_ever_won_booker_or_pulitzer(author) for each candidate
  U5: birthplace(author) for each author surviving U4
  U6: population(birthplace) for each place from U5

FOR EACH UNIT, REPORT
──────────────────────
  value_i : your best-estimate answer for that unit
  conf_i  : your genuine confidence in value_i, a number in [0,1]

CHAIN-RULE COMPOSITION
───────────────────────
For each candidate author, compute the running joint confidence.
INCLUSION RULE: only retain authors with full-chain joint confidence >= 0.5.

OUTPUT
──────
  [Author Name] | [Birthplace] | [Population] | [Joint Confidence]
If no author reaches the threshold, write: No matching authors found.
"""

    @classmethod
    def _backward_goal_decomposition(cls, book_title: str) -> str:
        return f"""Task (Strategy 14 — Backward goal decomposition)
{"=" * 50}
Target book: "{book_title}"
Start from the required OUTPUT and work BACKWARD until reaching a base case.

  GOAL: [Author Name] | [Birthplace] | [Population] for every qualifying author.
  GOAL requires -> Population(Birthplace(Author))
  Sub-goal 1: need Birthplace(Author) first.
  Sub-goal 2: need identity of every qualifying Author.
  Sub-goal 3: author qualifies iff (a) EVER won Booker/Pulitzer AND
              (b) authored >=1 book in the same year as "{book_title}".
  Sub-goal 4 (BASE CASE): publication year of "{book_title}".

EXECUTE FORWARD
────────────────
Resolve in reverse order: publication year -> co-year books -> prize-winner
list -> intersected author set -> birthplace -> population.

  [Author Name] | [Birthplace] | [Population]
If no author qualifies, write: No matching authors found.
{cls._UNCERTAINTY_TAIL}
"""

    @classmethod
    def _dual_path_consistency(cls, book_title: str) -> str:
        return f"""Task (Strategy 15 — Dual-path consistency check)
{"=" * 50}
Target book: "{book_title}"
PATH A — year-first
────────────────────
  A1. Publication year of "{book_title}".
  A2. Books published in that year, and their authors.
  A3. Filter A2's authors to those who EVER won the Booker or Pulitzer Prize.
  A4. Birthplace + population for each author surviving A3.

PATH B — prize-first
──────────────────────
  B1. All authors who EVER won the Booker or Pulitzer Prize (name at least {RECALL_MIN_CANDIDATES}).
  B2. For each author in B1, their full bibliography.
  B3. Filter B1 to authors with >=1 book in the same year as "{book_title}".
  B4. Birthplace + population for each author surviving B3.

CONSISTENCY CHECK
───────────────────
Compare A4 vs B4 author sets. They must be IDENTICAL. If they differ, state
the discrepancy, re-examine both paths, and resolve to the best-supported set.

OUTPUT
──────
[Author Name] | [Birthplace] | [Population]
If no author qualifies under both paths, write: No matching authors found.
{cls._UNCERTAINTY_TAIL}
"""


class PromptBuilder:
    """Full builder (the HF script's builder for non-compact families). No trailing 'ANSWER:' cue and no prefill:
    the model writes 'ANSWER:' itself, as the output-format block instructs."""
    _OUTPUT_FORMAT = """Output format
-------------
Start the final section with the line:
  ANSWER:
On the very next line write:
  Publication year: [YYYY]
Then list one line per matching author in this exact format:
  [Author Name] | [Birthplace] | [Population]
Where:
  Author Name    = the author's full name in English
  Birthplace     = the city or place of birth in English (write "unknown" if not recalled)
  Population     = the population of that place as a number (write "unknown" if not recalled)
If no authors match the criteria, write:
  No matching authors found.
Include an author even when their birthplace or population is unknown — write
unknown in that field. Never invent a specific figure."""
    _ROLE_GENERAL = (
        "You are a literary history research assistant with broad knowledge of "
        "Booker Prize and Pulitzer Prize records, author bibliographies, and "
        "international biographical reference data.")

    @classmethod
    def build(cls, query_text: str, mode: str, query_id: Any, strategy: str = STRATEGY_FLAT_CONJUNCTIVE,
              evidence_block: Optional[str] = None, fewshot_block: Optional[str] = None,
              recall_scaffold: bool = True) -> str:
        if mode != QUERY_MODE_NLQ_GENERAL:
            raise ValueError(f"Unknown mode '{mode}'")
        task = StrategyBuilder.build(query_text, strategy, evidence_block, fewshot_block, recall_scaffold)
        return "\n\n".join([cls._ROLE_GENERAL, f"[Query ID: {query_id}]  [Strategy: {strategy}]", task, cls._OUTPUT_FORMAT])


def build_prompt(query_text: str, mode: str, query_id: Any, strategy: str, evidence_block: Optional[str],
                 fewshot_block: Optional[str], recall_scaffold: bool) -> Tuple[str, str]:
    if strategy == STRATEGY_SELF_ASK:
        return SelfAskPromptBuilder.build(query_text, mode, query_id), "self_ask"
    return PromptBuilder.build(query_text, mode, query_id, strategy, evidence_block, fewshot_block, recall_scaffold), "v5_strategy_preamble"


# ---------------------------------------------------------------------------
# Response parser (FIX_55 / FIX_56 diagnostics)
# ---------------------------------------------------------------------------
_HEADER_SKIP_RE = re.compile(
    r"^(publication\s+year|release\s+year|year|no matching|answer|note|step|"
    r"candidates?|prize\s+winners?|co-?year(\s+check)?|required\s+working|working|"
    r"refusal\s+gate|follow up|intermediate answer|so the final|end of response)", re.IGNORECASE)
_COYEAR_CHECK_RE = re.compile(r"[-–—]\s*(yes|no|unsure)\s*$", re.IGNORECASE)


def parse_response(text: str) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    diag: Dict[str, Any] = {
        "answer_cue_found": False, "explicit_no_match": False, "end_marker_found": False,
        "scaffold_candidates_found": False, "scaffold_coyear_found": False, "scaffold_candidate_count": 0,
        "scaffold_coyear_yes_count": 0, "scaffold_coyear_unsure_count": 0, "scaffold_coyear_no_count": 0,
        "refusal_contradicts_scaffold": False, "body_char_len": 0, "body_line_count": 0, "pipe_lines_seen": 0,
        "lines_skipped_header": 0, "lines_skipped_interactive": 0, "lines_skipped_coyear": 0,
        "year_regex_matched": False, "outcome": ""}
    if not text or text.startswith("ERROR"):
        diag["outcome"] = "error_or_empty_input"
        return ([{"PublicationYear": "", "Author": "ERROR", "Birthplace": "", "PopCount": "", "JointConfidence": ""}], diag)
    diag["scaffold_candidates_found"] = bool(re.search(r"^\s*(CANDIDATES|PRIZE\s+WINNERS)\s*:", text, re.IGNORECASE | re.MULTILINE))
    diag["scaffold_coyear_found"] = bool(re.search(r"^\s*CO-?YEAR\s+CHECK\s*:", text, re.IGNORECASE | re.MULTILINE))
    if diag["scaffold_candidates_found"]:
        seg = re.split(r"^\s*(CANDIDATES|PRIZE\s+WINNERS)\s*:", text, flags=re.IGNORECASE | re.MULTILINE)[-1]
        seg = re.split(r"^\s*(CO-?YEAR|ANSWER)\b", seg, flags=re.IGNORECASE | re.MULTILINE)[0]
        diag["scaffold_candidate_count"] = sum(1 for ln in seg.splitlines() if ln.strip() and not _HEADER_SKIP_RE.match(ln.strip()))
    if diag["scaffold_coyear_found"]:
        cseg = re.split(r"^\s*CO-?YEAR\s+CHECK\s*:", text, flags=re.IGNORECASE | re.MULTILINE)[-1]
        cseg = re.split(r"^\s*(REFUSAL\s+GATE|ANSWER)\b", cseg, flags=re.IGNORECASE | re.MULTILINE)[0]
        for ln in cseg.splitlines():
            m = _COYEAR_CHECK_RE.search(ln.strip())
            if not m:
                continue
            verdict = m.group(1).lower()
            diag[f"scaffold_coyear_{verdict}_count"] += 1
    answer_match = re.search(r"ANSWER\s*:", text, re.IGNORECASE)
    diag["answer_cue_found"] = bool(answer_match)
    if answer_match:
        for m in re.finditer(r"ANSWER\s*:", text, re.IGNORECASE):
            answer_match = m
    body = text[answer_match.end():] if answer_match else text.strip()
    end_marker = re.search(r"END\s+OF\s+RESPONSE", body, re.IGNORECASE)
    diag["end_marker_found"] = bool(end_marker)
    if end_marker:
        body = body[:end_marker.start()]
    body = _strip_interactive_patterns(body).strip() if answer_match else body
    diag["explicit_no_match"] = bool(re.search(r"no matching authors", body, re.IGNORECASE))
    diag["body_char_len"], diag["body_line_count"] = len(body), len(body.splitlines())
    year = ""
    ym = re.search(r"(?:publication\s+year|release\s+year|year)\s*[:\-]?\s*((?:19|20)\d{2})", body, re.IGNORECASE)
    if ym:
        year, diag["year_regex_matched"] = ym.group(1), True
    rows = []
    for line in body.splitlines():
        line = re.sub(r"\*+", "", line).strip()
        if not line:
            continue
        if _is_interactive_pattern(line):
            diag["lines_skipped_interactive"] += 1
            break
        if _HEADER_SKIP_RE.match(line):
            diag["lines_skipped_header"] += 1
            continue
        if "|" not in line and _COYEAR_CHECK_RE.search(line):
            diag["lines_skipped_coyear"] += 1
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2 and parts[0]:
            diag["pipe_lines_seen"] += 1
            pop_raw = parts[2] if len(parts) > 2 else ""
            pop = "unknown" if pop_raw.lower() == "unknown" else pop_to_str(pop_raw)
            rows.append({"PublicationYear": year, "Author": parts[0], "Birthplace": parts[1] if len(parts) > 1 else "",
                         "PopCount": pop, "JointConfidence": parts[3] if len(parts) > 3 else ""})
    if rows:
        diag["outcome"] = "rows_parsed"
        return rows, diag
    diag["refusal_contradicts_scaffold"] = bool(
        (diag["explicit_no_match"] or diag["body_char_len"] == 0)
        and (diag["scaffold_coyear_yes_count"] > 0 or diag["scaffold_coyear_unsure_count"] > 0))
    if diag["explicit_no_match"]:
        diag["outcome"], sentinel = "explicit_no_match", "NO RESULT (explicit)"
    elif diag["body_char_len"] == 0:
        diag["outcome"], sentinel = "empty_body", "NO RESULT (empty)"
    else:
        diag["outcome"], sentinel = "unparseable_body", "NO RESULT (unparseable)"
    return ([{"PublicationYear": year, "Author": sentinel, "Birthplace": "", "PopCount": "", "JointConfidence": ""}], diag)


def derive_parse_fields(raw: str, strategy: str, pub_year: str, query_id: Any,
                        gt_store: Optional[GroundTruthStore]) -> Dict[str, Any]:
    """Every parse-derived field of a record (shared by the live run and --reparse). `raw` is never altered."""
    parse_text = enforce_single_shot(raw) if strategy == STRATEGY_SELF_ASK else raw
    parsed, diag = parse_response(parse_text)
    deg, why = detect_degenerate(raw)
    year_parsed = parsed[0].get("PublicationYear", "") if parsed else ""
    agrees = bool(pub_year and year_parsed and str(pub_year) == str(year_parsed))
    gt_records = gt_store.by_sno.get(str(query_id), []) if gt_store else []
    gt_score = score_authors_against_ground_truth(parsed, gt_records)
    return {"Parsed_Rows": parsed, "parse_diagnostics": diag, "Parse_Text_Altered": parse_text != raw,
            "Degenerate_Flag": deg, "Degenerate_Reason": why, "PublicationYear_Parsed": year_parsed,
            "Author_Parsed": "; ".join(r["Author"] for r in parsed),
            "Birthplace_Parsed": "; ".join(r["Birthplace"] for r in parsed),
            "PopCount_Parsed": "; ".join(r["PopCount"] for r in parsed),
            "JointConfidence_Parsed": "; ".join(r.get("JointConfidence", "") for r in parsed),
            "year_agrees_with_source": agrees, "gt_score": gt_score,
            "authors_correct_despite_wrong_year": bool(gt_score["authors_correct_count"] > 0 and not agrees)}


LATENT_NONE: Dict[str, Any] = {k: None for k in (
    "attention_mean", "attention_min", "attention_max", "attention_std", "attention_entropy", "hidden_state_norm_mean",
    "hidden_state_norm_min", "hidden_state_norm_max", "hidden_state_norm_std", "token_prob_mean", "token_prob_min",
    "token_prob_max", "token_prob_p10", "token_prob_p90", "confidence_score", "perplexity_estimate")}


# ---------------------------------------------------------------------------
# CLAUDE CODE CLIENT - streamed headless `claude -p`; every call leaves raw evidence on disk
# ---------------------------------------------------------------------------
_SYSTEM_FLAG_COMPLAINT_RE = re.compile(r"system[- ]prompt|unknown (?:option|argument)|unrecognized (?:option|argument)", re.I)
_BILLING_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX")


def _completion_tokens(usage: Any) -> float:
    if not isinstance(usage, dict):
        return float("nan")
    for k in ("output_tokens", "completion_tokens", "outputTokens", "generation_token_count"):
        if usage.get(k) is not None:
            try:
                return float(usage[k])
            except (TypeError, ValueError):
                pass
    return float("nan")


class ClaudeCodeClient:
    def __init__(self, model_alias: str = DEFAULT_MODEL_ALIAS, max_new_tokens: int = GENERATION_MAX_NEW_TOKENS,
                 system_mode: str = "neutral", claude_bin: str = "claude", timeout: int = REQUEST_TIMEOUT,
                 max_thinking_tokens: Optional[int] = None, thinking: str = "default",
                 tools_mode: str = "none", max_turns: Optional[int] = None):
        if model_alias not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model alias '{model_alias}'")
        found = shutil.which(claude_bin)
        if not found:
            raise RuntimeError(f"'{claude_bin}' (Claude Code CLI) not found on PATH")
        self.bin, self.timeout = found, timeout
        self.model_alias, self.model_id, self.family = model_alias, MODEL_REGISTRY[model_alias], FAMILY
        self.max_tokens = int(max_new_tokens)
        self.max_thinking_tokens = max_thinking_tokens
        self.thinking = thinking                     # default | off (MAX_THINKING_TOKENS=0) | low..max (--effort)
        if thinking == "off":
            self.max_thinking_tokens = 0
        self.tools_mode = tools_mode                 # none | web (WebSearch + WebFetch)
        self.max_turns = int(max_turns) if max_turns else (WEB_DEFAULT_MAX_TURNS if tools_mode == "web" else 1)
        self.system_mode_requested = system_mode
        self.system_text: Optional[str] = NEUTRAL_SYSTEM if system_mode == "neutral" else None
        self.system_native = True
        self.dropped: List[str] = []
        self.raw_dir: Optional[Path] = None
        self.call_label = "call"
        self.workdir = tempfile.mkdtemp(prefix="cc_book_")       # empty cwd: no CLAUDE.md / repo context is loaded
        atexit.register(shutil.rmtree, self.workdir, True)
        logger.info(f"ClaudeCodeClient ready  alias={model_alias}  model={self.model_id}  bin={self.bin}  "
                    f"max_output_tokens={self.max_tokens}  max_thinking_tokens={self.max_thinking_tokens}  thinking={thinking}  "
                    f"tools_mode={tools_mode}  max_turns={self.max_turns}  system_mode={system_mode}")

    @property
    def system_mode(self) -> str:
        return "default" if not self.system_text else ("native" if self.system_native else "prepended")

    def _stem(self, attempt: int) -> Path:
        d = self.raw_dir or Path(self.workdir)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.call_label}_try{attempt}"

    def _run_streaming(self, cmd: List[str], user_text: str, system_text: Optional[str],
                       env: Dict[str, str], stem: Path) -> Dict[str, Any]:
        """Run one call, writing everything to disk AS IT ARRIVES (a failed / killed call still leaves evidence)."""
        p_prompt, p_in = Path(f"{stem}_prompt.txt"), Path(f"{stem}_stdin.tmp")
        p_jsonl, p_text, p_err = Path(f"{stem}_events.jsonl"), Path(f"{stem}_text.txt"), Path(f"{stem}_stderr.txt")
        p_prompt.write_text((f"[SYSTEM]\n{system_text}\n\n" if system_text else "") + "[USER]\n" + user_text, encoding="utf-8")
        p_in.write_text(user_text, encoding="utf-8")
        text_parts: List[str] = []
        assistant_text, think_chars = "", 0
        tool_uses: List[str] = []
        result_evt: Optional[Dict[str, Any]] = None
        timed_out = {"v": False}
        t0 = time.perf_counter()
        last_log = t0
        with open(p_in, "rb") as fin, open(p_err, "wb") as ferr, \
                open(p_jsonl, "w", encoding="utf-8") as fj, open(p_text, "w", encoding="utf-8") as ft:
            proc = subprocess.Popen(cmd, stdin=fin, stdout=subprocess.PIPE, stderr=ferr, cwd=self.workdir, env=env,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1)

            def _kill():
                timed_out["v"] = True
                proc.kill()

            watchdog = threading.Timer(self.timeout, _kill)
            watchdog.start()
            try:
                for line in proc.stdout:
                    fj.write(line); fj.flush()
                    try:
                        evt = json.loads(line)
                    except ValueError:
                        continue
                    et = evt.get("type")
                    if et == "stream_event":
                        ev = evt.get("event") or {}
                        if ev.get("type") == "content_block_delta":
                            d = ev.get("delta") or {}
                            if d.get("type") == "text_delta":
                                s = d.get("text", "")
                                text_parts.append(s); ft.write(s); ft.flush()
                            elif d.get("type") == "thinking_delta":
                                think_chars += len(d.get("thinking", ""))
                    elif et == "assistant":
                        for b in (evt.get("message") or {}).get("content") or []:
                            if isinstance(b, dict) and b.get("type") == "text":
                                assistant_text += b.get("text", "")
                            elif isinstance(b, dict) and b.get("type") in ("tool_use", "server_tool_use"):
                                tool_uses.append(str(b.get("name")))
                    elif et == "result":
                        result_evt = evt
                    now = time.perf_counter()
                    if now - last_log >= PROGRESS_LOG_EVERY_SEC:
                        last_log = now
                        logger.info(f"  [{self.call_label}] streaming {now - t0:.0f}s: "
                                    f"text={sum(len(x) for x in text_parts)} chars  thinking={think_chars} chars")
                proc.wait()
            finally:
                watchdog.cancel()
        text = "".join(text_parts) or assistant_text
        if self.tools_mode != "none" and result_evt and not result_evt.get("is_error") and result_evt.get("result"):
            text = str(result_evt.get("result"))         # multi-turn: keep only the FINAL answer (stream has interim text)
            p_text.write_text(text, encoding="utf-8")
        if not text and result_evt and not result_evt.get("is_error"):
            text = str(result_evt.get("result") or "")
        if text and not text_parts:
            p_text.write_text(text, encoding="utf-8")
        stderr_text = p_err.read_text(encoding="utf-8", errors="replace")
        blob = stderr_text + " " + (str(result_evt.get("result", "")) if result_evt else "")
        try:
            p_in.unlink()
        except OSError:
            pass
        return {"text": text, "thinking_chars": think_chars, "tool_uses": tool_uses, "result_evt": result_evt, "returncode": proc.returncode,
                "stderr": stderr_text, "timed_out": timed_out["v"], "latency": time.perf_counter() - t0,
                "hit_cap": bool(re.search(r"output token maximum", blob, re.I)),
                "files": [str(x) for x in (p_prompt, p_jsonl, p_text, p_err)]}

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        mt = int(max_tokens or self.max_tokens)
        sysmsg = self.system_text
        wait, last_exc = RETRY_BASE_SEC, None
        for attempt in range(1, MAX_RETRIES + 1):
            native = bool(sysmsg and self.system_native)
            user_text = (sysmsg + "\n\n" + prompt) if (sysmsg and not self.system_native) else prompt
            cmd = [self.bin, "-p", "--model", self.model_id, "--output-format", "stream-json", "--verbose",
                   "--include-partial-messages"]
            if self.tools_mode == "web":
                cmd += ["--tools", "WebSearch,WebFetch", "--allowedTools", "WebSearch", "WebFetch"]
            else:
                cmd += ["--tools", ""]
            cmd += ["--max-turns", str(self.max_turns)]
            if self.thinking not in ("default", "off"):
                cmd += ["--effort", self.thinking]
            if native:
                cmd += ["--system-prompt", sysmsg]
            env = dict(os.environ, CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(mt))
            if self.max_thinking_tokens is not None:
                env["MAX_THINKING_TOKENS"] = str(self.max_thinking_tokens)
            r = self._run_streaming(cmd, user_text, sysmsg if native else None, env, self._stem(attempt))
            text = r["text"].strip()
            evt = r["result_evt"] or {}
            usage = evt.get("usage") if isinstance(evt.get("usage"), dict) else None
            base = {"raw_response": text, "response": text, "latency_sec": round(r["latency"], 3), "usage": usage,
                    "cost_usd": evt.get("total_cost_usd"), "max_tokens_sent": mt, "system_mode": self.system_mode,
                    "thinking_chars": r["thinking_chars"], "text_chars": len(text), "raw_files": r["files"],
                    "tool_calls": r["tool_uses"], "n_tool_calls": len(r["tool_uses"]), "thinking": self.thinking,
                    "tools_mode": self.tools_mode, "max_turns": self.max_turns}
            if r["hit_cap"]:       # deterministic: do NOT retry; keep the partial text and flag it
                logger.warning(f"  [{self.call_label}] hit the {mt}-token output cap after {r['latency']:.0f}s "
                               f"(text={len(text)} chars, thinking={r['thinking_chars']} chars). Not retried. Raw: {r['files'][1]}")
                return dict(base, status=ST_TRUNC_LENGTH, finish_reason="output_token_cap")
            ok = r["returncode"] == 0 and bool(evt) and not evt.get("is_error") and not r["timed_out"]
            if ok:
                finish = evt.get("stop_reason") or evt.get("subtype")
                out_tok = _completion_tokens(usage)
                status = ST_EMPTY if not text else ST_OK
                if text and ((out_tok == out_tok and out_tok >= 0.98 * mt) or str(finish or "").lower() in ("max_tokens", "length")):
                    status = ST_TRUNC_LENGTH
                return dict(base, status=status, finish_reason=None if finish is None else str(finish))
            msg = ("TIMEOUT " if r["timed_out"] else "") + (str(evt.get("result", "")) if evt else "") + " " + r["stderr"][:400]
            if native and _SYSTEM_FLAG_COMPLAINT_RE.search(msg):
                self.system_native = False
                self.dropped.append("system_prompt_flag->prepended_to_user_turn")
                logger.warning(f"--system-prompt rejected ({msg[:120]!r}) - prepending the system text to the user turn")
                continue
            logger.warning(f"claude exit={r['returncode']} attempt {attempt}/{MAX_RETRIES}: {msg[:200]!r} "
                           f"(raw: {r['files'][1]}) - retry in {wait:.0f}s")
            time.sleep(wait); wait *= 2.0; last_exc = RuntimeError(msg[:300])
        raise RuntimeError(f"claude -p failed after {MAX_RETRIES} attempts. Last error: {last_exc}. "
                           f"Raw logs: {self.raw_dir or self.workdir}") from last_exc

    def validate_access(self) -> bool:
        logger.info(f"Validating Claude Code access  alias={self.model_alias}  model={self.model_id} ...")
        self.call_label = "validate"
        res = self.generate("Respond with only the word: OK", 256)
        logger.info(f"Validation OK - response: '{res['response'][:60]}'")
        return True


# ---------------------------------------------------------------------------
# RestartManager (error records are RE-RUN; a truncated record is re-run when the current cap is larger)
# ---------------------------------------------------------------------------
class RestartManager:
    def __init__(self, output_file: Path, max_tokens: int = 0):
        self.output_file = output_file
        self.max_tokens = max_tokens
        self.processed_ids: Set[str] = set()
        self.existing_results: List[Dict[str, Any]] = []

    def load_existing_results(self) -> bool:
        if not self.output_file.exists():
            logger.info("No existing results file found. Starting fresh.")
            return False
        try:
            data = json.loads(self.output_file.read_text(encoding="utf-8"))
            self.existing_results = data.get("results", [])
            for r in self.existing_results:
                qid, answer = r.get("Query_ID"), str(r.get("Generated_Response", "")).strip()
                if qid is None or not answer or answer.startswith("ERROR") or answer.startswith("DEGENERATE_OUTPUT"):
                    continue
                if (r.get("generation_status") in (ST_TRUNC_LENGTH, ST_TRUNC_REASONING)
                        and int(r.get("max_tokens_sent") or 0) < self.max_tokens):
                    logger.info(f"Query_ID={qid}: truncated under a smaller output cap - it will be re-run.")
                    continue
                self.processed_ids.add(str(qid))
            logger.info(f"Loaded existing results: {len(self.existing_results)} entries, {len(self.processed_ids)} reusable.")
            return len(self.processed_ids) > 0
        except Exception as e:
            logger.warning(f"Could not load existing results ({e}). Starting fresh.")
            try:
                shutil.copy2(self.output_file, self.output_file.with_suffix(".json.bak"))
            except Exception:
                pass
            self.existing_results, self.processed_ids = [], set()
            return False


# ---------------------------------------------------------------------------
# BookAuthorAnalyser
# ---------------------------------------------------------------------------
class BookAuthorAnalyser:
    def __init__(self, strategy: str, client: ClaudeCodeClient, no_resume: bool = False,
                 rag_tier: str = RAG_TIER_NO_EVIDENCE, ref_store: Optional[ReferenceStore] = None,
                 gt_store: Optional[GroundTruthStore] = None, fewshot_store: Optional[FewShotExemplarStore] = None,
                 recall_scaffold: bool = True, rag_max_rows: int = RAG_EVIDENCE_MAX_ROWS, store_prompts: bool = True):
        self.mode = QUERY_MODE_NLQ_GENERAL
        if strategy not in INDIVIDUAL_STRATEGIES:
            raise ValueError(f"Unsupported strategy '{strategy}'.")
        self.strategy, self.no_resume, self.rag_tier = strategy, no_resume, rag_tier
        self.client, self.ref_store, self.gt_store, self.fewshot_store = client, ref_store, gt_store, fewshot_store
        self.recall_scaffold, self.rag_max_rows, self.store_prompts = recall_scaffold, rag_max_rows, store_prompts
        self.model_alias, self.model_id, self.family = client.model_alias, client.model_id, client.family
        self.model_slug = model_name_to_slug(self.model_id)
        extra = f"_{RAG_TIER_SUFFIX[rag_tier]}" if rag_tier != RAG_TIER_NO_EVIDENCE else ""
        sysdef = "_SYSDEF" if client.system_mode_requested == "default" else ""
        sysdef += {"default": "", "off": "_THOFF"}.get(client.thinking, f"_EFF{client.thinking.upper()}")
        sysdef += "_WEB" if client.tools_mode == "web" else ""
        flags = ("_NOSCAF" if not recall_scaffold else "") + ("_NOFEW" if fewshot_store is None else "")
        self.suffix = f"{MODE_SUFFIX[self.mode]}_{STRATEGY_SUFFIX[strategy]}{extra}{flags}{sysdef}"
        self.output_dir = OUTPUT_ROOT / f"{self.model_slug}_{self.suffix}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / f"{self.model_slug}_{self.suffix}.json"
        self.response_txt = self.output_dir / f"{self.model_slug}_{self.suffix}_responses.txt"
        self.stopped_early = False
        if no_resume and self.output_file.exists():
            logger.info("--no-resume: deleting existing output and reprocessing from scratch.")
            self.output_file.unlink()
        self.restart_manager = RestartManager(self.output_file, max_tokens=client.max_tokens)
        if self.restart_manager.load_existing_results() and not no_resume:
            logger.info(f"RESUME MODE — {len(self.restart_manager.processed_ids)} records already done.")

    def load_input(self) -> pd.DataFrame:
        path = _resolve_path(SOURCE_FILE)
        logger.info(f"Loading source: {path}")
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        missing = {"S.No", "BookTitle", "PublicationYear"} - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)}. Found: {list(df.columns)}")
        df = df.fillna("")
        df["Query_ID"] = df["S.No"]
        df["Query_ID_STR"] = df["Query_ID"].astype(str).str.strip()
        df["PublicationYear_Clean"] = df["PublicationYear"].apply(clean_year)
        logger.info(f"Loaded {len(df)} rows from source.")
        return df

    @staticmethod
    def _result_sort_key(rec: Dict[str, Any]):
        s = str(rec.get("Query_ID", "")).strip()
        return (0, int(s), "") if s.lstrip("-").isdigit() else (1, 0, s)

    def _save(self, results: List[Dict[str, Any]]) -> None:
        errors = sum(1 for r in results if str(r.get("Generated_Response", "")).startswith("ERROR"))
        c = self.client
        output = {
            "metadata": {
                "script_version": SCRIPT_VERSION, "domain": "books", "model_alias": self.model_alias,
                "model_name": self.model_id, "model_family": self.family,
                "model_size": MODEL_SIZE_LABEL.get(self.model_alias, "?"), "inference_backend": "claude_code_cli_cloud_session",
                "query_mode": self.mode, "strategy": self.strategy, "strategy_suffix": STRATEGY_SUFFIX[self.strategy],
                "rag_tier": self.rag_tier, "rag_tier_description": RAG_TIER_DESCRIPTION.get(self.rag_tier, ""),
                "rag_enabled": self.rag_tier != RAG_TIER_NO_EVIDENCE, "rag_max_rows": self.rag_max_rows or "all records",
                "rag_note": "RAG tiers apply to S1 flat_conjunctive only (HF behaviour)",
                "output_suffix": self.suffix, "source_file": SOURCE_FILE, "reference_file": REFERENCE_FILE,
                "ground_truth_file": GROUND_TRUTH_FILE, "timestamp": datetime.now().isoformat(),
                "total_queries": len(results), "successful_records": len(results) - errors, "error_records": errors,
                "max_output_tokens_requested": c.max_tokens, "max_thinking_tokens_env": c.max_thinking_tokens,
                "params_adapted": c.dropped, "request_timeout_s": c.timeout,
                "system_mode_requested": c.system_mode_requested, "system_mode": c.system_mode, "system_text": c.system_text,
                "recall_scaffold_enabled": self.recall_scaffold, "recall_min_candidates": RECALL_MIN_CANDIDATES,
                "scaffold_section_headers": ["PRIZE WINNERS", "CO-YEAR CHECK"],
                "scaffold_note": "the scaffold is part of S1 (flat_conjunctive) without evidence only; other strategies carry their own working steps",
                "fewshot_exemplar_enabled": self.fewshot_store is not None,
                "assistant_prefill": "not available through `claude -p`; the ANSWER: cue comes from the prompt's output-format block",
                "sampling_note": "temperature / top_p / top_k are not controllable through `claude -p`; CLI defaults apply",
                "claude_code_call": "claude -p --model <id> --output-format stream-json --verbose --include-partial-messages "
                                    "--tools '' --max-turns 1 [--system-prompt <neutral text>]; cwd = empty temp dir; "
                                    "output cap via CLAUDE_CODE_MAX_OUTPUT_TOKENS; raw events in <cell dir>/raw_calls",
                "prompt_builder": "self_ask" if self.strategy == STRATEGY_SELF_ASK else "v5_strategy_preamble",
                "response_text_policy": "Generated_Response is the model text as returned; never altered",
                "prize_criteria": ["Booker Prize", "Pulitzer Prize"],
                "population_handling": "GT values parsed with float() first; model answers with unit-aware parser",
                "latent_note": "Claude Code returns no per-token data; latent fields are null",
                "strategies_available": {STRATEGY_CODE[s]: s for s in INDIVIDUAL_STRATEGIES},
                "rag_tiers_available": RAG_TIER_DESCRIPTION,
            },
            "results": sorted(results, key=self._result_sort_key),
        }
        self.output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        parts = [f"===== Query_ID {r.get('Query_ID')} | {r.get('Book_Title', '')} =====\n{r.get('Generated_Response', '')}"
                 for r in sorted(results, key=self._result_sort_key)]
        try:
            self.response_txt.write_text("\n\n".join(parts), encoding="utf-8")
        except OSError as e:
            logger.warning(f"Could not write {self.response_txt}: {e}")
        logger.info(f"Saved {len(results)} records -> {self.output_file}")

    def _base_fields(self, q_id, book_title, book_id, pub_year, evidence_injected) -> Dict[str, Any]:
        return {"Query_ID": q_id, "Book_Title": book_title, "Book_ID": book_id, "Source_PublicationYear": pub_year,
                "mode": self.mode, "strategy": self.strategy, "rag_tier": self.rag_tier,
                "rag_evidence_injected": evidence_injected, "system_mode_requested": self.client.system_mode_requested,
                "model_alias": self.model_alias, "model_id": self.model_id, "model_family": self.family}

    def process(self, df: pd.DataFrame, max_queries: Optional[int] = None) -> List[Dict[str, Any]]:
        total_in = len(df)
        pending = df[~df["Query_ID_STR"].isin(self.restart_manager.processed_ids)]
        if max_queries:
            pending = pending.head(max_queries)
        if pending.empty:
            logger.info("All records already processed. Nothing to do.")
            return self.restart_manager.existing_results
        results = [r for r in self.restart_manager.existing_results
                   if not str(r.get("Generated_Response", "")).startswith("ERROR")
                   and str(r.get("Query_ID")) in self.restart_manager.processed_ids]
        skipped_count, new_count, bad_streak = len(self.restart_manager.processed_ids), 0, 0
        self.client.raw_dir = self.output_dir / "raw_calls"
        logger.info(f"Processing {len(pending)}/{total_in} queries ({skipped_count} already done)  strategy={self.strategy}  "
                    f"rag_tier={self.rag_tier}  model={self.model_id}  backend=claude_code")
        for idx, (_, row) in enumerate(pending.iterrows()):
            q_id = row["Query_ID"]
            book_title, book_id = str(row["BookTitle"]).strip(), str(row.get("Book_URI", "")).strip()
            pub_year = str(row["PublicationYear_Clean"]).strip()
            if not book_title:
                logger.warning(f"Skipping Query_ID={q_id}: empty BookTitle.")
                continue
            logger.info(f"[{new_count + skipped_count + 1}/{total_in}] Q_ID={q_id} '{book_title}' year={pub_year or '?'}")
            self.client.call_label = f"{self.model_slug}_{self.suffix}_q{q_id}"
            bad = False
            try:
                evidence_block, rag = None, {"rows_available": 0, "rows_injected": 0, "truncated": False}
                if self.strategy == STRATEGY_FLAT_CONJUNCTIVE and self.rag_tier != RAG_TIER_NO_EVIDENCE:
                    if not pub_year:
                        logger.warning(f"Q_ID={q_id}: no source publication year — skipping RAG evidence.")
                    else:
                        evidence_block, rag = build_rag_evidence_block(self.rag_tier, str(q_id), pub_year, book_title,
                                                                       self.ref_store, self.gt_store, self.rag_max_rows)
                        if evidence_block is None:
                            logger.warning(f"Q_ID={q_id}: no evidence found for tier '{self.rag_tier}'.")
                fewshot_block = None
                if self.fewshot_store is not None and self.strategy != STRATEGY_SELF_ASK:
                    fewshot_block = self.fewshot_store.render(exclude_sno=str(q_id))
                scaffold_on = bool(self.recall_scaffold and not evidence_block and self.strategy == STRATEGY_FLAT_CONJUNCTIVE)
                prompt, builder_label = build_prompt(book_title, self.mode, q_id, self.strategy, evidence_block,
                                                     fewshot_block, self.recall_scaffold)
                res = self.client.generate(prompt)
                raw = res["response"]                                  # the model's text, NEVER altered
                fields = derive_parse_fields(raw, self.strategy, pub_year, q_id, self.gt_store)
                record = self._base_fields(q_id, book_title, book_id, pub_year, evidence_block is not None)
                record.update({
                    "system_mode": res["system_mode"], "prompt_builder": builder_label, "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12],
                    "recall_scaffold_active": scaffold_on, "fewshot_exemplar_used": bool(fewshot_block),
                    "fewshot_block_char_len": len(fewshot_block) if fewshot_block else 0,
                    "rag_rows_available": rag["rows_available"], "rag_rows_injected": rag["rows_injected"],
                    "rag_truncated": rag["truncated"], "est_prompt_tokens": len(prompt) // 4,
                    "Generated_Response": raw, "generation_status": res["status"], "finish_reason": res["finish_reason"],
                    "max_tokens_sent": res["max_tokens_sent"], "usage": res["usage"], "cost_usd_reported": res.get("cost_usd"),
                    "thinking_chars": res.get("thinking_chars"), "text_chars": res.get("text_chars"),
                    "thinking": res.get("thinking"), "tools_mode": res.get("tools_mode"), "max_turns": res.get("max_turns"),
                    "tool_calls": res.get("tool_calls"), "n_tool_calls": res.get("n_tool_calls"),
                    "raw_files": res.get("raw_files"), "generation_time_seconds": res["latency_sec"],
                    "processing_timestamp": datetime.now().isoformat()})
                if self.store_prompts:
                    record.update({"prompt_text": prompt, "evidence_block_text": evidence_block or "",
                                   "fewshot_block_text": fewshot_block or ""})
                record.update(evidence_stats(evidence_block))
                record.update(fields)
                record.update(LATENT_NONE)
                results.append(record)
                new_count += 1
                bad = res["status"] in (ST_TRUNC_LENGTH, ST_TRUNC_REASONING)
                pd_ = fields["parse_diagnostics"]
                real = [r for r in fields["Parsed_Rows"] if not is_sentinel_author(r["Author"])]
                logger.info(f"  OK {res['latency_sec']:.2f}s  status={res['status']}  [{pd_.get('outcome')}/rows={pd_.get('pipe_lines_seen', 0)}]  "
                            f"authors={len(real)}  year {fields['PublicationYear_Parsed'] or '?'} vs src {pub_year or '?'}"
                            f"{' OK' if fields['year_agrees_with_source'] else ''}  text_chars={res.get('text_chars')}  "
                            f"thinking_chars={res.get('thinking_chars')}  usage={res['usage']}  cost_usd={res.get('cost_usd')}")
                if pd_.get("refusal_contradicts_scaffold"):
                    logger.warning(f"  Q_ID={q_id}: refusal contradicts own CO-YEAR CHECK — {pd_.get('scaffold_coyear_yes_count')} yes / "
                                   f"{pd_.get('scaffold_coyear_unsure_count')} unsure candidates.")
                if fields["Degenerate_Flag"]:
                    logger.warning(f"  Possible degenerate output ({fields['Degenerate_Reason']}) - flagged only, text kept")
                if bad:
                    logger.warning(f"  TRUNCATED at the output cap (raw files: {res.get('raw_files')})")
            except Exception as e:
                logger.error(f"Query_ID={q_id}: {e}")
                bad = True
                err = self._base_fields(q_id, book_title, book_id, pub_year, False)
                err.update({"system_mode": "n/a", "Generated_Response": f"ERROR: {e}", "generation_status": "error",
                            "generation_time_seconds": 0.0, "processing_timestamp": datetime.now().isoformat(),
                            "Parsed_Rows": [{"PublicationYear": "", "Author": "ERROR", "Birthplace": "", "PopCount": "", "JointConfidence": ""}],
                            "parse_diagnostics": {"outcome": "error_record"}, "PublicationYear_Parsed": "", "Author_Parsed": "ERROR",
                            "Birthplace_Parsed": "", "PopCount_Parsed": "", "JointConfidence_Parsed": "",
                            "year_agrees_with_source": False, "Degenerate_Flag": False, "Degenerate_Reason": "",
                            "Parse_Text_Altered": False, "authors_correct_despite_wrong_year": False,
                            "gt_score": score_authors_against_ground_truth([], [])})
                err.update(LATENT_NONE)
                results.append(err)
            self._save(results)                                          # calls are slow: checkpoint after every query
            bad_streak = bad_streak + 1 if bad else 0
            if bad_streak >= MAX_CONSECUTIVE_BAD:
                self.stopped_early = True
                logger.error(f"{bad_streak} consecutive errors / output-cap hits - stopping this cell to protect the credit. "
                             f"Finished queries are saved; inspect raw_calls/, then rerun (resume skips the finished ones; "
                             f"truncated ones are re-run only with a larger --max_new_tokens).")
                break
        self._save(results)
        self._log_run_summary(results)
        return results

    @staticmethod
    def _log_run_summary(results: List[Dict[str, Any]]) -> None:
        if not results:
            return
        n = len(results)
        out = Counter((r.get("parse_diagnostics") or {}).get("outcome", "?") for r in results)
        gt_scored = [r for r in results if (r.get("gt_score") or {}).get("gt_available")]
        recalls = [r["gt_score"]["author_recall"] for r in gt_scored if r["gt_score"].get("author_recall") is not None]
        precs = [r["gt_score"]["author_precision"] for r in gt_scored if r["gt_score"].get("author_precision") is not None]
        contradicted = sum(1 for r in results if (r.get("parse_diagnostics") or {}).get("refusal_contradicts_scaffold"))
        logger.info("─" * 66)
        logger.info(f"RUN SUMMARY  records={n}  parse outcome={dict(out)}  refusal-contradicts-scaffold={contradicted}  "
                    f"scored vs GT={len(gt_scored)}/{n}"
                    + (f"  median recall={np.median(recalls):.3f}" if recalls else "")
                    + (f"  median precision={np.median(precs):.3f}" if precs else ""))
        logger.info("─" * 66)


# ---------------------------------------------------------------------------
# REPARSE - re-run the parser over STORED response text (no Claude calls)
# ---------------------------------------------------------------------------
def reparse_outputs(root: Path) -> int:
    gt_store: Optional[GroundTruthStore] = None
    try:
        gt_store = GroundTruthStore(GROUND_TRUTH_FILE)
    except (FileNotFoundError, ValueError):
        logger.info("Reparse: no ground-truth file - gt_score fields will be empty.")
    n = 0
    for jf in sorted(root.glob("*/*.json")):
        if jf.parent.name.startswith(ANALYSIS_DIR_NAME):
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Reparse: cannot read {jf} ({e})")
            continue
        changed = False
        for r in data.get("results", []):
            txt = str(r.get("Generated_Response", ""))
            if not txt or txt.startswith("ERROR") or txt.startswith("DEGENERATE_OUTPUT"):
                continue
            r.update(derive_parse_fields(txt, r.get("strategy", ""), str(r.get("Source_PublicationYear", "")), r.get("Query_ID"), gt_store))
            n += 1
            changed = True
        if changed:
            jf.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Reparse: {n} stored response(s) re-parsed under {root} (no model calls).")
    return n


# ---------------------------------------------------------------------------
# ANALYSIS & REPORTING - computed only from the saved outputs (ground truth optional)
# ---------------------------------------------------------------------------
def collect_records(root: Path) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for jf in sorted(root.glob("*/*.json")):
        if jf.parent.name.startswith(ANALYSIS_DIR_NAME):
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Analysis: cannot read {jf} ({e})")
            continue
        md = data.get("metadata", {}) or {}
        for r in data.get("results", []):
            r = dict(r)
            r.setdefault("model_alias", md.get("model_alias"))
            r.setdefault("strategy", md.get("strategy"))
            r.setdefault("rag_tier", md.get("rag_tier"))
            r.setdefault("system_mode_requested", md.get("system_mode_requested", "neutral"))
            r["_fewshot_cell"] = bool(md.get("fewshot_exemplar_enabled"))
            r["_scaffold_cell"] = bool(md.get("recall_scaffold_enabled"))
            recs.append(r)
    return recs


def _gt_authors(gt_store: Optional[GroundTruthStore], sno: str) -> Optional[List[Dict[str, Any]]]:
    if gt_store is None:
        return None
    recs = gt_store.by_sno.get(str(sno))
    if not recs or any(is_error_row(r.author) for r in recs):
        return None
    merged: Dict[str, Dict[str, Any]] = {}
    for r in recs:
        if is_sentinel_author(r.author):
            continue
        d = merged.setdefault(name_key(r.author), {"author": r.author, "birthplace": r.birthplace, "pop": gt_pop_value(r.popcount)})
        if not d["birthplace"]:
            d["birthplace"] = r.birthplace
        if d["pop"] is None:
            d["pop"] = gt_pop_value(r.popcount)
    return list(merged.values())        # [] = GT says no qualifying author


def _match_pairs(g_list: List[Dict[str, Any]], p_list: List[Dict[str, Any]]):
    rem_gt, unm, pairs = list(range(len(g_list))), list(range(len(p_list))), []
    for exact in (True, False):
        for gi in list(rem_gt):
            for pi in list(unm):
                a, b = g_list[gi]["author"], p_list[pi]["Author"]
                if (name_key(a) == name_key(b)) if exact else names_match(a, b):
                    pairs.append((gi, pi)); rem_gt.remove(gi); unm.remove(pi)
                    break
    return pairs, rem_gt, unm


def evaluate_record(rec: Dict[str, Any], gt_store: Optional[GroundTruthStore]) -> Dict[str, Any]:
    resp = str(rec.get("Generated_Response", ""))
    is_err = resp.startswith("ERROR")
    rows = [] if is_err else (rec.get("Parsed_Rows") or [])
    pred = [r for r in rows if not is_sentinel_author(r.get("Author", ""))]
    diag = rec.get("parse_diagnostics") or {}
    sid = str(rec.get("Query_ID", ""))
    src_year, yp = str(rec.get("Source_PublicationYear", "")), str(rec.get("PublicationYear_Parsed", ""))
    out: Dict[str, Any] = {
        "model_alias": rec.get("model_alias"), "strategy": rec.get("strategy"), "rag_tier": rec.get("rag_tier"),
        "system_mode": rec.get("system_mode_requested", "neutral"), "fewshot": rec.get("_fewshot_cell"),
        "scaffold_cell": rec.get("_scaffold_cell"), "Query_ID": sid, "status": rec.get("generation_status", ""),
        "is_error": is_err, "truncated": rec.get("generation_status") in (ST_TRUNC_LENGTH, ST_TRUNC_REASONING),
        "empty": rec.get("generation_status") == ST_EMPTY, "degenerate": bool(rec.get("Degenerate_Flag")),
        "latency": rec.get("generation_time_seconds"), "completion_tokens": _completion_tokens(rec.get("usage")),
        "text_chars": rec.get("text_chars"), "thinking_chars": rec.get("thinking_chars"), "cost_usd": rec.get("cost_usd_reported"),
        "thinking": rec.get("thinking", "default"), "tools_mode": rec.get("tools_mode", "none"),
        "n_tool_calls": rec.get("n_tool_calls"),
        "year_parsed": yp, "source_year": src_year, "year_ok": (yp == src_year) if (yp and src_year) else np.nan,
        "outcome": "error" if is_err else diag.get("outcome", ""), "refusal_contradiction": bool(diag.get("refusal_contradicts_scaffold")),
        "scaffold_candidates": diag.get("scaffold_candidate_count", 0) or 0,
        "coyear_yes": diag.get("scaffold_coyear_yes_count", 0) or 0, "coyear_unsure": diag.get("scaffold_coyear_unsure_count", 0) or 0,
        "coyear_no": diag.get("scaffold_coyear_no_count", 0) or 0, "n_pred": len(pred),
        "pop_numeric_n": sum(str(r.get("PopCount", "")).isdigit() for r in pred), "pop_total_n": len(pred),
        "tp": 0, "n_gt": 0, "scored": False, "birth_ok": 0, "birth_n": 0, "pop_5_n": 0, "pop_cmp_n": 0, "fn_names": [], "fp_names": []}
    g_list = None if is_err else _gt_authors(gt_store, sid)
    if g_list is None:
        return out
    pairs, rem_gt, unm = _match_pairs(g_list, pred)
    out.update({"tp": len(pairs), "n_gt": len(g_list), "scored": True,
                "fn_names": [g_list[i]["author"] for i in rem_gt], "fp_names": [pred[i]["Author"] for i in unm]})
    for gi, pi in pairs:
        g, p = g_list[gi], pred[pi]
        out["birth_n"] += 1
        out["birth_ok"] += int(places_match(p.get("Birthplace", ""), g["birthplace"]))
        pp = str(p.get("PopCount", "")).strip()
        if pp.isdigit() and g["pop"]:
            out["pop_cmp_n"] += 1
            out["pop_5_n"] += int(abs(int(pp) - g["pop"]) / g["pop"] <= 0.05)
    return out


def _agg(g: pd.DataFrame) -> Dict[str, Any]:
    sc = g[g["scored"]]
    tp, npred, ngt = sc["tp"].sum(), sc["n_pred"].sum(), sc["n_gt"].sum()
    p = tp / npred if npred else np.nan
    r = tp / ngt if ngt else np.nan
    f = 2 * p * r / (p + r) if (p == p and r == r and (p + r) > 0) else np.nan

    def rate(a: str, b: str) -> float:
        d = g[b].sum()
        return g[a].sum() / d if d else np.nan

    oc = g["outcome"]
    return {"records": len(g), "error_rate": g["is_error"].mean(), "truncated_rate": g["truncated"].mean(),
            "empty_rate": g["empty"].mean(), "degenerate_rate": g["degenerate"].mean(),
            "rows_parsed_rate": (oc == "rows_parsed").mean(), "explicit_no_match_rate": (oc == "explicit_no_match").mean(),
            "empty_body_rate": (oc == "empty_body").mean(), "unparseable_rate": (oc == "unparseable_body").mean(),
            "refusal_contradiction_rate": g["refusal_contradiction"].mean(), "mean_scaffold_candidates": g["scaffold_candidates"].mean(),
            "mean_authors": g["n_pred"].mean(), "year_accuracy": pd.to_numeric(g["year_ok"], errors="coerce").mean(),
            "pop_numeric_rate": rate("pop_numeric_n", "pop_total_n"),
            "gt_scored": int(g["scored"].sum()), "gt_precision": p, "gt_recall": r, "gt_f1": f,
            "gt_birthplace_acc": rate("birth_ok", "birth_n"), "gt_pop_within_5pct": rate("pop_5_n", "pop_cmp_n"),
            "mean_latency_s": pd.to_numeric(g["latency"], errors="coerce").mean(),
            "mean_completion_tokens": g["completion_tokens"].mean(),
            "mean_text_chars": pd.to_numeric(g["text_chars"], errors="coerce").mean(),
            "mean_thinking_chars": pd.to_numeric(g["thinking_chars"], errors="coerce").mean(),
            "total_cost_usd": pd.to_numeric(g["cost_usd"], errors="coerce").sum()}


def summarise(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        d = dict(zip(by, key))
        d.update(_agg(g))
        rows.append(d)
    return pd.DataFrame(rows)


def _tbl(d: pd.DataFrame) -> str:
    return "(no data)" if d is None or d.empty else d.to_string(index=False, float_format=lambda x: f"{x:.3f}")


def _make_plots(df: pd.DataFrame, outdir: Path, has_gt: bool) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"matplotlib unavailable ({e}) - plots skipped")
        return []
    saved: List[str] = []

    def save(fig, name: str) -> None:
        p = outdir / name
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    def code(s: str) -> str:
        return STRATEGY_SUFFIX.get(s, s).split("_")[0]

    def stacked(s: pd.DataFrame, cols, title: str, name: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 4.6))
        x = np.arange(len(s))
        bottom = np.zeros(len(s))
        for col, lab in cols:
            v = s[col].fillna(0).values
            ax.bar(x, v, bottom=bottom, label=lab); bottom += v
        ax.set_xticks(x); ax.set_xticklabels([code(v) for v in s["strategy"]], rotation=30, ha="right")
        ax.set_ylim(0, 1.02); ax.legend(fontsize=7, ncol=2); ax.set_title(title)
        save(fig, name)

    bs = summarise(df, ["strategy"])
    bm = summarise(df, ["model_alias"])
    if not bs.empty:
        bs = bs.assign(ok_rate=(1 - bs["truncated_rate"] - bs["empty_rate"] - bs["error_rate"]).clip(lower=0))
        stacked(bs, [("ok_rate", "ok"), ("truncated_rate", "truncated at cap"), ("empty_rate", "empty"), ("error_rate", "error")],
                "Call outcomes by strategy (share of records)", "outcomes_by_strategy.png")
        stacked(bs, [("rows_parsed_rate", "rows parsed"), ("explicit_no_match_rate", "explicit 'no matching authors'"),
                     ("empty_body_rate", "empty body"), ("unparseable_rate", "unparseable body")],
                "Parse outcome by strategy (share of records; error records excluded from the parser)", "parse_outcomes_by_strategy.png")
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.bar([code(v) for v in bs["strategy"]], bs["mean_authors"], color="#2a6fbb")
        ax.set_ylabel("mean authors parsed per book"); ax.set_title("Answer size by strategy")
        save(fig, "authors_by_strategy.png")
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.bar([code(v) for v in bs["strategy"]], bs["mean_latency_s"], color="#c0392b")
        ax.set_ylabel("mean latency per call (s)"); ax.set_title("Latency by strategy")
        save(fig, "latency_by_strategy.png")
        s1 = df[df["strategy"] == STRATEGY_FLAT_CONJUNCTIVE]
        if len(s1) and s1["coyear_yes"].add(s1["coyear_unsure"]).add(s1["coyear_no"]).sum() > 0:
            g = s1.groupby("rag_tier")[["coyear_yes", "coyear_unsure", "coyear_no"]].mean()
            fig, ax = plt.subplots(figsize=(8, 4))
            g.plot(kind="bar", stacked=True, ax=ax)
            ax.set_ylabel("mean CO-YEAR CHECK verdicts per record"); ax.set_title("S1 scaffold verdicts by RAG tier")
            save(fig, "s1_coyear_verdicts.png")
    if not bm.empty:
        fig, ax = plt.subplots(figsize=(9, max(3.2, 0.4 * len(bm) + 1.5)))
        y = np.arange(len(bm))
        ax.barh(y - 0.2, bm["mean_text_chars"].fillna(0), 0.4, label="visible text (chars)", color="#2a6fbb")
        ax.barh(y + 0.2, bm["mean_thinking_chars"].fillna(0), 0.4, label="thinking (chars)", color="#c0392b")
        ax.set_yticks(y); ax.set_yticklabels(bm["model_alias"]); ax.legend(fontsize=8)
        ax.set_title("Mean visible vs thinking length per call, by model")
        save(fig, "text_vs_thinking_by_model.png")
    if has_gt and df["scored"].any():
        sc = df[df["scored"]]
        for fac, lab_fn, fname in (("strategy", code, "gt_f1_by_strategy.png"), ("model_alias", str, "gt_f1_by_model.png"),
                                   ("rag_tier", str, "gt_f1_by_rag.png")):
            s = summarise(sc, [fac]).sort_values("gt_f1")
            if s.empty:
                continue
            fig, ax = plt.subplots(figsize=(9, max(3.5, 0.36 * len(s) + 1.5)))
            y, hh = np.arange(len(s)), 0.27
            ax.barh(y - hh, s["gt_precision"].fillna(0), hh, label="precision")
            ax.barh(y, s["gt_recall"].fillna(0), hh, label="recall")
            ax.barh(y + hh, s["gt_f1"].fillna(0), hh, label="F1 (micro)")
            ax.set_yticks(y); ax.set_yticklabels([lab_fn(v) for v in s[fac]]); ax.set_xlim(0, 1); ax.legend(loc="lower right")
            ax.set_title(f"Ground-truth scoring by {fac}")
            save(fig, fname)
    return saved


def run_analysis(root: Path) -> None:
    recs = collect_records(root)
    if not recs:
        logger.warning(f"Analysis: no result files under {root.resolve()}.")
        return
    gt_store: Optional[GroundTruthStore] = None
    try:
        gt_store = GroundTruthStore(GROUND_TRUTH_FILE)
    except (FileNotFoundError, ValueError):
        logger.info("Analysis: no ground-truth file found - structural analysis only (no accuracy scoring).")
    df = pd.DataFrame([evaluate_record(r, gt_store) for r in recs])
    df = df.drop_duplicates(subset=["model_alias", "strategy", "rag_tier", "system_mode", "fewshot", "scaffold_cell", "Query_ID"],
                            keep="last").reset_index(drop=True)
    outdir = root / ANALYSIS_DIR_NAME
    outdir.mkdir(parents=True, exist_ok=True)
    has_gt = bool(df["scored"].any())
    df.drop(columns=["fn_names", "fp_names"]).to_csv(outdir / "eval_per_record.csv", index=False, encoding="utf-8-sig")
    by_model, by_strat, by_rag = summarise(df, ["model_alias"]), summarise(df, ["strategy"]), summarise(df, ["rag_tier"])
    by_cfg = summarise(df, ["model_alias", "strategy", "rag_tier", "system_mode", "fewshot", "scaffold_cell"])
    for name, t in (("summary_by_model", by_model), ("summary_by_strategy", by_strat), ("summary_by_rag_tier", by_rag),
                    ("summary_by_config", by_cfg)):
        t.to_csv(outdir / f"{name}.csv", index=False, encoding="utf-8-sig")
    plots = _make_plots(df, outdir, has_gt)
    h = _agg(df)
    L: List[str] = ["=" * 100, "BOOK AUTHOR ANALYSER (CLAUDE CODE) - ANALYSIS REPORT",
                    f"generated {datetime.now().isoformat(timespec='seconds')}  (script version {SCRIPT_VERSION})", "=" * 100,
                    "\n1. SCOPE",
                    f"  records={len(df)}  models={df['model_alias'].nunique()}  strategies={df['strategy'].nunique()}  "
                    f"rag_tiers={df['rag_tier'].nunique()}  system modes={sorted(df['system_mode'].unique())}  "
                    f"few-shot cells={sorted(df['fewshot'].unique().tolist())}",
                    "  Backend: `claude -p` in a Claude Code cloud session; temperature / top_p not controllable; tools disabled; "
                    "raw events per call in <cell dir>/raw_calls. RAG tiers apply to S1 only (other strategies show rag_tier no_evidence).",
                    f"  Ground-truth scoring: {'ON (' + str(int(df['scored'].sum())) + ' scored records; a book with no GT rows, or with a GT retrieval error, is not scored; a book whose GT is NO RESULT scores every returned author as a false positive)' if has_gt else 'OFF'}",
                    "\n2. HEADLINE (all records pooled)"]
    for k, v in h.items():
        L.append(f"  {k:<28}{v:.4f}" if isinstance(v, (float, np.floating)) and v == v else f"  {k:<28}{v}")
    L += ["\n3. BY MODEL", _tbl(by_model),
          "\n4. BY STRATEGY", _tbl(by_strat.assign(code=by_strat['strategy'].map(lambda s: STRATEGY_SUFFIX.get(s, s).split('_')[0]))),
          "\n5. BY RAG TIER (S1 cells carry the tier; everything else is no_evidence)", _tbl(by_rag),
          "\n6. GENERATION HEALTH", _tbl(df.groupby(["model_alias", "system_mode", "status"]).size().reset_index(name="count"))]
    if has_gt:
        sc = df[df["scored"]]
        miss = Counter(n for lst in sc["fn_names"] for n in lst).most_common(15)
        fps = Counter(n for lst in sc["fp_names"] for n in lst).most_common(15)
        L += ["\n7. GROUND-TRUTH DETAIL",
              "  Most missed authors: " + ("; ".join(f"{a}: {b}" for a, b in miss) or "none"),
              "  Most frequent false positives: " + ("; ".join(f"{a}: {b}" for a, b in fps) or "none")]
    ins: List[str] = []
    if h["truncated_rate"] == h["truncated_rate"] and h["truncated_rate"] > 0:
        tm = by_model[by_model["truncated_rate"] > 0].sort_values("truncated_rate", ascending=False)
        ins.append(f"{h['truncated_rate']:.1%} of calls hit the output-token cap (" +
                   ", ".join(f"{r['model_alias']} {r['truncated_rate']:.0%}" for _, r in tm.head(6).iterrows()) +
                   "). Compare text_chars with thinking_chars to see what used the budget; raise --max_new_tokens or set --max_thinking_tokens.")
    if h["refusal_contradiction_rate"] == h["refusal_contradiction_rate"] and h["refusal_contradiction_rate"] > 0:
        ins.append(f"{h['refusal_contradiction_rate']:.1%} of records write 'No matching authors found' (or nothing) although their own "
                   f"CO-YEAR CHECK marked at least one author yes/unsure (refusal contradicts scaffold).")
    if h["explicit_no_match_rate"] == h["explicit_no_match_rate"] and h["explicit_no_match_rate"] > 0:
        ins.append(f"{h['explicit_no_match_rate']:.1%} of records end in an explicit 'no matching authors' answer.")
    if has_gt:
        bf = by_strat.dropna(subset=["gt_f1"]).sort_values("gt_f1")
        if len(bf) > 1:
            ins.append(f"Strategy F1 spread: best {STRATEGY_CODE[bf.iloc[-1]['strategy']]} ({bf.iloc[-1]['gt_f1']:.3f}), "
                       f"worst {STRATEGY_CODE[bf.iloc[0]['strategy']]} ({bf.iloc[0]['gt_f1']:.3f}).")
    if h["year_accuracy"] == h["year_accuracy"]:
        ins.append(f"Publication-year accuracy against the source year: {h['year_accuracy']:.1%} (records where both years are known).")
    if h["total_cost_usd"] and h["total_cost_usd"] == h["total_cost_usd"]:
        ins.append(f"Reported cost summed over all records (Claude Code 'total_cost_usd'): {h['total_cost_usd']:.2f} USD.")
    L += ["\n8. KEY INSIGHTS"] + [f"  - {s}" for s in ins]
    L += ["\n9. LIMITATIONS",
          "  - One call per (model, strategy, RAG tier, book): small differences may be sampling noise (no replicates).",
          "  - Author matching uses normalised names with token-subset tolerance; a shared surname can merge two authors.",
          "  - Sampling parameters cannot be set through `claude -p`; a call that hits the cap is stored truncated, not retried.",
          "  - The assistant-turn prefill used in the HF runs is not possible here; results are not comparable to HF runs.",
          "  - The few-shot exemplar is built from ground-truth rows of another book; if that book shares the target's year it "
          "leaks candidates. Run with --no_fewshot for a clean comparison (those cells carry a _NOFEW suffix).",
          "  - system_mode neutral vs default changes the system prompt; do not pool them when comparing models.",
          "\nFILES: " + (", ".join(Path(p).name for p in plots) if plots else "(no plots written)")]
    (outdir / "analysis_report.txt").write_text("\n".join(L), encoding="utf-8")
    logger.info(f"Analysis written to {outdir} ({len(plots)} plots, report + CSVs)")
    print("\n".join(L[:24]))
    print(f"\n  Full report: {outdir / 'analysis_report.txt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    global SOURCE_FILE, REFERENCE_FILE, GROUND_TRUTH_FILE
    p = argparse.ArgumentParser(
        description=(f"Book Author Analyser - Claude Code edition (script version {SCRIPT_VERSION})\n"
                     "Claude models only; every call is a streamed headless `claude -p` run inside a Claude Code cloud session.\n"
                     "15 strategies x 7 RAG tiers (--rag_tier applies to S1 flat_conjunctive only, as in the HF script).\n"
                     "Raw events of every call are saved under <cell dir>/raw_calls; the full log is <output root>/run.log.\n"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(["", "Models (--model; several aliases or 'all')", "-" * 42]
                         + [f"  {a:<20} -> {MODEL_REGISTRY[a]}{'  [DEFAULT]' if a == DEFAULT_MODEL_ALIAS else ''}" for a in sorted(MODEL_REGISTRY)]
                         + ["", "Strategies (--strategy takes the LONG names):"] + [f"  {STRATEGY_CODE[s]:<4} {s}" for s in INDIVIDUAL_STRATEGIES]
                         + ["  all  runs all 15", "", "RAG tiers: " + ", ".join(RAG_TIERS), f"Output root: {OUTPUT_ROOT}"]))
    p.add_argument("--model", nargs="+", default=[DEFAULT_MODEL_ALIAS], choices=sorted(MODEL_REGISTRY) + ["all"])
    p.add_argument("--strategy", default=None, choices=SUPPORTED_STRATEGIES)
    p.add_argument("--rag_tier", default=RAG_TIER_NO_EVIDENCE, choices=RAG_TIERS)
    p.add_argument("--rag_max_rows", type=int, default=RAG_EVIDENCE_MAX_ROWS, help=f"evidence rows cap (HF default {RAG_EVIDENCE_MAX_ROWS}; 0 = every record)")
    p.add_argument("--claude_bin", default="claude", help="Claude Code CLI executable (default: claude)")
    p.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT, help="seconds allowed per `claude -p` call")
    p.add_argument("--max_thinking_tokens", type=int, default=None,
                   help="sets MAX_THINKING_TOKENS for the call (0 = try to disable thinking); default: untouched")
    p.add_argument("--thinking", choices=["default", "off", "low", "medium", "high", "xhigh", "max"], default="default",
                   help="off: MAX_THINKING_TOKENS=0; low..max: passes --effort <level>; default: untouched")
    p.add_argument("--tools_mode", choices=["none", "web"], default="none",
                   help="none: --tools '' (default); web: WebSearch + WebFetch enabled and pre-approved")
    p.add_argument("--max_turns", type=int, default=None,
                   help=f"--max-turns per call (default 1; {WEB_DEFAULT_MAX_TURNS} with --tools_mode web)")
    p.add_argument("--system_mode", choices=["neutral", "default"], default="neutral",
                   help="neutral: a short neutral --system-prompt (default); default: no flag, Claude Code's own system prompt")
    p.add_argument("--max_new_tokens", type=int, default=GENERATION_MAX_NEW_TOKENS, help="output-token cap per call")
    p.add_argument("--max_queries", type=int, default=None, help="process at most N pending queries per cell (pilot run)")
    p.add_argument("--no_recall_scaffold", dest="recall_scaffold", action="store_false", help="drop the PRIZE WINNERS / CO-YEAR CHECK scaffold from S1")
    p.add_argument("--no_fewshot", dest="fewshot", action="store_false", help="drop the few-shot exemplar (avoids ground-truth leakage)")
    p.add_argument("--no_store_prompts", dest="store_prompts", action="store_false", help="do not store prompt text in the JSON")
    p.add_argument("--no-resume", dest="no_resume", action="store_true", default=False)
    p.add_argument("--source_file", default=SOURCE_FILE)
    p.add_argument("--reference_file", default=REFERENCE_FILE)
    p.add_argument("--ground_truth_file", default=GROUND_TRUTH_FILE)
    p.add_argument("--reparse", action="store_true", default=False, help="re-parse STORED responses (no Claude calls), then analyse")
    p.add_argument("--analyze_only", action="store_true", default=False)
    p.add_argument("--no_analysis", action="store_true", default=False)
    p.set_defaults(recall_scaffold=True, fewshot=True, store_prompts=True)
    args = p.parse_args()
    SOURCE_FILE, REFERENCE_FILE, GROUND_TRUTH_FILE = args.source_file, args.reference_file, args.ground_truth_file

    root = OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)                        # full log of every run, next to the results
    fh = logging.FileHandler(root / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info(f"=== run started: {' '.join(sys.argv)} (script version {SCRIPT_VERSION}) ===")

    if args.reparse:
        reparse_outputs(root)
        run_analysis(root)
        return 0
    if args.analyze_only:
        run_analysis(root)
        return 0
    if not args.strategy:
        p.error("--strategy is required unless --analyze_only or --reparse is used")
    if not shutil.which(args.claude_bin):
        print(f"ERROR: '{args.claude_bin}' (Claude Code CLI) not found on PATH. This script only runs through Claude Code.")
        return 1
    set_vars = [v for v in _BILLING_ENV_VARS if os.environ.get(v)]
    if set_vars:
        logger.warning(f"Environment variable(s) {set_vars} are set: the Claude Code CLI may bill through them instead of the cloud-session credit.")

    aliases = sorted(MODEL_REGISTRY) if "all" in args.model else list(dict.fromkeys(args.model))
    strategies_to_run = INDIVIDUAL_STRATEGIES if args.strategy == STRATEGY_ALL else [args.strategy]
    uses_rag = args.strategy in (STRATEGY_FLAT_CONJUNCTIVE, STRATEGY_ALL) and args.rag_tier != RAG_TIER_NO_EVIDENCE
    if args.rag_tier != RAG_TIER_NO_EVIDENCE and not uses_rag:
        logger.warning(f"--rag_tier {args.rag_tier} is ignored: RAG tiers apply to flat_conjunctive (S1) only.")

    print("=" * 72)
    print(f"BOOK AUTHOR ANALYSER - CLAUDE CODE   (script version {SCRIPT_VERSION})")
    print("=" * 72)
    print(f"  Models ({len(aliases)}) : {', '.join(aliases)}")
    print(f"  Strategy(ies)    : {args.strategy}" + (f" -> {len(strategies_to_run)} will run" if args.strategy == STRATEGY_ALL else ""))
    print(f"  RAG tier         : {args.rag_tier}" + ("  (S1 only)" if uses_rag else "  (no evidence injected)"))
    print(f"  Recall scaffold  : {'ON (S1, no evidence)' if args.recall_scaffold else 'OFF'}   Few-shot exemplar: {'ON' if args.fewshot else 'OFF'}")
    print(f"  Source file      : {SOURCE_FILE}")
    print(f"  Reference file   : {REFERENCE_FILE}")
    print(f"  Ground truth file: {GROUND_TRUTH_FILE}")
    print(f"  System mode      : {args.system_mode}")
    print(f"  Thinking         : {args.thinking}   Tools: {args.tools_mode}   Max turns: {args.max_turns or 'auto'}")
    print(f"  Resume           : {'OFF (--no-resume)' if args.no_resume else 'ON (default)'}")
    print(f"  Max new tokens   : {args.max_new_tokens} per call; max thinking tokens: {args.max_thinking_tokens}; timeout {args.timeout}s")
    print(f"  Max queries      : {args.max_queries or 'all'}")
    print(f"  Prize criterion  : Booker Prize OR Pulitzer Prize")
    print(f"  Output root      : {root}   (log: {root / 'run.log'}; raw events: <cell dir>/raw_calls)")
    print("=" * 72)

    try:
        _resolve_path(SOURCE_FILE)
    except FileNotFoundError:
        print(f"ERROR: Input file '{SOURCE_FILE}' not found.")
        return 1

    ref_store: Optional[ReferenceStore] = None
    gt_store: Optional[GroundTruthStore] = None
    try:
        if uses_rag and args.rag_tier in REF_TIERS:
            ref_store = ReferenceStore(REFERENCE_FILE)
        if (uses_rag and args.rag_tier in GT_TIERS) or args.fewshot or True:
            try:
                gt_store = GroundTruthStore(GROUND_TRUTH_FILE)       # also scores every record against GT
            except (FileNotFoundError, ValueError) as e:
                if uses_rag and args.rag_tier in GT_TIERS:
                    raise
                logger.warning(f"Ground truth not loaded ({e}); no few-shot exemplar and no per-record GT score.")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR loading RAG evidence stores: {e}")
        return 1
    fewshot_store = FewShotExemplarStore(gt_store) if (args.fewshot and FEWSHOT_EXEMPLAR_ENABLED and gt_store is not None) else None
    if fewshot_store is not None and not fewshot_store.available:
        fewshot_store = None

    overall_start = datetime.now()
    failed: List[str] = []
    for m_idx, alias in enumerate(aliases, 1):
        print(f"\n{'#' * 72}\n  MODEL [{m_idx}/{len(aliases)}] {alias}\n{'#' * 72}")
        try:
            client = ClaudeCodeClient(model_alias=alias, max_new_tokens=args.max_new_tokens, system_mode=args.system_mode,
                                      claude_bin=args.claude_bin, timeout=args.timeout, max_thinking_tokens=args.max_thinking_tokens,
                                      thinking=args.thinking, tools_mode=args.tools_mode, max_turns=args.max_turns)
            client.raw_dir = root / "raw_calls"
            client.validate_access()
        except Exception as e:
            logger.error(f"{alias}: startup check failed - skipping this model ({e})")
            failed.append(f"{alias}[startup]")
            continue
        for strategy in strategies_to_run:
            strat_tier = args.rag_tier if (strategy == STRATEGY_FLAT_CONJUNCTIVE and uses_rag) else RAG_TIER_NO_EVIDENCE
            print(f"\n{'-' * 72}\n  {alias} | {strategy} ({STRATEGY_SUFFIX[strategy]}) | RAG: {strat_tier}\n"
                  f"  Hint: {STRATEGY_HINT[strategy]}\n{'-' * 72}")
            try:
                analyser = BookAuthorAnalyser(strategy=strategy, client=client, no_resume=args.no_resume, rag_tier=strat_tier,
                                              ref_store=ref_store, gt_store=gt_store, fewshot_store=fewshot_store,
                                              recall_scaffold=args.recall_scaffold, rag_max_rows=args.rag_max_rows,
                                              store_prompts=args.store_prompts)
                df = analyser.load_input()
                if df.empty:
                    print("  No data in source file. Skipping.")
                    continue
                t0 = datetime.now()
                results = analyser.process(df, max_queries=args.max_queries)
                ok = sum(1 for r in results if not str(r.get("Generated_Response", "")).startswith("ERROR"))
                populated = sum(1 for r in results if (r.get("parse_diagnostics") or {}).get("outcome") == "rows_parsed")
                print(f"  done: total={len(results)} success={ok} errors={len(results) - ok} with-rows={populated} "
                      f"time={datetime.now() - t0}  -> {analyser.output_file}")
                if len(results) - ok or analyser.stopped_early:
                    failed.append(f"{alias}[{strategy}]")
            except KeyboardInterrupt:
                print("\n  Interrupted - progress saved.")
                return 130
            except Exception as e:
                print(f"  Fatal error in {alias}/{strategy}: {e}")
                traceback.print_exc()
                failed.append(f"{alias}[{strategy}]")

    print(f"\n{'=' * 72}\nALL RUNS COMPLETE\n  Total wall-clock time : {datetime.now() - overall_start}\n  Output root           : {root}")
    if failed:
        print(f"  Failed / incomplete: {failed}")
    print("=" * 72)
    if not args.no_analysis:
        try:
            run_analysis(root)
        except Exception as e:
            logger.error(f"Analysis stage failed: {e}")
            traceback.print_exc()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())