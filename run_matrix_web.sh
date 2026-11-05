#!/usr/bin/env bash
# Rerun of the web-tools cells only: NLQ_GENERAL x S1 flat_conjunctive x 7 RAG tiers
#   x thinking {off, low effort} x tools WebSearch+WebFetch, --max_turns 30, turn-limit hits stored and NOT retried.
# Output cells end in _WEB_T30 (first-run _WEB cells with 10 turns are kept untouched). Each cell runs once.
cd "$(dirname "$0")"
MODEL=claude-sonnet-4-6
TIERS="no_evidence year_list_only titles_authors titles_authors_prize full_with_birthplace full_with_population full_with_distractors"
for th in off low; do
  for tier in $TIERS; do
    echo "################ CELL thinking=$th tools=web max_turns=30 rag_tier=$tier  $(date -Is) ################"
    python3 book_analyzer_inf.py --model "$MODEL" --strategy flat_conjunctive --rag_tier "$tier" \
      --thinking "$th" --tools_mode web --max_turns 30 --max_queries 2 --max_new_tokens 3000 --timeout 600 \
      --no-resume --no_analysis
    echo "################ EXIT $? thinking=$th tools=web max_turns=30 rag_tier=$tier  $(date -Is) ################"
  done
done
echo "################ ANALYSIS $(date -Is) ################"
python3 book_analyzer_inf.py --analyze_only
echo "################ MATRIX DONE exit=$? $(date -Is) ################"
