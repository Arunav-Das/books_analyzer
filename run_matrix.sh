#!/usr/bin/env bash
# NLQ_GENERAL x S1 flat_conjunctive x 7 RAG tiers (no_evidence + RAG1..RAG6)
#   x thinking {off, low effort} x tools {none, WebSearch+WebFetch}
# Order: no-web/no-tools baseline first. Each cell runs once (no rerun on failure).
cd "$(dirname "$0")"
MODEL=claude-sonnet-4-6
TIERS="no_evidence year_list_only titles_authors titles_authors_prize full_with_birthplace full_with_population full_with_distractors"
CONFIGS="off:none low:none off:web low:web"
for cfg in $CONFIGS; do
  th=${cfg%%:*}; tl=${cfg##*:}
  for tier in $TIERS; do
    echo "################ CELL thinking=$th tools=$tl rag_tier=$tier  $(date -Is) ################"
    python3 book_analyzer_inf.py --model "$MODEL" --strategy flat_conjunctive --rag_tier "$tier" \
      --thinking "$th" --tools_mode "$tl" --max_queries 2 --max_new_tokens 3000 --timeout 600 \
      --no-resume --no_analysis
    echo "################ EXIT $? thinking=$th tools=$tl rag_tier=$tier  $(date -Is) ################"
  done
done
echo "################ ANALYSIS $(date -Is) ################"
python3 book_analyzer_inf.py --analyze_only
echo "################ MATRIX DONE exit=$? $(date -Is) ################"
