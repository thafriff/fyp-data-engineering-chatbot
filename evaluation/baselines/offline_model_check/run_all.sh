#!/bin/sh
# Offline-model check, 2026-11-23: the same frozen inputs for each local model, then coach mode G01-G10 x3.
cd "$(dirname "$0")/../../.."   # project root
OUT=evaluation/baselines/offline_model_check
for m in llama3.2:3b qwen3:1.7b qwen3.5:2b; do
  f=$(echo $m | tr ':' '_')
  venv/Scripts/python evaluation/run_baseline.py --model $m --contexts evaluation/notebook04/contexts.json --out $OUT/baseline_$f.json > $OUT/log_baseline_$f.txt 2>&1
  curl -s http://localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" > /dev/null
done
for m in llama3.2:3b qwen3:1.7b qwen3.5:2b; do
  f=$(echo $m | tr ':' '_')
  venv/Scripts/python evaluation/guardrail_router_coach_run.py $OUT/coach_$f.json --runs 3 --model $m > $OUT/log_coach_$f.txt 2>&1
  curl -s http://localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" > /dev/null
done
echo ALL DONE > $OUT/done.txt
