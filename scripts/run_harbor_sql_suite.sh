#!/bin/sh
set -eu

if ! command -v harbor >/dev/null 2>&1; then
  echo "Harbor is required. Install it with: uv tool install 'harbor>=0.22,<1'" >&2
  exit 1
fi

mode="${1:-frozen}"
case "$mode" in
  frozen)
    agent="geomed_copilot.harbor_agent:RadMeasureFrozenPlannerAgent"
    output="outputs/harbor/frozen_qwen"
    ;;
  oracle)
    agent="oracle"
    output="outputs/harbor/oracle"
    ;;
  *)
    echo "usage: $0 [frozen|oracle]" >&2
    exit 2
    ;;
esac

PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src" \
  harbor run \
    --path harbor/tasks/radmeasure_sql_repair_v1 \
    --agent "$agent" \
    --n-concurrent 1 \
    --jobs-dir "$output" \
    --yes
