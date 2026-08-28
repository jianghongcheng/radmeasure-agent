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
    task_path="harbor/tasks/radmeasure_sql_repair_v1"
    ;;
  v3-frozen)
    agent="geomed_copilot.harbor_agent:RadMeasureV3FrozenPlannerAgent"
    output="outputs/harbor/v3_frozen_qwen"
    task_path="harbor/tasks/radmeasure_sql_repair_v3"
    ;;
  v3-oracle)
    agent="oracle"
    output="outputs/harbor/v3_oracle"
    task_path="harbor/tasks/radmeasure_sql_repair_v3"
    ;;
  *)
    echo "usage: $0 [frozen|oracle|v3-frozen|v3-oracle]" >&2
    exit 2
    ;;
esac

task_path="${task_path:-harbor/tasks/radmeasure_sql_repair_v1}"

PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src" \
  harbor run \
    --path "$task_path" \
    --agent "$agent" \
    --n-concurrent 1 \
    --jobs-dir "$output" \
    --yes
