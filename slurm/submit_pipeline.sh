#!/usr/bin/env bash
set -euo pipefail

: "${SI_ACCOUNT:?set SI_ACCOUNT}"
: "${SI_CPU_PARTITION:?set SI_CPU_PARTITION}"
: "${SI_GPU_PARTITION:?set SI_GPU_PARTITION}"
: "${SI_REPO_ROOT:?set SI_REPO_ROOT}"
: "${SI_DATA_ROOT:?set SI_DATA_ROOT}"
: "${SI_ARTIFACT_ROOT:?set SI_ARTIFACT_ROOT}"
: "${SI_SLURM_CONF:?set SI_SLURM_CONF}"

export SI_ACCOUNT SI_CPU_PARTITION SI_GPU_PARTITION
export SI_REPO_ROOT SI_DATA_ROOT SI_ARTIFACT_ROOT SI_SLURM_CONF
export SI_CONFIG="${SI_CONFIG:-${SI_REPO_ROOT}/configs/v0.yaml}"
export SI_MANIFEST="${SI_MANIFEST:-${SI_ARTIFACT_ROOT}/studies/v0/data_manifest.json}"
export SI_MODE="${SI_MODE:-confirmatory}"
if [[ "$SI_MODE" != "confirmatory" && "$SI_MODE" != "development" ]]; then
    echo "SI_MODE must be confirmatory or development" >&2
    exit 2
fi

cd "$SI_REPO_ROOT"
read -r embed_count task_count < <(
    .venv/bin/python - "$SI_MANIFEST" "$SI_CONFIG" "$SI_MODE" <<'PY'
import json
import sys

import yaml

manifest_path, config_path, mode = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
with open(config_path, encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
selected = manifest["selected_assays"]
confirmatory = manifest["confirmatory_ids"]
seeds = config["seeds"]
if len(selected) != len(confirmatory) + 1 or not seeds:
    raise SystemExit("manifest/config cardinalities are invalid")
embed_count = len(selected)
task_count = (len(confirmatory) if mode == "confirmatory" else 1) * len(seeds)
print(embed_count, task_count)
PY
)
if ! [[ "$embed_count" =~ ^[1-9][0-9]*$ && "$task_count" =~ ^[1-9][0-9]*$ ]]; then
    echo "derived array cardinalities are invalid" >&2
    exit 2
fi

run_id="${SI_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_root="${SI_REPO_ROOT}/local/slurm"
run_dir="${run_root}/${run_id}"
mkdir -p "$run_root"
mkdir "$run_dir"

prepare_job="$(sbatch --parsable \
    --account="$SI_ACCOUNT" \
    --partition="$SI_CPU_PARTITION" \
    slurm/prepare.sbatch)"
embed_job="$(sbatch --parsable \
    --account="$SI_ACCOUNT" \
    --partition="$SI_GPU_PARTITION" \
    --array=0-$((embed_count - 1)) \
    --dependency=afterok:"$prepare_job" \
    slurm/embed_array.sbatch)"
task_job="$(sbatch --parsable \
    --account="$SI_ACCOUNT" \
    --partition="$SI_CPU_PARTITION" \
    --array=0-$((task_count - 1)) \
    --dependency=afterok:"$embed_job" \
    slurm/task_array.sbatch)"
aggregate_job="$(sbatch --parsable \
    --account="$SI_ACCOUNT" \
    --partition="$SI_CPU_PARTITION" \
    --dependency=afterok:"$task_job" \
    slurm/aggregate.sbatch)"

job_manifest="$run_dir/job_ids.json"
temporary="$run_dir/.job_ids.json.tmp"
printf '{\n  "aggregate": "%s",\n  "embed": "%s",\n  "prepare": "%s",\n  "task": "%s"\n}\n' \
    "$aggregate_job" "$embed_job" "$prepare_job" "$task_job" >"$temporary"
mv "$temporary" "$job_manifest"
printf '%s\n' "$job_manifest"
