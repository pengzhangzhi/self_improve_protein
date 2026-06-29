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
export SLURM_CONF="$SI_SLURM_CONF"
export OPENBLAS_CORETYPE=Haswell
export SI_CONFIG="${SI_CONFIG:-${SI_REPO_ROOT}/configs/v0.yaml}"
export SI_MANIFEST="${SI_MANIFEST:-${SI_ARTIFACT_ROOT}/studies/v0/data_manifest.json}"
export SI_PROCESSED_ROOT="${SI_PROCESSED_ROOT:-${SI_DATA_ROOT}/processed/v0}"
export SI_EMBEDDING_ROOT="${SI_EMBEDDING_ROOT:-${SI_DATA_ROOT}/embeddings/v0}"
export SI_RESULTS_ROOT="${SI_RESULTS_ROOT:-${SI_ARTIFACT_ROOT}/studies/v0}"
export SI_MODE="${SI_MODE:-development}"
if [[ "$SI_MODE" != "confirmatory" && "$SI_MODE" != "development" ]]; then
    echo "SI_MODE must be confirmatory or development" >&2
    exit 2
fi

cd "$SI_REPO_ROOT"
read -r embed_count task_count < <(
    .venv/bin/python - "$SI_CONFIG" "$SI_MODE" <<'PY'
import sys

from self_improve_protein.config import load_protocol
from self_improve_protein.experiment import canonical_protocol_digest

config_path, mode = sys.argv[1:]
protocol = load_protocol(config_path)
locked = "0b2a74ff76b8c7c508ceea16b004a1c128ba15704138138d49b2c153bcbfa49a"
if canonical_protocol_digest(protocol) != locked:
    raise SystemExit("SI_CONFIG must have the locked v0 protocol digest")
if protocol.seeds[:2] != (0, 1):
    raise SystemExit("locked development seeds must begin with 0,1")
embed_count = protocol.assay_count + 1
task_count = 2 if mode == "development" else protocol.assay_count * len(protocol.seeds)
print(embed_count, task_count)
PY
)
if ! [[ "$embed_count" =~ ^[1-9][0-9]*$ && "$task_count" =~ ^[1-9][0-9]*$ ]]; then
    echo "derived array cardinalities are invalid" >&2
    exit 2
fi

if [[ "$SI_MODE" == "confirmatory" ]]; then
    : "${SI_R5_GATE:?confirmatory submission requires SI_R5_GATE}"
    export SI_R5_GATE
    .venv/bin/self-improve-protein --config "$SI_CONFIG" verify \
        --manifest "$SI_MANIFEST" \
        --processed-root "$SI_PROCESSED_ROOT" \
        --embedding-root "$SI_EMBEDDING_ROOT" \
        --results-root "$SI_RESULTS_ROOT" \
        --r5-gate "$SI_R5_GATE"
fi

run_id="${SI_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_root="${SI_REPO_ROOT}/local/slurm"
run_dir="${run_root}/${run_id}"
mkdir -p "$run_root"
mkdir "$run_dir"

prepare_job=""
embed_job=""
task_job=""
aggregate_job=""
job_manifest="$run_dir/job_ids.json"
temporary="$run_dir/.job_ids.json.tmp"

json_job_id() {
    local value="$1"
    if [[ -z "$value" ]]; then
        printf 'null'
    else
        printf '"%s"' "$value"
    fi
}

write_job_manifest() {
    {
        printf '{\n  "jobs": {\n    "aggregate": '
        json_job_id "$aggregate_job"
        printf ',\n    "embed": '
        json_job_id "$embed_job"
        printf ',\n    "prepare": '
        json_job_id "$prepare_job"
        printf ',\n    "task": '
        json_job_id "$task_job"
        printf '\n  },\n  "mode": "%s",\n  "schema_version": 1\n}\n' "$SI_MODE"
    } >"$temporary"
    mv "$temporary" "$job_manifest"
}

submit_job() {
    local raw job_id
    raw="$(sbatch --parsable "$@")"
    if [[ ! "$raw" =~ ^[0-9]+(\;[A-Za-z0-9._-]+)?$ ]]; then
        echo "sbatch returned an invalid parsable job ID" >&2
        return 2
    fi
    job_id="${raw%%;*}"
    printf '%s' "$job_id"
}

prepare_job="$(submit_job \
    --account="$SI_ACCOUNT" \
    --partition="$SI_CPU_PARTITION" \
    slurm/prepare.sbatch)"
write_job_manifest

embed_job="$(submit_job \
    --account="$SI_ACCOUNT" \
    --partition="$SI_GPU_PARTITION" \
    --array=0-$((embed_count - 1)) \
    --dependency=afterok:"$prepare_job" \
    slurm/embed_array.sbatch)"
write_job_manifest

task_job="$(submit_job \
    --account="$SI_ACCOUNT" \
    --partition="$SI_CPU_PARTITION" \
    --array=0-$((task_count - 1)) \
    --dependency=afterok:"$embed_job" \
    slurm/task_array.sbatch)"
write_job_manifest

aggregate_job="$(submit_job \
    --account="$SI_ACCOUNT" \
    --partition="$SI_CPU_PARTITION" \
    --dependency=afterok:"$task_job" \
    slurm/aggregate.sbatch)"
write_job_manifest

printf '%s\n' "$job_manifest"
